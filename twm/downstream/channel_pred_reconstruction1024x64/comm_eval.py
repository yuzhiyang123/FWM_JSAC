from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class NonFAEvalOutput:
    ser: float
    num_errors: int
    num_symbols: int


@dataclass
class FABeamformingOutput:
    avg_rx_power: float
    oracle_avg_rx_power: float
    power_ratio: float
    mean_effective_channel_power: float
    beamformer: Tensor


def _ri_to_complex(x: Tensor) -> Tensor:
    if torch.is_complex(x):
        return x
    if x.ndim < 3:
        raise ValueError(f"Expected at least 3 dims for real/imag tensor, got {tuple(x.shape)}")
    if x.shape[-3] != 2:
        raise ValueError(f"Expected real tensor with channel dim 2 at -3, got {tuple(x.shape)}")
    return torch.complex(x[..., 0, :, :], x[..., 1, :, :])


def _constellation(modulation: str, device: torch.device, dtype: torch.dtype) -> Tensor:
    mod = str(modulation).strip().lower()
    if mod == 'qpsk':
        vals = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=device, dtype=torch.complex64) / (2.0 ** 0.5)
    elif mod in {'16qam', '16-qam', '16_qam'}:
        levels = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=device, dtype=torch.float32) / (10.0 ** 0.5)
        vals = torch.combinations(levels, r=2, with_replacement=False)
        grid = torch.cartesian_prod(levels, levels)
        vals = torch.complex(grid[:, 0], grid[:, 1]).to(torch.complex64)
    else:
        raise ValueError(f"Unsupported modulation: {modulation!r}")
    if dtype == torch.complex128:
        vals = vals.to(torch.complex128)
    return vals


def _sample_symbols(shape: tuple[int, ...], modulation: str, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    const = _constellation(modulation, device=device, dtype=dtype)
    idx = torch.randint(low=0, high=const.numel(), size=shape, device=device)
    return const[idx]


def _hard_detect(received: Tensor, modulation: str) -> Tensor:
    const = _constellation(modulation, device=received.device, dtype=received.dtype)
    dist = (received.unsqueeze(-1) - const.view(*([1] * received.ndim), -1)).abs().square()
    return const[dist.argmin(dim=-1)]


def evaluate_non_fa_ser(
    reconstructed_channel: Tensor,
    target_channel: Tensor,
    *,
    modulation: str = 'qpsk',
    noise_var: float = 0.0,
    data_mask: Tensor | None = None,
    equalizer: str = 'zf',
) -> NonFAEvalOutput:
    """Evaluate uncoded SER with single-stream SIMO equalization from reconstructed CSI.

    Both channel tensors can be complex ``[..., K, M]`` or real/imag ``[..., 2, K, M]``.
    We synthesize uncoded symbols on selected resource elements, generate
    ``y_k = h_k s_k + z_k`` across the 64 receive antennas, and use the
    reconstructed channel for ZF/LMMSE equalization and hard detection.
    """
    h_hat = _ri_to_complex(reconstructed_channel)
    h_true = _ri_to_complex(target_channel)
    if h_hat.shape != h_true.shape:
        raise ValueError(f"Expected reconstructed/target shapes to match, got {tuple(h_hat.shape)} vs {tuple(h_true.shape)}")
    if h_hat.ndim < 2:
        raise ValueError(f"Expected channel shape [..., K, M], got {tuple(h_hat.shape)}")
    *prefix, num_sc, num_ant = h_hat.shape
    symbols = _sample_symbols(tuple(prefix) + (num_sc,), modulation, device=h_hat.device, dtype=h_hat.dtype)
    if noise_var > 0.0:
        sigma = (float(noise_var) / 2.0) ** 0.5
        noise = sigma * (torch.randn(*prefix, num_sc, num_ant, device=h_hat.device, dtype=h_hat.real.dtype) + 1j * torch.randn(*prefix, num_sc, num_ant, device=h_hat.device, dtype=h_hat.real.dtype))
        noise = noise.to(h_hat.dtype)
    else:
        noise = torch.zeros(*prefix, num_sc, num_ant, device=h_hat.device, dtype=h_hat.dtype)
    y = h_true * symbols.unsqueeze(-1) + noise
    denom = h_hat.abs().square().sum(dim=-1)
    mode = str(equalizer).strip().lower()
    if mode == 'zf':
        gains = denom.clamp_min(1e-8)
    elif mode == 'lmmse':
        gains = (denom + float(noise_var)).clamp_min(1e-8)
    else:
        raise ValueError(f"Unsupported equalizer: {equalizer!r}")
    detected_soft = (h_hat.conj() * y).sum(dim=-1) / gains
    detected_hard = _hard_detect(detected_soft, modulation)
    symbol_errors = detected_hard != symbols
    if data_mask is not None:
        mask = data_mask.to(device=symbol_errors.device, dtype=torch.bool)
        if mask.shape != symbol_errors.shape:
            raise ValueError(f"Expected data_mask shape {tuple(symbol_errors.shape)}, got {tuple(mask.shape)}")
        symbol_errors = symbol_errors & mask
        num_symbols = int(mask.sum().item())
        num_errors = int(symbol_errors.sum().item())
    else:
        num_symbols = int(symbol_errors.numel())
        num_errors = int(symbol_errors.sum().item())
    ser = float(num_errors / max(1, num_symbols))
    return NonFAEvalOutput(ser=ser, num_errors=num_errors, num_symbols=num_symbols)


def design_rb_beamformer(reconstructed_channel: Tensor) -> Tensor:
    """Return the unit-norm principal-eigenvector beamformer from reconstructed CSI.

    Input can be complex ``[..., K, M]`` or real/imag ``[..., 2, K, M]`` and is interpreted as one RB.
    """
    h_hat = _ri_to_complex(reconstructed_channel)
    if h_hat.ndim < 2:
        raise ValueError(f"Expected channel shape [..., K, M], got {tuple(h_hat.shape)}")
    gram = h_hat.conj().transpose(-1, -2) @ h_hat
    eigvals, eigvecs = torch.linalg.eigh(gram)
    w = eigvecs[..., -1]
    return w / w.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _select_masked_subcarriers(channel: Tensor, sc_mask: Tensor | None) -> Tensor:
    if sc_mask is None:
        return channel
    mask = sc_mask.to(device=channel.device, dtype=torch.bool)
    if mask.ndim != 1 or mask.shape[0] != channel.shape[-2]:
        raise ValueError(
            f'Expected subcarrier mask shape ({channel.shape[-2]},), got {tuple(mask.shape)}'
        )
    if not bool(mask.any().item()):
        raise ValueError('Expected at least one masked/non-pilot subcarrier for FA evaluation.')
    return channel[..., mask, :]


def evaluate_fa_beamforming(
    reconstructed_channel: Tensor,
    target_channel: Tensor,
    *,
    sc_mask: Tensor | None = None,
) -> FABeamformingOutput:
    """Evaluate RB-wise single-stream beamforming quality on one 1024x64 RB.

    Returns average received power on the true channel using a beamformer built from the
    reconstructed CSI, plus the oracle power from true CSI.
    """
    h_hat = _ri_to_complex(reconstructed_channel)
    h_true = _ri_to_complex(target_channel)
    if h_hat.shape != h_true.shape:
        raise ValueError(f"Expected reconstructed/target shapes to match, got {tuple(h_hat.shape)} vs {tuple(h_true.shape)}")
    h_hat_eval = _select_masked_subcarriers(h_hat, sc_mask)
    h_true_eval = _select_masked_subcarriers(h_true, sc_mask)
    w_hat = design_rb_beamformer(h_hat_eval)
    w_oracle = design_rb_beamformer(h_true_eval)
    eff_hat = (h_true_eval * w_hat.unsqueeze(-2)).sum(dim=-1)
    eff_oracle = (h_true_eval * w_oracle.unsqueeze(-2)).sum(dim=-1)
    avg_rx_power = float(eff_hat.abs().square().mean().item())
    oracle_avg_rx_power = float(eff_oracle.abs().square().mean().item())
    power_ratio = float(avg_rx_power / max(oracle_avg_rx_power, 1e-12))
    mean_effective_channel_power = float(h_hat_eval.abs().square().mean().item())
    return FABeamformingOutput(
        avg_rx_power=avg_rx_power,
        oracle_avg_rx_power=oracle_avg_rx_power,
        power_ratio=power_ratio,
        mean_effective_channel_power=mean_effective_channel_power,
        beamformer=w_hat,
    )
