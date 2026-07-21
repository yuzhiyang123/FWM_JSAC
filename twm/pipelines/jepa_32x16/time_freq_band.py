from __future__ import annotations

import torch
from torch import Tensor, nn


class AbstractMaskGenerator(nn.Module):
    """Build a boolean mask over the time-band grid.

    Expected output shape: [B, T, N], where True means the cell is masked.
    """

    def forward(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        raise NotImplementedError


    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        return None


class RandomMaskGenerator(AbstractMaskGenerator):
    def __init__(self, mask_rate: float) -> None:
        super().__init__()
        if not (0.0 <= float(mask_rate) <= 1.0):
            raise ValueError(f'mask_rate must be in [0, 1], got {mask_rate}')
        self.mask_rate = float(mask_rate)

    def forward(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if x.ndim < 3:
            raise ValueError(f'Expected input with at least 3 dims, got {tuple(x.shape)}')
        batch_size, time_steps, num_bands = x.shape[:3]
        mask = torch.rand(batch_size, time_steps, num_bands, device=x.device) < self.mask_rate
        if valid_mask is not None:
            mask = mask & valid_mask.to(device=x.device, dtype=torch.bool)
        return mask


class RandomMaskExceptLastGenerator(AbstractMaskGenerator):
    def __init__(self, mask_rate: float) -> None:
        super().__init__()
        if not (0.0 <= float(mask_rate) <= 1.0):
            raise ValueError(f'mask_rate must be in [0, 1], got {mask_rate}')
        self.mask_rate = float(mask_rate)

    def forward(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if x.ndim < 3:
            raise ValueError(f'Expected input with at least 3 dims, got {tuple(x.shape)}')
        batch_size, time_steps, num_bands = x.shape[:3]
        mask = torch.zeros(batch_size, time_steps, num_bands, device=x.device, dtype=torch.bool)
        if time_steps > 1:
            mask[:, :-1, :] = torch.rand(batch_size, time_steps - 1, num_bands, device=x.device) < self.mask_rate
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
            mask = mask & valid_mask
            mask[:, -1, :] = valid_mask[:, -1, :]
        else:
            mask[:, -1, :] = True
        return mask



class RandomMaskExceptLastVariableRateGenerator(AbstractMaskGenerator):
    def __init__(self, min_mask_rate: float, max_mask_rate: float) -> None:
        super().__init__()
        min_rate = float(min_mask_rate)
        max_rate = float(max_mask_rate)
        if not (0.0 <= min_rate <= 1.0):
            raise ValueError(f'min_mask_rate must be in [0, 1], got {min_mask_rate}')
        if not (0.0 <= max_rate <= 1.0):
            raise ValueError(f'max_mask_rate must be in [0, 1], got {max_mask_rate}')
        if min_rate > max_rate:
            raise ValueError(f'min_mask_rate must be <= max_mask_rate, got {min_mask_rate} > {max_mask_rate}')
        self.min_mask_rate = min_rate
        self.max_mask_rate = max_rate
        self.last_mask_rate = min_rate

    def _sample_mask_rate(self, device: torch.device) -> float:
        if self.min_mask_rate == self.max_mask_rate:
            return self.min_mask_rate
        sampled = torch.empty((), device=device).uniform_(self.min_mask_rate, self.max_mask_rate)
        return float(sampled.item())

    def forward(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if x.ndim < 3:
            raise ValueError(f'Expected input with at least 3 dims, got {tuple(x.shape)}')
        batch_size, time_steps, num_bands = x.shape[:3]
        mask = torch.zeros(batch_size, time_steps, num_bands, device=x.device, dtype=torch.bool)
        sampled_rate = self._sample_mask_rate(x.device)
        self.last_mask_rate = sampled_rate
        if time_steps > 1:
            mask[:, :-1, :] = torch.rand(batch_size, time_steps - 1, num_bands, device=x.device) < sampled_rate
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
            mask = mask & valid_mask
            mask[:, -1, :] = valid_mask[:, -1, :]
        else:
            mask[:, -1, :] = True
        return mask



class CurriculumRandomMaskExceptLastGenerator(AbstractMaskGenerator):
    def __init__(
        self,
        start_mask_rate: float,
        end_mask_rate: float,
        ramp_epochs: int = 50,
        sample_jitter: float = 0.1,
        final_min_mask_rate: float = 0.1,
        final_max_mask_rate: float = 0.9,
        final_beta_alpha: float = 4.0,
        final_beta_beta: float = 1.2,
    ) -> None:
        super().__init__()
        start_rate = float(start_mask_rate)
        end_rate = float(end_mask_rate)
        jitter = float(sample_jitter)
        ramp = int(ramp_epochs)
        final_min = float(final_min_mask_rate)
        final_max = float(final_max_mask_rate)
        beta_alpha = float(final_beta_alpha)
        beta_beta = float(final_beta_beta)
        if not (0.0 <= start_rate <= 1.0):
            raise ValueError(f'start_mask_rate must be in [0, 1], got {start_mask_rate}')
        if not (0.0 <= end_rate <= 1.0):
            raise ValueError(f'end_mask_rate must be in [0, 1], got {end_mask_rate}')
        if ramp <= 0:
            raise ValueError(f'ramp_epochs must be positive, got {ramp_epochs}')
        if jitter < 0.0:
            raise ValueError(f'sample_jitter must be nonnegative, got {sample_jitter}')
        if not (0.0 <= final_min <= 1.0):
            raise ValueError(f'final_min_mask_rate must be in [0, 1], got {final_min_mask_rate}')
        if not (0.0 <= final_max <= 1.0):
            raise ValueError(f'final_max_mask_rate must be in [0, 1], got {final_max_mask_rate}')
        if final_min > final_max:
            raise ValueError(
                f'final_min_mask_rate must be <= final_max_mask_rate, got {final_min_mask_rate} > {final_max_mask_rate}'
            )
        if beta_alpha <= 0.0 or beta_beta <= 0.0:
            raise ValueError('final_beta_alpha and final_beta_beta must be positive')
        self.start_mask_rate = start_rate
        self.end_mask_rate = end_rate
        self.ramp_epochs = ramp
        self.sample_jitter = jitter
        self.final_min_mask_rate = final_min
        self.final_max_mask_rate = final_max
        self.final_beta_alpha = beta_alpha
        self.final_beta_beta = beta_beta
        self.current_epoch = 1
        self.current_total_epochs = ramp
        self.last_mask_rate = start_rate

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.current_epoch = max(1, int(epoch))
        if total_epochs is not None:
            self.current_total_epochs = max(1, int(total_epochs))

    def _epoch_progress(self) -> float:
        total_epochs = max(1, int(self.current_total_epochs))
        if total_epochs <= 1:
            return 1.0
        clamped_epoch = min(max(self.current_epoch, 1), total_epochs)
        return float(clamped_epoch - 1) / float(total_epochs - 1)

    def _sample_final_rates(self, batch_size: int, device: torch.device) -> Tensor:
        beta_dist = torch.distributions.Beta(self.final_beta_alpha, self.final_beta_beta)
        scaled = beta_dist.sample((batch_size,)).to(device=device)
        return self.final_min_mask_rate + scaled * (self.final_max_mask_rate - self.final_min_mask_rate)

    def _sample_rates(self, batch_size: int, device: torch.device) -> Tensor:
        progress = self._epoch_progress()
        final_rates = self._sample_final_rates(batch_size, device)
        base_rates = torch.full((batch_size,), self.start_mask_rate, device=device)
        blended = torch.lerp(base_rates, final_rates, progress)
        return blended.clamp_(0.0, 1.0)

    def forward(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if x.ndim < 3:
            raise ValueError(f'Expected input with at least 3 dims, got {tuple(x.shape)}')
        batch_size, time_steps, num_bands = x.shape[:3]
        mask = torch.zeros(batch_size, time_steps, num_bands, device=x.device, dtype=torch.bool)
        per_sample_rates = self._sample_rates(batch_size, x.device)
        self.last_mask_rate = float(per_sample_rates.mean().item())
        if time_steps > 1:
            random_values = torch.rand(batch_size, time_steps - 1, num_bands, device=x.device)
            mask[:, :-1, :] = random_values < per_sample_rates[:, None, None]
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
            mask = mask & valid_mask
            mask[:, -1, :] = valid_mask[:, -1, :]
        else:
            mask[:, -1, :] = True
        return mask
