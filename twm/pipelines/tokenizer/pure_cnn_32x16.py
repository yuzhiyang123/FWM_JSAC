from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from .base import AbstractCellTokenizer
from .cnn_blocks import DownBlock, MidBlock

__all__ = ["build_32x16"]


@dataclass
class PureCNNCompressor32x16Config:
    num_tokens: int
    token_dim: int
    output_norm: str = 'batchnorm'
    down_channels: tuple[int, ...] = (32, 64, 128, 128)
    mid_channels: tuple[int, ...] = (128, 128)
    down_sample: tuple[bool, ...] = (True, True, True)
    use_downsample_conv: tuple[bool, ...] = (True, True, True)
    num_down_layers: int = 1
    num_mid_layers: int = 1
    attn_down: tuple[bool, ...] = (False, False, False)
    norm_channels: int = 8
    num_heads: int = 4


class PureCNNCompressor32x16(AbstractCellTokenizer):
    """Pure CNN tokenizer for fixed input shape [M, 2, 32, 16].

    The encoder follows the copied VAE encoder style. Its final convolution emits
    `Nt` channels directly, and each output channel is treated as one token map.
    A shared two-layer MLP maps each flattened channel map to one token of length `Lt`.
    """

    input_hw = (32, 16)

    def __init__(self, config: PureCNNCompressor32x16Config) -> None:
        super().__init__(num_tokens=config.num_tokens, token_dim=config.token_dim, output_norm=config.output_norm)
        self.config = config
        self._validate_config()

        down_channels = tuple(int(v) for v in config.down_channels)
        mid_channels = tuple(int(v) for v in config.mid_channels)

        self.encoder_conv_in = nn.Conv2d(2, down_channels[0], kernel_size=3, padding=1)
        self.encoder_layers = nn.ModuleList([
            DownBlock(
                down_channels[i],
                down_channels[i + 1],
                down_sample=config.down_sample[i],
                use_downsample_conv=config.use_downsample_conv[i],
                num_heads=config.num_heads,
                num_layers=config.num_down_layers,
                attn=config.attn_down[i],
                norm_channels=config.norm_channels,
            )
            for i in range(len(down_channels) - 1)
        ])
        self.encoder_mids = nn.ModuleList([
            MidBlock(
                mid_channels[i],
                mid_channels[i + 1],
                num_heads=config.num_heads,
                num_layers=config.num_mid_layers,
                norm_channels=config.norm_channels,
            )
            for i in range(len(mid_channels) - 1)
        ])
        self.encoder_norm_out = nn.GroupNorm(config.norm_channels, down_channels[-1])
        self.encoder_act_out = nn.SiLU()
        self.encoder_conv_out = nn.Conv2d(down_channels[-1], self.Nt, kernel_size=3, padding=1)

        latent_hw = self._infer_latent_hw()
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(latent_hw),
            nn.Linear(latent_hw, 2 * self.Lt),
            nn.GELU(),
            nn.Linear(2 * self.Lt, self.Lt),
        )

    def _validate_config(self) -> None:
        if self.config.mid_channels[0] != self.config.down_channels[-1]:
            raise ValueError('mid_channels[0] must match down_channels[-1]')
        if self.config.mid_channels[-1] != self.config.down_channels[-1]:
            raise ValueError('mid_channels[-1] must match down_channels[-1]')
        if len(self.config.down_sample) != len(self.config.down_channels) - 1:
            raise ValueError('down_sample length must equal len(down_channels) - 1')
        if len(self.config.attn_down) != len(self.config.down_channels) - 1:
            raise ValueError('attn_down length must equal len(down_channels) - 1')
        if len(self.config.use_downsample_conv) != len(self.config.down_channels) - 1:
            raise ValueError('use_downsample_conv length must equal len(down_channels) - 1')

    def _encode_conv_maps(self, x: Tensor) -> Tensor:
        out = self.encoder_conv_in(x)
        for down in self.encoder_layers:
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = self.encoder_act_out(out)
        out = self.encoder_conv_out(out)
        return out

    def _infer_latent_hw(self) -> int:
        with torch.no_grad():
            probe = torch.zeros(1, 2, *self.input_hw)
            out = self._encode_conv_maps(probe)
        return int(out.shape[-2] * out.shape[-1])

    def validate_realization_input(self, x: Tensor) -> None:
        if tuple(x.shape[-2:]) != self.input_hw:
            raise ValueError(f'PureCNNCompressor32x16 expects H,W={self.input_hw}, got {tuple(x.shape[-2:])}')

    def forward_tokens(self, x: Tensor) -> Tensor:
        out = self._encode_conv_maps(x)
        batch_size = out.shape[0]
        token_inputs = out.reshape(batch_size, self.Nt, -1)
        tokens = self.token_mlp(token_inputs)
        return tokens


config_S = PureCNNCompressor32x16Config(
    num_tokens=1,
    token_dim=1,
    down_channels=(64, 128),
    mid_channels=(128,),
    down_sample=(True,),
    use_downsample_conv=(True,),
    num_mid_layers=2,
    attn_down=(False,),
    norm_channels=8,
)

config_M = PureCNNCompressor32x16Config(
    num_tokens=1,
    token_dim=1,
    down_channels=(32, 64, 128),
    mid_channels=(128,),
    down_sample=(False, True),
    use_downsample_conv=(False, True),
    num_mid_layers=2,
    attn_down=(False, False),
    norm_channels=8,
)

config_A = PureCNNCompressor32x16Config(
    num_tokens=1,
    token_dim=1,
    down_channels=(32, 64, 128),
    mid_channels=(128,),
    down_sample=(False, True),
    use_downsample_conv=(False, True),
    num_mid_layers=2,
    attn_down=(True, True),
    norm_channels=8,
)

config_L = PureCNNCompressor32x16Config(
    num_tokens=1,
    token_dim=1,
    down_channels=(64, 128, 256),
    mid_channels=(256, 256),
    down_sample=(False, True),
    use_downsample_conv=(False, True),
    num_mid_layers=2,
    attn_down=(False, False),
    norm_channels=16,
)

def build_32x16(
    num_tokens: int,
    token_dim: int,
    config_name: str,
    output_norm: str = 'batchnorm',
) -> PureCNNCompressor32x16:
    key = str(config_name).strip().lower()
    if key in {'s', 'small'}:
        C = config_S
    elif key in {'m', 'medium'}:
        C = config_M
    elif key in {'l', 'large'}:
        C = config_L
    elif key in {'a', 'attention'}:
        C = config_A
    else:
        raise ValueError('Unknown config_name. Expected one of: S/small, M/medium, A/attention, L/large')
    return PureCNNCompressor32x16(
        replace(C, num_tokens=int(num_tokens), token_dim=int(token_dim), output_norm=str(output_norm))
    )
