from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..dit32x16 import DiT32x16Output, TokenConditionalDiT32x16, _BaseDiTDiffusionModel32x16


@dataclass
class RecoveryDiffusionOutput:
    predicted_x0: Tensor

    @property
    def predicted_noise(self) -> Tensor:
        return self.predicted_x0


class DiffusersConditionalRecoveryModel32x16(TokenConditionalDiT32x16):
    def __init__(
        self,
        *,
        token_dim: int,
        down_block_out_channels: tuple[int, ...] = (64, 128, 128),
        layers_per_block: int = 1,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        hidden_dim = int(max(down_block_out_channels)) if down_block_out_channels else 128
        depth = max(3, len(tuple(down_block_out_channels)) * max(1, int(layers_per_block)))
        super().__init__(
            token_dim=int(token_dim),
            sample_channels=2,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=8,
            mlp_ratio=4.0,
            num_train_timesteps=int(num_train_timesteps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
        )

    def forward(self, noisy_target: Tensor, tokens: Tensor, timesteps: Tensor) -> RecoveryDiffusionOutput:
        out: DiT32x16Output = super().forward(noisy_target, tokens, timesteps)
        return RecoveryDiffusionOutput(predicted_x0=out.predicted_x0)


class DiffusersUnconditionalRecoveryModel32x16(_BaseDiTDiffusionModel32x16):
    def __init__(
        self,
        *,
        down_block_out_channels: tuple[int, ...] = (64, 128, 128),
        layers_per_block: int = 1,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__(
            num_train_timesteps=int(num_train_timesteps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
        )
        hidden_dim = int(max(down_block_out_channels)) if down_block_out_channels else 128
        depth = max(3, len(tuple(down_block_out_channels)) * max(1, int(layers_per_block)))
        from ..dit32x16 import _FinalLayer, _SinusoidalTimeEmbedding, _ContextCrossBlock
        from torch import nn
        self.sample_channels = 2
        self.height = 32
        self.width = 16
        self.hidden_dim = hidden_dim
        self.x_embed = nn.Conv2d(self.sample_channels, self.hidden_dim, kernel_size=1, bias=True)
        self.time_embed = nn.Sequential(_SinusoidalTimeEmbedding(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.pos_embed = nn.Parameter(__import__('torch').zeros(1, self.height * self.width, self.hidden_dim))
        self.blocks = nn.ModuleList([_ContextCrossBlock(self.hidden_dim, num_heads=8, mlp_ratio=4.0, num_contexts=0) for _ in range(depth)])
        self.final_layer = _FinalLayer(self.hidden_dim, self.sample_channels)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        from torch import nn
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, noisy_target: Tensor, timesteps: Tensor) -> RecoveryDiffusionOutput:
        if noisy_target.ndim != 4 or tuple(noisy_target.shape[1:]) != (2, 32, 16):
            raise ValueError(f'Expected noisy_target shape [B, 2, 32, 16], got {tuple(noisy_target.shape)}')
        x_tokens = self.x_embed(noisy_target).flatten(2).transpose(1, 2) + self.pos_embed
        cond = self.time_embed(timesteps)
        for block in self.blocks:
            x_tokens = block(x_tokens, cond)
        predicted_x0 = self.final_layer(x_tokens, cond).transpose(1, 2).reshape(noisy_target.shape[0], 2, 32, 16)
        return RecoveryDiffusionOutput(predicted_x0=predicted_x0)
