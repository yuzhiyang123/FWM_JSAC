from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from twm.pipelines.tokenizer.cnn_blocks import MidBlock


@dataclass
class RecoveryHeadOutput:
    recovered: Tensor


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        up_sample: bool,
        num_layers: int = 1,
        norm_channels: int = 16,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.up_sample = bool(up_sample)
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for i in range(self.num_layers)
        ])
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for _ in range(self.num_layers)
        ])
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(self.num_layers)
        ])
        self.up_sample_conv = (
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if self.up_sample
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for i in range(self.num_layers):
            residual = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](residual)
        return self.up_sample_conv(out)


class SymmetricRecoveryDecoder32x16Large(nn.Module):
    """Decoder mirrored from the 32x16 large tokenizer encoder.

    It expects tokenizer outputs with shape [M, Nt, Lt] and reconstructs one
    preprocessed channel with shape [M, 2, 32, 16].
    """

    latent_hw = (16, 8)

    def __init__(self, *, num_tokens: int, token_dim: int) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        latent_area = int(self.latent_hw[0] * self.latent_hw[1])
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 2 * latent_area),
            nn.GELU(),
            nn.Linear(2 * latent_area, latent_area),
        )
        self.decoder_conv_in = nn.Conv2d(self.num_tokens, 256, kernel_size=3, padding=1)
        self.decoder_mids = nn.ModuleList([
            MidBlock(256, 256, num_heads=4, num_layers=2, norm_channels=16),
        ])
        self.decoder_layers = nn.ModuleList([
            UpBlock(256, 128, up_sample=True, num_layers=1, norm_channels=16),
            UpBlock(128, 64, up_sample=False, num_layers=1, norm_channels=16),
        ])
        self.decoder_norm_out = nn.GroupNorm(16, 64)
        self.decoder_act_out = nn.SiLU()
        self.decoder_conv_out = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, tokens: Tensor) -> RecoveryHeadOutput:
        if tokens.ndim != 3:
            raise ValueError(f'Expected token shape [M, Nt, Lt], got {tuple(tokens.shape)}')
        if tokens.shape[1] != self.num_tokens or tokens.shape[2] != self.token_dim:
            raise ValueError(
                f'Expected token shape [M, {self.num_tokens}, {self.token_dim}], got {tuple(tokens.shape)}'
            )
        batch_size = tokens.shape[0]
        maps = self.token_mlp(tokens).view(batch_size, self.num_tokens, *self.latent_hw)
        out = self.decoder_conv_in(maps)
        for mid in self.decoder_mids:
            out = mid(out)
        for up in self.decoder_layers:
            out = up(out)
        out = self.decoder_norm_out(out)
        out = self.decoder_act_out(out)
        out = self.decoder_conv_out(out)
        return RecoveryHeadOutput(recovered=out)


class SymmetricRecoveryDecoder128x64Large(nn.Module):
    """Recovery decoder for 128x64 student preprocessors.

    It expects tokenizer outputs with shape [M, Nt, Lt] and reconstructs one
    preprocessed channel with shape [M, 2, 128, 64].
    """

    latent_hw = (16, 8)

    def __init__(self, *, num_tokens: int, token_dim: int) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        latent_area = int(self.latent_hw[0] * self.latent_hw[1])
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 4 * latent_area),
            nn.GELU(),
            nn.Linear(4 * latent_area, latent_area),
        )
        self.decoder_conv_in = nn.Conv2d(self.num_tokens, 256, kernel_size=3, padding=1)
        self.decoder_mids = nn.ModuleList([
            MidBlock(256, 256, num_heads=4, num_layers=2, norm_channels=16),
            MidBlock(256, 256, num_heads=4, num_layers=2, norm_channels=16),
        ])
        self.decoder_layers = nn.ModuleList([
            UpBlock(256, 192, up_sample=True, num_layers=1, norm_channels=16),
            UpBlock(192, 128, up_sample=True, num_layers=1, norm_channels=16),
            UpBlock(128, 64, up_sample=True, num_layers=1, norm_channels=16),
            UpBlock(64, 64, up_sample=False, num_layers=1, norm_channels=16),
        ])
        self.decoder_norm_out = nn.GroupNorm(16, 64)
        self.decoder_act_out = nn.SiLU()
        self.decoder_conv_out = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, tokens: Tensor) -> RecoveryHeadOutput:
        if tokens.ndim != 3:
            raise ValueError(f'Expected token shape [M, Nt, Lt], got {tuple(tokens.shape)}')
        if tokens.shape[1] != self.num_tokens or tokens.shape[2] != self.token_dim:
            raise ValueError(
                f'Expected token shape [M, {self.num_tokens}, {self.token_dim}], got {tuple(tokens.shape)}'
            )
        batch_size = tokens.shape[0]
        maps = self.token_mlp(tokens).view(batch_size, self.num_tokens, *self.latent_hw)
        out = self.decoder_conv_in(maps)
        for mid in self.decoder_mids:
            out = mid(out)
        for up in self.decoder_layers:
            out = up(out)
        out = self.decoder_norm_out(out)
        out = self.decoder_act_out(out)
        out = self.decoder_conv_out(out)
        return RecoveryHeadOutput(recovered=out)
