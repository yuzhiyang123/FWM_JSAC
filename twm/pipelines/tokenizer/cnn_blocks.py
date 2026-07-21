from __future__ import annotations

from torch import Tensor, nn


class DownBlock(nn.Module):
    """ResNet-style down block copied into `twm` for tokenizer use."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        down_sample: bool,
        use_downsample_conv: bool,
        num_heads: int,
        num_layers: int,
        attn: bool,
        norm_channels: int,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.down_sample = bool(down_sample)
        self.use_downsample_conv = bool(use_downsample_conv)
        self.attn = bool(attn)

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

        if self.attn:
            self.attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(self.num_layers)
            ])
            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(self.num_layers)
            ])

        self.down_sample_conv = (
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if self.down_sample and self.use_downsample_conv
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for i in range(self.num_layers):
            residual = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](residual)

            if self.attn:
                batch_size, channels, height, width = out.shape
                tokens = out.reshape(batch_size, channels, height * width)
                tokens = self.attention_norms[i](tokens)
                tokens = tokens.transpose(1, 2)
                attn_out, _ = self.attentions[i](tokens, tokens, tokens)
                attn_out = attn_out.transpose(1, 2).reshape(batch_size, channels, height, width)
                out = out + attn_out

        return self.down_sample_conv(out)


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_heads: int,
        num_layers: int,
        norm_channels: int,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for i in range(self.num_layers + 1)
        ])
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for _ in range(self.num_layers + 1)
        ])
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(self.num_layers + 1)
        ])
        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(norm_channels, out_channels)
            for _ in range(self.num_layers)
        ])
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(self.num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        out = x
        residual = out
        out = self.resnet_conv_first[0](out)
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](residual)

        for i in range(self.num_layers):
            batch_size, channels, height, width = out.shape
            tokens = out.reshape(batch_size, channels, height * width)
            tokens = self.attention_norms[i](tokens)
            tokens = tokens.transpose(1, 2)
            attn_out, _ = self.attentions[i](tokens, tokens, tokens)
            attn_out = attn_out.transpose(1, 2).reshape(batch_size, channels, height, width)
            out = out + attn_out

            residual = out
            out = self.resnet_conv_first[i + 1](out)
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](residual)

        return out
