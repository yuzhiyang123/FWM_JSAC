from __future__ import annotations

from .pipeline import TimeBandJEPAConfig, TimeBandViTConfig


_TOKENIZER_MAP = {
    'small': ('32x16_s', '32x16_s'),
    'medium': ('32x16_m', '32x16_m'),
    'attn': ('32x16_a', '32x16_a'),
    'large': ('32x16_l', '32x16_l'),
}

_LAYER_MAP = {
    'shallow': {'depth': 2, 'num_heads': 4},
    'medium': {'depth': 4, 'num_heads': 8},
    'deep': {'depth': 6, 'num_heads': 8},
}


def build_lazy_config(
    preset_name: str,
    *,
    num_low_bands: int = 5,
    num_tokens: int = 4,
    token_dim: int = 192,
    predictor_dim: int = 192,
    tokenizer_output_norm: str = 'layernorm',
    query_init_std: float = 0.02,
    encoder_mlp_ratio: float = 4.0,
    encoder_dropout: float = 0.0,
    predictor_mlp_ratio: float = 4.0,
    predictor_dropout: float = 0.0,
) -> TimeBandJEPAConfig:
    parts = str(preset_name).strip().lower().split('_')
    if len(parts) != 4:
        raise ValueError(
            'preset_name must look like 32x16_small_shallow_shallow: '
            '[tokenizer_family]_[tokenizer_size]_[encoder_size]_[predictor_size]'
        )

    tokenizer_family, tokenizer_size, encoder_size, predictor_size = parts
    if tokenizer_family != '32x16':
        raise ValueError('Only tokenizer family 32x16 is currently supported')
    if tokenizer_size not in _TOKENIZER_MAP:
        raise ValueError('Unknown tokenizer_size. Expected one of: small, medium, attn, large')
    if encoder_size not in _LAYER_MAP:
        raise ValueError('Unknown encoder_size. Expected one of: shallow, medium, deep')
    if predictor_size not in _LAYER_MAP:
        raise ValueError('Unknown predictor_size. Expected one of: shallow, medium, deep')

    tokenizer_name_3p5, tokenizer_name_28 = _TOKENIZER_MAP[tokenizer_size]
    encoder_spec = _LAYER_MAP[encoder_size]
    predictor_spec = _LAYER_MAP[predictor_size]

    return TimeBandJEPAConfig(
        num_low_bands=int(num_low_bands),
        predictor_dim=int(predictor_dim),
        query_init_std=float(query_init_std),
        preprocessor_name='uniform_grid_32x16_quadrant',
        preprocessor_kwargs={},
        tokenizer_name_3p5=tokenizer_name_3p5,
        tokenizer_name_28=tokenizer_name_28,
        num_tokens=int(num_tokens),
        token_dim=int(token_dim),
        tokenizer_output_norm=str(tokenizer_output_norm),
        encoder=TimeBandViTConfig(
            depth=int(encoder_spec['depth']),
            num_heads=int(encoder_spec['num_heads']),
            mlp_ratio=float(encoder_mlp_ratio),
            dropout=float(encoder_dropout),
        ),
        predictor=TimeBandViTConfig(
            depth=int(predictor_spec['depth']),
            num_heads=int(predictor_spec['num_heads']),
            mlp_ratio=float(predictor_mlp_ratio),
            dropout=float(predictor_dropout),
        ),
    )


__all__ = ['build_lazy_config']
