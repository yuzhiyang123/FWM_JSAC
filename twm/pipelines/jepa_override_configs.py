from __future__ import annotations

from dataclasses import asdict, dataclass, field

from twm.pipelines.jepa_32x16 import BasicJEPATrainingConfig, JEPATrainingConfig, NamedJEPAConfig


@dataclass
class BasicTrainingConfig(BasicJEPATrainingConfig):
    jepa_config: NamedJEPAConfig = field(default_factory=NamedJEPAConfig)
    jepa_loss_weight: float = 1.0
    remained_nt_vicreg_weight: float = 0.0
    remained_nt_vicreg_floor: float = 1.0
    remained_cov_fro_weight: float = 0.0
    teacher_momentum: float | None = 0.95
    masked_target_source: str | None = 'teacher_tokenizer'
    tokenizer_output_norm: str | None = 'layernorm'
    mask_style: str = 'curriculum_random_except_last'
    mask_rate: float | None = 0.3
    alignment_loss_weight: float = 1.0
    recovery_loss_weight: float = 0.0
    amplitude_recovery_loss_weight: float = 0.0
    diffusion_loss_weight: float = 0.0
    epochs: int = 0
    stage2_epochs: int = 0
    stage2_freeze_tokenizers: int = 0
    stage2_jepa_only: int = 0

    def to_overrides(self) -> dict[str, object]:
        payload = asdict(self)
        if int(payload.get('epochs', 0)) <= 0:
            payload.pop('epochs', None)
        for key in ('teacher_momentum', 'masked_target_source', 'tokenizer_output_norm', 'mask_rate'):
            if payload.get(key) is None:
                payload.pop(key, None)
        return payload


@dataclass
class TrainingOverrideSpec:
    name: str
    basic: BasicTrainingConfig


OVERRIDE_CONFIGS: list[TrainingOverrideSpec] = []

_BASE_NUM_TOKENS = 2
_BASE_BATCH_SIZE = 32
_BASE_LR = 1e-4
_BASE_MASK_STYLE = 'curriculum_random_except_last'


def _format_penalty_name(*, m: float, cov: float, aux: str) -> str:
    if m == 0.0 and cov == 0.0:
        penalty = 'penalty_none'
    else:
        penalty = f'penalty_m{m:g}_cov{cov:g}'
    return f'{penalty}__{aux}'


def _append_override(
    *,
    preset_name: str,
    token_dim: int,
    predictor_dim: int,
    remained_nt_vicreg_weight: float,
    remained_cov_fro_weight: float,
    recovery_loss_weight: float = 0.0,
    amplitude_recovery_loss_weight: float = 0.0,
    diffusion_loss_weight: float = 0.0,
) -> None:
    if recovery_loss_weight > 0:
        aux = f'recovery{recovery_loss_weight:g}'
    elif amplitude_recovery_loss_weight > 0:
        aux = f'amp_recovery{amplitude_recovery_loss_weight:g}'
    elif diffusion_loss_weight > 0:
        aux = f'diffusion{diffusion_loss_weight:g}'
    else:
        aux = 'no_aux'
    suffix = _format_penalty_name(m=remained_nt_vicreg_weight, cov=remained_cov_fro_weight, aux=aux)
    name = (
        f'{preset_name}'
        f'__nt{_BASE_NUM_TOKENS}'
        f'__td{token_dim}'
        f'__pd{predictor_dim}'
        f'__mask-{_BASE_MASK_STYLE}'
        f'__bs{_BASE_BATCH_SIZE}'
        f'__lr{_BASE_LR:.0e}'
        f'__{suffix}'
        f'__ema0p95'
        f'__toknorm_layernorm'
    )
    OVERRIDE_CONFIGS.append(
        TrainingOverrideSpec(
            name=name,
            basic=BasicTrainingConfig(
                jepa_config=NamedJEPAConfig(
                    preset_name=preset_name,
                    num_tokens=_BASE_NUM_TOKENS,
                    token_dim=token_dim,
                    predictor_dim=predictor_dim,
                ),
                batch_size=_BASE_BATCH_SIZE,
                lr=_BASE_LR,
                jepa_loss_weight=1.0,
                remained_nt_vicreg_weight=remained_nt_vicreg_weight,
                remained_cov_fro_weight=remained_cov_fro_weight,
                recovery_loss_weight=recovery_loss_weight,
                amplitude_recovery_loss_weight=amplitude_recovery_loss_weight,
                diffusion_loss_weight=diffusion_loss_weight,
            ),
        )
    )


# Open-source subset: 20 unique configs
# Indices 0-8  : loss9 sweep
# Indices 9-19 : topology12 additions beyond the shared 134 config

_KEPT_CONFIGS = [
    # loss9
    dict(preset_name='32x16_medium_medium_shallow', token_dim=192, predictor_dim=256, m=0.0, cov=0.0, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_medium_shallow', token_dim=192, predictor_dim=256, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_medium_shallow', token_dim=192, predictor_dim=256, m=0.0, cov=0.2, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_medium_shallow', token_dim=192, predictor_dim=256, m=0.0, cov=0.0, rec=0.0, amp=0.2, diff=0.0),
    dict(preset_name='32x16_medium_medium_shallow', token_dim=192, predictor_dim=256, m=0.0, cov=0.0, rec=0.0, amp=0.0, diff=0.2),
    dict(preset_name='32x16_medium_medium_shallow', token_dim=192, predictor_dim=256, m=0.0, cov=0.0, rec=0.2, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_medium_shallow', token_dim=192, predictor_dim=256, m=0.0, cov=0.1, rec=0.0, amp=0.2, diff=0.0),
    dict(preset_name='32x16_medium_medium_shallow', token_dim=192, predictor_dim=256, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.2),
    dict(preset_name='32x16_medium_medium_shallow', token_dim=192, predictor_dim=256, m=0.0, cov=0.1, rec=0.2, amp=0.0, diff=0.0),
    # topology12 additions beyond the shared 134 config
    dict(preset_name='32x16_medium_medium_shallow', token_dim=128, predictor_dim=192, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_medium_shallow', token_dim=128, predictor_dim=256, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_medium_shallow', token_dim=192, predictor_dim=192, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_medium_shallow', token_dim=256, predictor_dim=256, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_medium_shallow', token_dim=256, predictor_dim=384, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_deep_medium', token_dim=128, predictor_dim=192, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_deep_medium', token_dim=128, predictor_dim=256, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_deep_medium', token_dim=192, predictor_dim=192, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_deep_medium', token_dim=256, predictor_dim=256, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_deep_medium', token_dim=256, predictor_dim=384, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.0),
    dict(preset_name='32x16_medium_deep_medium', token_dim=192, predictor_dim=256, m=0.0, cov=0.1, rec=0.0, amp=0.0, diff=0.0),
]

for spec in _KEPT_CONFIGS:
    _append_override(
        preset_name=spec['preset_name'],
        token_dim=spec['token_dim'],
        predictor_dim=spec['predictor_dim'],
        remained_nt_vicreg_weight=spec['m'],
        remained_cov_fro_weight=spec['cov'],
        recovery_loss_weight=spec['rec'],
        amplitude_recovery_loss_weight=spec['amp'],
        diffusion_loss_weight=spec['diff'],
    )


def get_override_spec(override_id: int) -> TrainingOverrideSpec:
    idx = int(override_id)
    if idx < 0 or idx >= len(OVERRIDE_CONFIGS):
        raise IndexError(f'override_id {idx} is out of range for {len(OVERRIDE_CONFIGS)} configs')
    return OVERRIDE_CONFIGS[idx]


def apply_override(base_config: JEPATrainingConfig, override_id: int) -> JEPATrainingConfig:
    spec = get_override_spec(override_id)
    payload = base_config.to_dict()
    payload.update(spec.basic.to_overrides())
    payload['dataset_file'] = list(base_config.dataset_file)
    payload['working_dir'] = base_config.working_dir
    payload['jepa_config'] = NamedJEPAConfig(**payload['jepa_config'])
    return JEPATrainingConfig(**payload)


def get_named_override_config(base_config: JEPATrainingConfig, override_id: int) -> tuple[str, JEPATrainingConfig]:
    spec = get_override_spec(override_id)
    return spec.name, apply_override(base_config, override_id)


__all__ = [
    'BasicTrainingConfig',
    'TrainingOverrideSpec',
    'OVERRIDE_CONFIGS',
    'get_override_spec',
    'apply_override',
    'get_named_override_config',
]
