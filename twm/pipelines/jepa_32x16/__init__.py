from __future__ import annotations

from importlib import import_module

__all__ = [
    "AbstractMaskGenerator",
    "RandomMaskExceptLastGenerator",
    "CurriculumRandomMaskExceptLastGenerator",
    "RandomMaskExceptLastVariableRateGenerator",
    "RandomMaskGenerator",
    "TimeBandJEPAConfig",
    "TimeBandJEPAOutput",
    "TimeBandJEPAPipeline",
    "TimeBandViTConfig",
    "build_preprocessor",
    "build_tokenizer",
    "build_lazy_config",
    "AbstractJEPALoss",
    "BasicJEPATrainingConfig",
    "NamedJEPAConfig",
    "AbstractJEPALossComponent",
    "FlattenedOffDiagonalCovarianceFrobeniusLossComponent",
    "JEPALossOutput",
    "JEPATrainingConfig",
    "JEPAStage2TrainingConfig",
    "StandardJEPALossComponent",
    "VICRegNtVarianceLossComponent",
    "WeightedJEPALoss",
    "WarmupCosineScheduleConfig",
    "build_warmup_cosine_scheduler",
    "evaluate_jepa_batch",
    "run_jepa_training",
    "run_jepa_stage2_training",
    "train_jepa_batch",
    "make_mask_generator",
]

_MODULE_MAP = {
    "AbstractMaskGenerator": (".time_freq_band", "AbstractMaskGenerator"),
    "RandomMaskExceptLastGenerator": (".time_freq_band", "RandomMaskExceptLastGenerator"),
    "CurriculumRandomMaskExceptLastGenerator": (".time_freq_band", "CurriculumRandomMaskExceptLastGenerator"),
    "RandomMaskExceptLastVariableRateGenerator": (".time_freq_band", "RandomMaskExceptLastVariableRateGenerator"),
    "RandomMaskGenerator": (".time_freq_band", "RandomMaskGenerator"),
    "build_preprocessor": (".utils", "build_preprocessor"),
    "build_tokenizer": (".utils", "build_tokenizer"),
    "build_lazy_config": (".default_pipeline_configs", "build_lazy_config"),
    "TimeBandJEPAConfig": (".pipeline", "TimeBandJEPAConfig"),
    "TimeBandJEPAOutput": (".pipeline", "TimeBandJEPAOutput"),
    "TimeBandJEPAPipeline": (".pipeline", "TimeBandJEPAPipeline"),
    "TimeBandViTConfig": (".pipeline", "TimeBandViTConfig"),
    "AbstractJEPALoss": (".training", "AbstractJEPALoss"),
    "BasicJEPATrainingConfig": (".training", "BasicJEPATrainingConfig"),
    "NamedJEPAConfig": (".training", "NamedJEPAConfig"),
    "AbstractJEPALossComponent": (".training", "AbstractJEPALossComponent"),
    "FlattenedOffDiagonalCovarianceFrobeniusLossComponent": (".training", "FlattenedOffDiagonalCovarianceFrobeniusLossComponent"),
    "JEPALossOutput": (".training", "JEPALossOutput"),
    "JEPATrainingConfig": (".training", "JEPATrainingConfig"),
    "JEPAStage2TrainingConfig": (".training", "JEPAStage2TrainingConfig"),
    "StandardJEPALossComponent": (".training", "StandardJEPALossComponent"),
    "VICRegNtVarianceLossComponent": (".training", "VICRegNtVarianceLossComponent"),
    "WeightedJEPALoss": (".training", "WeightedJEPALoss"),
    "WarmupCosineScheduleConfig": (".training", "WarmupCosineScheduleConfig"),
    "build_warmup_cosine_scheduler": (".training", "build_warmup_cosine_scheduler"),
    "evaluate_jepa_batch": (".training", "evaluate_jepa_batch"),
    "run_jepa_training": (".training", "run_jepa_training"),
    "run_jepa_stage2_training": (".training", "run_jepa_stage2_training"),
    "train_jepa_batch": (".training", "train_jepa_batch"),
    "make_mask_generator": (".training", "make_mask_generator"),
}


def __getattr__(name: str):
    if name not in _MODULE_MAP:
        raise AttributeError(name)
    module_name, attr_name = _MODULE_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
