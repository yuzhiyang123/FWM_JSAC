from __future__ import annotations

from importlib import import_module

__all__ = [
    "AbstractCellTokenizer",
    "AbstractMaskGenerator",
    "TeacherStudentConfig",
    "TeacherStudentOutput",
    "TeacherStudentPipeline",
    "TeacherStudentLoss",
    "TeacherStudentTrainingConfig",
    "TeacherStudentExperimentConfig",
    "TeacherStudentLossComponent",
    "AlignmentNtVarianceLossComponent",
    "AlignmentMAxisVarianceLossComponent",
    "AlignmentFlattenedOffDiagonalCovarianceFrobeniusLossComponent",
    "configure_teacher_student_loss_weights",
    "build_teacher_student_experiment_configs",
    "evaluate_batch",
    "train_batch",
    "run_teacher_student_training",
    "run_teacher_student_experiment_suite",
    "AbstractTeacherStudentLossComponent",
    "AlignmentLossComponent",
    "MaskedDiffusionNoiseLossComponent",
    "MaskedRecoveryLossComponent",
    "PairedObservationBatch",
    "PredictionLossComponent",
    "RandomMaskExceptLastGenerator",
    "RandomMaskGenerator",
    "TeacherStudentJEPAConfig",
    "TeacherStudentJEPAOutput",
    "TeacherStudentJEPAPipeline",
    "TeacherStudentLossOutput",
    "TimeBandJEPAConfig",
    "TimeBandJEPAOutput",
    "TimeBandJEPAPipeline",
    "TimeBandViTConfig",
    "WeightedTeacherStudentLoss",
    "build_lazy_config",
    "build_preprocessor",
    "build_tokenizer",
    "evaluate_teacher_student_batch",
    "train_teacher_student_batch",
]

_MODULE_MAP = {
    "AbstractCellTokenizer": (".tokenizer", "AbstractCellTokenizer"),
    "AbstractMaskGenerator": (".jepa_32x16", "AbstractMaskGenerator"),
    "TeacherStudentConfig": (".teacher_student", "TeacherStudentConfig"),
    "TeacherStudentOutput": (".teacher_student", "TeacherStudentOutput"),
    "TeacherStudentPipeline": (".teacher_student", "TeacherStudentPipeline"),
    "TeacherStudentLoss": (".teacher_student", "TeacherStudentLoss"),
    "TeacherStudentTrainingConfig": (".teacher_student", "TeacherStudentTrainingConfig"),
    "TeacherStudentExperimentConfig": (".teacher_student", "TeacherStudentExperimentConfig"),
    "TeacherStudentLossComponent": (".teacher_student", "TeacherStudentLossComponent"),
    "AlignmentNtVarianceLossComponent": (".teacher_student", "AlignmentNtVarianceLossComponent"),
    "AlignmentMAxisVarianceLossComponent": (".teacher_student", "AlignmentMAxisVarianceLossComponent"),
    "AlignmentFlattenedOffDiagonalCovarianceFrobeniusLossComponent": (".teacher_student", "AlignmentFlattenedOffDiagonalCovarianceFrobeniusLossComponent"),
    "configure_teacher_student_loss_weights": (".teacher_student", "configure_teacher_student_loss_weights"),
    "build_teacher_student_experiment_configs": (".teacher_student", "build_teacher_student_experiment_configs"),
    "evaluate_batch": (".teacher_student", "evaluate_batch"),
    "train_batch": (".teacher_student", "train_batch"),
    "run_teacher_student_training": (".teacher_student", "run_teacher_student_training"),
    "run_teacher_student_experiment_suite": (".teacher_student", "run_teacher_student_experiment_suite"),
    "AbstractTeacherStudentLossComponent": (".teacher_student", "AbstractTeacherStudentLossComponent"),
    "AlignmentLossComponent": (".teacher_student", "AlignmentLossComponent"),
    "MaskedDiffusionNoiseLossComponent": (".teacher_student", "MaskedDiffusionNoiseLossComponent"),
    "MaskedRecoveryLossComponent": (".teacher_student", "MaskedRecoveryLossComponent"),
    "PairedObservationBatch": (".teacher_student", "PairedObservationBatch"),
    "PredictionLossComponent": (".teacher_student", "PredictionLossComponent"),
    "RandomMaskExceptLastGenerator": (".jepa_32x16", "RandomMaskExceptLastGenerator"),
    "RandomMaskGenerator": (".jepa_32x16", "RandomMaskGenerator"),
    "TeacherStudentJEPAConfig": (".teacher_student", "TeacherStudentJEPAConfig"),
    "TeacherStudentJEPAOutput": (".teacher_student", "TeacherStudentJEPAOutput"),
    "TeacherStudentJEPAPipeline": (".teacher_student", "TeacherStudentJEPAPipeline"),
    "TeacherStudentLossOutput": (".teacher_student", "TeacherStudentLossOutput"),
    "TimeBandJEPAConfig": (".jepa_32x16", "TimeBandJEPAConfig"),
    "TimeBandJEPAOutput": (".jepa_32x16", "TimeBandJEPAOutput"),
    "TimeBandJEPAPipeline": (".jepa_32x16", "TimeBandJEPAPipeline"),
    "TimeBandViTConfig": (".jepa_32x16", "TimeBandViTConfig"),
    "WeightedTeacherStudentLoss": (".teacher_student", "WeightedTeacherStudentLoss"),
    "build_lazy_config": (".jepa_32x16", "build_lazy_config"),
    "build_preprocessor": (".jepa_32x16", "build_preprocessor"),
    "build_tokenizer": (".jepa_32x16", "build_tokenizer"),
    "evaluate_teacher_student_batch": (".teacher_student", "evaluate_teacher_student_batch"),
    "train_teacher_student_batch": (".teacher_student", "train_teacher_student_batch"),
}


def __getattr__(name: str):
    if name not in _MODULE_MAP:
        raise AttributeError(name)
    module_name, attr_name = _MODULE_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
