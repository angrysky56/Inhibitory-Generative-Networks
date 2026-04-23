"""Evaluation metrics, baselines, and reporting for IGN."""
from .metrics import signature_distance
from .baseline import AdditiveModel
from .report import compare_models

__all__ = ["signature_distance", "AdditiveModel", "compare_models"]
