"""Executable B1--B8 architecture baselines for the TRACTOR paper matrix."""

from .agent import ComparisonAgent
from .model import ComparisonModel, EncodedComparisonState

__all__ = ["ComparisonAgent", "ComparisonModel", "EncodedComparisonState"]
