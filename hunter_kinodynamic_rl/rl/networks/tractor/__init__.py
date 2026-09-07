"""TRACTOR-TQC model family.

The public surface is intentionally small: typed contracts plus the composed
``TractorTQC`` model.  Submodules remain importable for isolated ablations and
property tests.
"""

from .contracts import TractorConfig, TractorInputs
from .model import TractorTQC

__all__ = ["TractorConfig", "TractorInputs", "TractorTQC"]
