"""Compatibility wrapper for repository-root imports."""

from .sensitivity_lab.engine import SweepResult, run_parameter_sweep

__all__ = ["SweepResult", "run_parameter_sweep"]
