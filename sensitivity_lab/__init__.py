"""Compatibility exports for repository-root imports."""

from .sensitivity_lab import SweepResult, build_sensitivity_figure, run_parameter_sweep

__all__ = ["SweepResult", "run_parameter_sweep", "build_sensitivity_figure"]
