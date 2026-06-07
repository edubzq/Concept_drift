"""Compatibility entry point for the dynamic Learn++NSE MOEA experiment.

The implementation is split between:
- src.optimization.dynamic_config: configuration dataclasses and validation.
- src.optimization.dynamic_moea: dynamic optimization core.
- src.optimization.dynamic_runner: experiment orchestration and CLI.
"""

from src.optimization.dynamic_config import CandidateEvaluation, DynamicMOEAConfig
from src.optimization.dynamic_moea import (
    DynamicLearnPPNSGAProblem,
    choose_drift_compromise_solution,
    evaluate_dynamic_moea_learnpp,
    evaluate_fixed_learnpp,
    optimize_recent_window,
)
from src.optimization.dynamic_runner import main, parse_args, run_dynamic_moea_experiment

__all__ = [
    "CandidateEvaluation",
    "DynamicMOEAConfig",
    "DynamicLearnPPNSGAProblem",
    "choose_drift_compromise_solution",
    "evaluate_dynamic_moea_learnpp",
    "evaluate_fixed_learnpp",
    "optimize_recent_window",
    "parse_args",
    "run_dynamic_moea_experiment",
    "main",
]