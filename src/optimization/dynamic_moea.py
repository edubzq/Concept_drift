
import time

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from river import metrics

from src.ensembles.learnpp_nse import LearnPPNSE
from src.optimization.dynamic_config import CandidateEvaluation
from src.utils.metrics import compute_ensemble_diversity, _safe_nanmean


def decode_candidate(x, config):
    return {
        "a": float(x[0]),
        "b": float(x[1]),
        "grace_period": int(round(float(x[2]))),
        "log_delta": float(x[3]),
        "delta": 10 ** float(x[3]),
    }


def _evaluation_cache_key(candidate, cache_decimals):
    return (
        round(float(candidate["a"]), cache_decimals),
        round(float(candidate["b"]), cache_decimals),
        int(candidate["grace_period"]),
        round(float(candidate["log_delta"]), cache_decimals),
    )


def _evaluate_candidate_on_window(chunks, block_index, candidate, config):
    window_start = block_index + 1 - config.window_size
    window_end = block_index + 1
    window_chunks = chunks[window_start:window_end]

    # Versión sin checkpoint:
    # cada candidato parte de un Learn++NSE vacío y se entrena solo con la ventana reciente.
    model = LearnPPNSE(
        a=candidate["a"],
        b=candidate["b"],
        max_size=config.max_size,
        grace_period=candidate["grace_period"],
        delta=candidate["delta"],
    )

    accuracies = []
    diversities = []

    start = time.perf_counter()

    for X, y in window_chunks:
        if model.models:
            preds, base_predictions = model.predict_with_base_predictions(X)
            accuracies.append(float(np.mean(preds == y)))
            diversities.append(
                compute_ensemble_diversity(model, base_predictions=base_predictions)
            )

        model.fit_chunk(X, y)

    elapsed = time.perf_counter() - start

    # Con window_size=2, esto evalúa solo el último bloque de la ventana.
    objective_size = min(1, len(accuracies))
    recent_accuracies = accuracies[-objective_size:]
    recent_diversities = diversities[-objective_size:]

    return CandidateEvaluation(
        recent_accuracy=_safe_nanmean(recent_accuracies),
        diversity=_safe_nanmean(recent_diversities),
        complexity=float(elapsed),
        elapsed=float(elapsed),
    )


class PassiveLearnPPNSGAProblem(ElementwiseProblem):
    def __init__(self, chunks, block_index, config):
        self.chunks = chunks
        self.block_index = block_index
        self.config = config
        self.evaluation_cache = {}

        super().__init__(
            n_var=4,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.array(
                [
                    config.a_min,
                    config.b_min,
                    config.grace_period_min,
                    config.log_delta_min,
                ],
                dtype=float,
            ),
            xu=np.array(
                [
                    config.a_max,
                    config.b_max,
                    config.grace_period_max,
                    config.log_delta_max,
                ],
                dtype=float,
            ),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        candidate = decode_candidate(x, self.config)
        cache_key = _evaluation_cache_key(candidate, self.config.cache_decimals)

        if cache_key not in self.evaluation_cache:
            self.evaluation_cache[cache_key] = _evaluate_candidate_on_window(
            self.chunks,
            self.block_index,
            candidate,
            self.config,
        )
        evaluation = self.evaluation_cache[cache_key]

        out["F"] = np.array(
            [
                -evaluation.recent_accuracy,
                -evaluation.diversity,
                evaluation.elapsed,
            ],
            dtype=float,
        )

    def get_cached_evaluation(self, x):
        candidate = decode_candidate(x, self.config)
        cache_key = _evaluation_cache_key(candidate, self.config.cache_decimals)
        return self.evaluation_cache[cache_key]


def choose_compromise_solution(res):
    F = np.atleast_2d(np.asarray(res.F, dtype=float))
    X = np.atleast_2d(np.asarray(res.X, dtype=float))

    normalized = np.zeros_like(F, dtype=float)

    for objective_idx in range(F.shape[1]):
        values = F[:, objective_idx]
        min_value = np.min(values)
        max_value = np.max(values)

        if max_value > min_value:
            normalized[:, objective_idx] = (
                values - min_value
            ) / (max_value - min_value)

    # recent_accuracy, diversity, elapsed_time
    weights = np.array([0.60, 0.25, 0.15], dtype=float)
    score = normalized @ weights
    best_idx = int(np.argmin(score))

    return best_idx, X[best_idx], F[best_idx], float(score[best_idx])


def _pareto_to_df(res, problem, block_index):
    rows = []

    for x, _ in zip(np.atleast_2d(res.X), np.atleast_2d(res.F)):
        candidate = decode_candidate(x, problem.config)
        evaluation = problem.get_cached_evaluation(x)

        rows.append({
            "block_index": int(block_index),
            "a": round(candidate["a"], 4),
            "b": round(candidate["b"], 4),
            "grace_period": int(candidate["grace_period"]),
            "delta": float(candidate["delta"]),
            "recent_accuracy": round(evaluation.recent_accuracy, 6),
            "diversity": round(evaluation.diversity, 6),
            "complexity": round(evaluation.elapsed, 6),
            "evaluation_elapsed": round(evaluation.elapsed, 6),
        })

    return pd.DataFrame(rows).sort_values(
        by=["recent_accuracy", "diversity", "complexity"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _predict_update_metrics(model, X, y, kappa_metric):
    preds, base_predictions = model.predict_with_base_predictions(X)
    accuracy = float(np.mean(preds == y))

    for yt, yp in zip(y, preds):
        kappa_metric.update(yt, yp)

    diversity = compute_ensemble_diversity(
        model,
        X=X,
        base_predictions=base_predictions,
    )

    return accuracy, float(kappa_metric.get()), diversity


def _summarize_run(accuracies, kappas, diversities, elapsed, extra=None):
    summary = {
        "accuracy_mean": _safe_nanmean(accuracies),
        "accuracy_min": float(np.min(accuracies)) if len(accuracies) > 0 else np.nan,
        "kappa_mean": _safe_nanmean(kappas),
        "diversity_mean": _safe_nanmean(diversities),
        "time": float(elapsed),
        "curve": np.asarray(accuracies, dtype=float),
        "n_runs": 1,
    }

    if extra:
        summary.update(extra)

    return summary


def evaluate_fixed_learnpp(chunks, config):
    model = LearnPPNSE(
        a=config.baseline_a,
        b=config.baseline_b,
        max_size=config.baseline_max_size,
        grace_period=config.baseline_grace_period,
        delta=config.baseline_delta,
    )

    accuracies = []
    kappas = []
    diversities = []
    ensemble_sizes = []
    kappa_metric = metrics.CohenKappa()

    start = time.perf_counter()

    for block_index, (X, y) in enumerate(chunks):
        if block_index > 0:
            accuracy, kappa, diversity = _predict_update_metrics(
                model, X, y, kappa_metric
            )
            accuracies.append(accuracy)
            kappas.append(kappa)
            diversities.append(diversity)
            ensemble_sizes.append(len(model.models))

        model.fit_chunk(X, y)

    elapsed = time.perf_counter() - start

    return _summarize_run(
        accuracies,
        kappas,
        diversities,
        elapsed,
        extra={
            "optimization_time": 0.0,
            "stream_time": float(elapsed),
            "total_time": float(elapsed),
            "cost_mean": _safe_nanmean(ensemble_sizes),
            "num_reoptimizations": 0,
            "final_a": float(config.baseline_a),
            "final_b": float(config.baseline_b),
            "final_grace_period": int(config.baseline_grace_period),
            "final_delta": float(config.baseline_delta),
        },
    )


def optimize_recent_window(chunks, config, block_index):
    problem = PassiveLearnPPNSGAProblem(chunks, block_index, config)
    algorithm = NSGA2(pop_size=config.pop_size)
    termination = get_termination("n_gen", config.n_gen)

    start = time.perf_counter()
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=config.seed + int(block_index),
        verbose=False,
    )
    elapsed = time.perf_counter() - start

    best_idx, best_x, _, best_score = choose_compromise_solution(res)
    best_candidate = decode_candidate(best_x, config)
    best_evaluation = problem.get_cached_evaluation(best_x)

    selected = {
        "block_index": int(block_index),
        "window_start": int(block_index + 1 - config.window_size),
        "window_end": int(block_index),
        "a": float(best_candidate["a"]),
        "b": float(best_candidate["b"]),
        "grace_period": int(best_candidate["grace_period"]),
        "delta": float(best_candidate["delta"]),
        "window_recent_accuracy": float(best_evaluation.recent_accuracy),
        "window_diversity": float(best_evaluation.diversity),
        "window_complexity": float(best_evaluation.elapsed),
        "compromise_score": float(best_score),
        "pareto_index": int(best_idx),
        "optimizer_elapsed": float(elapsed),
        "pareto_size": int(len(np.atleast_2d(res.X))),
    }

    pareto_df = _pareto_to_df(res, problem, block_index)
    return selected, pareto_df


def evaluate_dynamic_moea_learnpp(chunks, config):
    model = LearnPPNSE(
        a=config.initial_a,
        b=config.initial_b,
        max_size=config.max_size,
        grace_period=config.initial_grace_period,
        delta=config.initial_delta,
    )

    accuracies = []
    kappas = []
    diversities = []
    ensemble_sizes = []
    config_curve = []
    reoptimization_rows = []
    pareto_frames = []
    optimization_time = 0.0
    kappa_metric = metrics.CohenKappa()

    stream_start = time.perf_counter()

    for block_index, (X, y) in enumerate(chunks):

        if block_index > 0:
            accuracy, kappa, diversity = _predict_update_metrics(
                model, X, y, kappa_metric
            )
            accuracies.append(accuracy)
            kappas.append(kappa)
            diversities.append(diversity)
            ensemble_sizes.append(len(model.models))

            config_curve.append({
                "block_index": block_index,
                "a": float(model.a),
                "b": float(model.b),
                "grace_period": int(model.grace_period),
                "log_delta": float(np.log10(model.delta)),
                "max_size": int(model.max_size),
                "ensemble_size": int(len(model.models)),
            })

        model.fit_chunk(X, y)

        has_next_block = block_index < len(chunks) - 1
        has_full_window = block_index + 1 >= config.window_size

        if has_next_block and has_full_window:
            selected, pareto_df = optimize_recent_window(
                chunks=chunks,
                config=config,
                block_index=block_index,
            )

            optimization_time += selected["optimizer_elapsed"]

            reoptimization_rows.append(selected)
            pareto_frames.append(pareto_df)

            # no sustituimos el modelo real, solo actualizamos sus hiperparámetros.
            model.set_config(
                a=selected["a"],
                b=selected["b"],
                grace_period=selected["grace_period"],
                delta=selected["delta"],
            )

            if config.verbose:
                print(
                    f"Reoptimización pasiva tras bloque {block_index}: "
                    f"a={selected['a']:.4f}, "
                    f"b={selected['b']:.4f}, "
                    f"grace_period={selected['grace_period']}, "
                    f"delta={selected['delta']:.2e},"
                    f"recent_accuracy={selected['window_recent_accuracy']:.4f}, "
                    f"diversity={selected['window_diversity']:.4f}, "
                    f"elapsed={selected['window_complexity']:.6f}s"
                )

    total_elapsed = time.perf_counter() - stream_start
    stream_elapsed = max(0.0, total_elapsed - optimization_time)

    pareto_history = (
        pd.concat(pareto_frames, ignore_index=True)
        if pareto_frames
        else pd.DataFrame()
    )

    return _summarize_run(
        accuracies,
        kappas,
        diversities,
        total_elapsed,
        extra={
            "optimization_time": float(optimization_time),
            "stream_time": float(stream_elapsed),
            "total_time": float(total_elapsed),
            "cost_mean": _safe_nanmean(ensemble_sizes),
            "num_reoptimizations": int(len(reoptimization_rows)),
            "final_a": float(model.a),
            "final_b": float(model.b),
            "final_grace_period": int(model.grace_period),
            "final_log_delta": float(np.log10(model.delta)),
            "final_delta": float(model.delta),
            "final_max_size": int(model.max_size),
            "reoptimizations": pd.DataFrame(reoptimization_rows),
            "pareto_history": pareto_history,
            "config_curve": pd.DataFrame(config_curve),
        },
    )