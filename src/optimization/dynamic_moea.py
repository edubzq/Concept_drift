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
from src.utils.metrics import (
    compute_ensemble_diversity,
    compute_multiple_recoveries,
    _safe_nanmean,
)


def _drift_behavior_metrics(
    accuracies,
    drop_threshold=0.15,
    pre_window=3,
    recovery_ratio=0.60,
    local_min_window=5,
):
    curve = np.asarray(accuracies, dtype=float)

    if len(curve) == 0:
        return {
            "recovery_time": np.nan,
            "num_drops": 0,
        }

    if len(curve) == 1:
        return {
            "recovery_time": 0.0,
            "num_drops": 0,
        }

    _, recovery_mean, _, num_drops = compute_multiple_recoveries(
        curve,
        drop_threshold=drop_threshold,
        pre_window=pre_window,
        recovery_ratio=recovery_ratio,
        local_min_window=local_min_window,
    )
    recovery_time = 0.0 if np.isnan(recovery_mean) else float(recovery_mean)

    return {
        "recovery_time": recovery_time,
        "num_drops": int(num_drops),
    }


def _evaluation_cache_key(a, b, max_size, cache_decimals):
    return (round(float(a), cache_decimals), round(float(b), cache_decimals), int(max_size))


def _evaluate_learnpp_config_on_window(chunks, a, b, max_size, config):
    model = LearnPPNSE(a=float(a), b=float(b), max_size=int(round(max_size)))
    accuracies = []
    diversities = []

    start = time.perf_counter()

    for block_index, (X, y) in enumerate(chunks):
        if block_index > 0:
            preds, base_predictions = model.predict_with_base_predictions(X)
            accuracies.append(float(np.mean(preds == y)))
            diversities.append(
                compute_ensemble_diversity(model, base_predictions=base_predictions)
            )

        model.fit_chunk(X, y)

    elapsed = time.perf_counter() - start
    drift_metrics = _drift_behavior_metrics(
        accuracies,
        drop_threshold=config.drop_threshold,
        pre_window=config.pre_window,
        recovery_ratio=config.recovery_ratio,
        local_min_window=config.local_min_window,
    )

    accuracy = _safe_nanmean(accuracies)
    accuracy_min = float(np.min(accuracies)) if len(accuracies) > 0 else np.nan
    diversity = _safe_nanmean(diversities)

    if config.use_elapsed_time_objective:
        cost = elapsed
    else:
        cost = int(round(max_size)) * max(len(chunks) - 1, 1)

    return CandidateEvaluation(
        accuracy=accuracy,
        accuracy_min=accuracy_min,
        diversity=diversity,
        recovery_time=drift_metrics["recovery_time"],
        cost=float(cost),
        elapsed=float(elapsed),
    )


class DynamicLearnPPNSGAProblem(ElementwiseProblem):
    def __init__(self, chunks, config):
        self.chunks = chunks
        self.config = config
        self.evaluation_cache = {}

        super().__init__(
            n_var=3,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.array(
                [config.a_min, config.b_min, config.max_size_min],
                dtype=float,
            ),
            xu=np.array(
                [config.a_max, config.b_max, config.max_size_max],
                dtype=float,
            ),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        a = float(x[0])
        b = float(x[1])
        max_size = int(round(x[2]))
        cache_key = _evaluation_cache_key(
            a,
            b,
            max_size,
            self.config.cache_decimals,
        )

        if cache_key not in self.evaluation_cache:
            self.evaluation_cache[cache_key] = _evaluate_learnpp_config_on_window(
                self.chunks,
                a=a,
                b=b,
                max_size=max_size,
                config=self.config,
            )

        evaluation = self.evaluation_cache[cache_key]
        out["F"] = np.array(
            [
                evaluation.recovery_time,
                -evaluation.accuracy_min,
                evaluation.cost,
            ],
            dtype=float,
        )

    def get_cached_evaluation(self, x):
        cache_key = _evaluation_cache_key(
            float(x[0]),
            float(x[1]),
            int(round(x[2])),
            self.config.cache_decimals,
        )
        return self.evaluation_cache[cache_key]


def choose_drift_compromise_solution(res):
    F = np.atleast_2d(np.asarray(res.F, dtype=float))
    X = np.atleast_2d(np.asarray(res.X, dtype=float))

    normalized = np.zeros_like(F, dtype=float)
    for objective_idx in range(F.shape[1]):
        values = F[:, objective_idx]
        min_value = np.min(values)
        max_value = np.max(values)
        if max_value > min_value:
            normalized[:, objective_idx] = (values - min_value) / (max_value - min_value)

    weights = np.array([0.45, 0.35, 0.20], dtype=float)
    score = normalized @ weights
    best_idx = int(np.argmin(score))

    return best_idx, X[best_idx], F[best_idx], float(score[best_idx])


def _pareto_to_df(res, problem, block_index):
    rows = []
    for x, f in zip(np.atleast_2d(res.X), np.atleast_2d(res.F)):
        evaluation = problem.get_cached_evaluation(x)
        rows.append({
            "block_index": int(block_index),
            "a": round(float(x[0]), 4),
            "b": round(float(x[1]), 4),
            "max_size": int(round(x[2])),
            "recovery_time": round(float(f[0]), 6),
            "accuracy_mean": round(evaluation.accuracy, 6),
            "accuracy_min": round(evaluation.accuracy_min, 6),
            "cost": round(float(f[2]), 6),
            "diversity": round(evaluation.diversity, 6),
            "evaluation_elapsed": round(evaluation.elapsed, 6),
        })

    return pd.DataFrame(rows).sort_values(
        by=["recovery_time", "accuracy_min", "cost"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _apply_learnpp_config(model, a, b, max_size):
    model.a = float(a)
    model.b = float(b)
    model.max_size = int(max_size)

    while len(model.models) > model.max_size:
        model.models.pop(0)
        model.beta_history.pop(0)

    if model.models:
        model._refresh_voting_weights()


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


def _summarize_run(accuracies, kappas, diversities, elapsed, config, extra=None):
    drift_metrics = _drift_behavior_metrics(
        accuracies,
        drop_threshold=config.drop_threshold,
        pre_window=config.pre_window,
        recovery_ratio=config.recovery_ratio,
        local_min_window=config.local_min_window,
    )

    summary = {
        "accuracy_mean": _safe_nanmean(accuracies),
        "accuracy_min": float(np.min(accuracies)) if len(accuracies) > 0 else np.nan,
        "kappa_mean": _safe_nanmean(kappas),
        "diversity_mean": _safe_nanmean(diversities),
        "recovery_time_mean": drift_metrics["recovery_time"],
        "num_drops": drift_metrics["num_drops"],
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
    )
    accuracies = []
    kappas = []
    diversities = []
    ensemble_sizes = []
    kappa_metric = metrics.CohenKappa()

    start = time.perf_counter()

    for block_index, (X, y) in enumerate(chunks):
        if block_index > 0:
            accuracy, kappa, diversity = _predict_update_metrics(model, X, y, kappa_metric)
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
        config=config,
        extra={
            "optimization_time": 0.0,
            "stream_time": float(elapsed),
            "total_time": float(elapsed),
            "cost_mean": _safe_nanmean(ensemble_sizes),
            "num_reoptimizations": 0,
            "final_a": float(config.baseline_a),
            "final_b": float(config.baseline_b),
            "final_max_size": int(config.baseline_max_size),
        },
    )


def optimize_recent_window(recent_chunks, config, block_index):
    problem = DynamicLearnPPNSGAProblem(recent_chunks, config)
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

    best_idx, best_x, best_f, best_score = choose_drift_compromise_solution(res)
    best_evaluation = problem.get_cached_evaluation(best_x)
    selected = {
        "block_index": int(block_index),
        "window_start": int(block_index - len(recent_chunks) + 1),
        "window_end": int(block_index),
        "a": float(best_x[0]),
        "b": float(best_x[1]),
        "max_size": int(round(best_x[2])),
        "window_recovery_time": float(best_f[0]),
        "window_accuracy_mean": float(best_evaluation.accuracy),
        "window_accuracy_min": float(best_evaluation.accuracy_min),
        "window_cost": float(best_f[2]),
        "window_diversity": float(best_evaluation.diversity),
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
        max_size=config.initial_max_size,
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
            accuracy, kappa, diversity = _predict_update_metrics(model, X, y, kappa_metric)
            accuracies.append(accuracy)
            kappas.append(kappa)
            diversities.append(diversity)
            ensemble_sizes.append(len(model.models))
            config_curve.append({
                "block_index": block_index,
                "a": float(model.a),
                "b": float(model.b),
                "max_size": int(model.max_size),
                "ensemble_size": int(len(model.models)),
            })

        model.fit_chunk(X, y)

        has_next_block = block_index < len(chunks) - 1
        has_full_window = block_index + 1 >= config.window_size
        due_for_reoptimization = (block_index + 1) % config.reopt_frequency == 0

        if has_next_block and has_full_window and due_for_reoptimization:
            recent_chunks = chunks[block_index + 1 - config.window_size:block_index + 1]
            selected, pareto_df = optimize_recent_window(recent_chunks, config, block_index)
            optimization_time += selected["optimizer_elapsed"]
            reoptimization_rows.append(selected)
            pareto_frames.append(pareto_df)
            _apply_learnpp_config(
                model,
                a=selected["a"],
                b=selected["b"],
                max_size=selected["max_size"],
            )

            if config.verbose:
                print(
                    "Reoptimización tras bloque "
                    f"{block_index}: a={selected['a']:.4f}, "
                    f"b={selected['b']:.4f}, "
                    f"max_size={selected['max_size']}, "
                    f"recovery={selected['window_recovery_time']:.2f}, "
                    f"accuracy_min={selected['window_accuracy_min']:.4f}"
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
        config=config,
        extra={
            "optimization_time": float(optimization_time),
            "stream_time": float(stream_elapsed),
            "total_time": float(total_elapsed),
            "cost_mean": _safe_nanmean(ensemble_sizes),
            "num_reoptimizations": int(len(reoptimization_rows)),
            "final_a": float(model.a),
            "final_b": float(model.b),
            "final_max_size": int(model.max_size),
            "reoptimizations": pd.DataFrame(reoptimization_rows),
            "pareto_history": pareto_history,
            "config_curve": pd.DataFrame(config_curve),
        },
    )