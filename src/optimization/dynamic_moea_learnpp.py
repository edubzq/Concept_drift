import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from river import metrics

from src.data.loading import load_blocks
from src.ensembles.learnpp_nse import LearnPPNSE
from src.utils.metrics import (
    compute_ensemble_diversity,
    compute_multiple_recoveries,
    _safe_nanmean,
)
from src.utils.plotting import plot_results


@dataclass
class DynamicMOEAConfig:
    dataset_path: str = "datasets/agrawal_abrupt.csv"
    window_size: int = 10
    reopt_frequency: int = 5
    pop_size: int = 20
    n_gen: int = 15
    seed: int = 42
    initial_a: float = 0.5
    initial_b: float = 5.0
    initial_max_size: int = 20
    baseline_a: float = 0.5
    baseline_b: float = 5.0
    baseline_max_size: int = 20
    a_min: float = 0.1
    a_max: float = 2.0
    b_min: float = 1.0
    b_max: float = 15.0
    max_size_min: int = 5
    max_size_max: int = 30
    drop_threshold: float = 0.15
    pre_window: int = 3
    recovery_ratio: float = 0.60
    local_min_window: int = 5
    cache_decimals: int = 4
    use_elapsed_time_objective: bool = False
    output_dir: str = "results"
    plots_dir: str = "plots"
    verbose: bool = True


@dataclass
class CandidateEvaluation:
    accuracy: float
    accuracy_min: float
    diversity: float
    recovery_time: float
    cost: float
    elapsed: float


def _validate_config(config):
    if config.window_size < 2:
        raise ValueError("window_size debe ser >= 2 para evaluar accuracy en la ventana.")
    if config.reopt_frequency < 1:
        raise ValueError("reopt_frequency debe ser >= 1.")
    if config.pop_size < 1:
        raise ValueError("pop_size debe ser >= 1.")
    if config.n_gen < 1:
        raise ValueError("n_gen debe ser >= 1.")
    if config.max_size_min < 1 or config.max_size_max < config.max_size_min:
        raise ValueError("Rango inválido para max_size.")
    if config.drop_threshold < 0:
        raise ValueError("drop_threshold debe ser >= 0.")
    if not 0 < config.recovery_ratio <= 1:
        raise ValueError("recovery_ratio debe estar en (0, 1].")


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


def _summary_to_df(results, dataset_name, config):
    rows = []
    for model_name, summary in results.items():
        rows.append({
            "dataset": dataset_name,
            "model": model_name,
            "window_size": config.window_size if "MOEA" in model_name else np.nan,
            "reopt_frequency": config.reopt_frequency if "MOEA" in model_name else np.nan,
            "pop_size": config.pop_size if "MOEA" in model_name else np.nan,
            "n_gen": config.n_gen if "MOEA" in model_name else np.nan,
            "drop_threshold": config.drop_threshold,
            "recovery_ratio": config.recovery_ratio,
            "accuracy_mean": summary["accuracy_mean"],
            "accuracy_min": summary["accuracy_min"],
            "kappa_mean": summary["kappa_mean"],
            "diversity_mean": summary["diversity_mean"],
            "recovery_time_mean": summary["recovery_time_mean"],
            "num_drops": summary["num_drops"],
            "cost_mean": summary["cost_mean"],
            "stream_time": summary["stream_time"],
            "optimization_time": summary["optimization_time"],
            "total_time": summary["total_time"],
            "num_reoptimizations": summary["num_reoptimizations"],
            "final_a": summary["final_a"],
            "final_b": summary["final_b"],
            "final_max_size": summary["final_max_size"],
        })

    return pd.DataFrame(rows)


def _curves_to_df(results):
    max_len = max(len(summary["curve"]) for summary in results.values())
    rows = []

    for curve_position in range(max_len):
        row = {"block_index": curve_position + 1}
        for model_name, summary in results.items():
            curve = summary["curve"]
            row[model_name] = curve[curve_position] if curve_position < len(curve) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _print_comparison(summary_df):
    print("\n=== Comparación Learn++NSE fijo vs Learn++NSE + MOEA dinámico ===")
    printable_cols = [
        "model",
        "accuracy_mean",
        "accuracy_min",
        "recovery_time_mean",
        "kappa_mean",
        "diversity_mean",
        "cost_mean",
        "stream_time",
        "optimization_time",
        "total_time",
        "num_reoptimizations",
        "final_a",
        "final_b",
        "final_max_size",
    ]
    print(summary_df[printable_cols].to_string(index=False))


def run_dynamic_moea_experiment(config=None, **kwargs):
    if config is None:
        config = DynamicMOEAConfig(**kwargs)
    elif kwargs:
        raise ValueError("Pasa config o kwargs, pero no ambos.")

    _validate_config(config)
    chunks = load_blocks(config.dataset_path)
    dataset_name = os.path.basename(config.dataset_path).replace(".csv", "")

    if config.verbose:
        print(f"Dataset: {config.dataset_path}")
        print(f"Bloques: {len(chunks)}")
        print(f"Ventana dinámica: {config.window_size} bloques")
        print(f"Frecuencia de reoptimización: cada {config.reopt_frequency} bloques")
        print(f"NSGA-II: pop_size={config.pop_size}, n_gen={config.n_gen}")
        print(
            "Objetivos MOEA: minimizar recovery_time y cost; maximizar accuracy_min"
        )

    baseline = evaluate_fixed_learnpp(chunks, config)
    dynamic = evaluate_dynamic_moea_learnpp(chunks, config)

    results = {
        "Learn++NSE fijo": baseline,
        "Learn++NSE + MOEA dinámico": dynamic,
    }

    os.makedirs(config.output_dir, exist_ok=True)
    summary_df = _summary_to_df(results, dataset_name, config)
    curves_df = _curves_to_df(results)

    summary_path = os.path.join(config.output_dir, f"dynamic_moea_{dataset_name}_summary.csv")
    curves_path = os.path.join(config.output_dir, f"dynamic_moea_{dataset_name}_accuracy_curve.csv")
    reopt_path = os.path.join(config.output_dir, f"dynamic_moea_{dataset_name}_reoptimizations.csv")
    pareto_path = os.path.join(config.output_dir, f"dynamic_moea_{dataset_name}_pareto_history.csv")
    config_curve_path = os.path.join(config.output_dir, f"dynamic_moea_{dataset_name}_config_curve.csv")

    summary_df.to_csv(summary_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    dynamic["reoptimizations"].to_csv(reopt_path, index=False)
    dynamic["pareto_history"].to_csv(pareto_path, index=False)
    dynamic["config_curve"].to_csv(config_curve_path, index=False)

    plot_results(
        {
            name: {"curve": summary["curve"], "n_runs": 1}
            for name, summary in results.items()
        },
        title=f"dynamic_moea_{dataset_name}",
        plots_dir=config.plots_dir,
    )

    if config.verbose:
        _print_comparison(summary_df)
        print("\nArchivos guardados:")
        print(f"  Resumen: {summary_path}")
        print(f"  Curvas de accuracy: {curves_path}")
        print(f"  Reoptimizaciones: {reopt_path}")
        print(f"  Historial Pareto: {pareto_path}")
        print(f"  Configuración por bloque: {config_curve_path}")
        print(f"  Plot: {os.path.join(config.plots_dir, f'dynamic_moea_{dataset_name}.png')}")

    return {
        "summary": summary_df,
        "curves": curves_df,
        "reoptimizations": dynamic["reoptimizations"],
        "pareto_history": dynamic["pareto_history"],
        "config_curve": dynamic["config_curve"],
        "raw_results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Learn++NSE fijo vs Learn++NSE con MOEA dinámico orientado "
            "a drift por ventana deslizante."
        )
    )
    parser.add_argument("--dataset", default=DynamicMOEAConfig.dataset_path)
    parser.add_argument("--window-size", type=int, default=DynamicMOEAConfig.window_size)
    parser.add_argument("--reopt-frequency", type=int, default=DynamicMOEAConfig.reopt_frequency)
    parser.add_argument("--pop-size", type=int, default=DynamicMOEAConfig.pop_size)
    parser.add_argument("--n-gen", type=int, default=DynamicMOEAConfig.n_gen)
    parser.add_argument("--seed", type=int, default=DynamicMOEAConfig.seed)
    parser.add_argument("--initial-a", type=float, default=DynamicMOEAConfig.initial_a)
    parser.add_argument("--initial-b", type=float, default=DynamicMOEAConfig.initial_b)
    parser.add_argument("--initial-max-size", type=int, default=DynamicMOEAConfig.initial_max_size)
    parser.add_argument("--baseline-a", type=float, default=DynamicMOEAConfig.baseline_a)
    parser.add_argument("--baseline-b", type=float, default=DynamicMOEAConfig.baseline_b)
    parser.add_argument("--baseline-max-size", type=int, default=DynamicMOEAConfig.baseline_max_size)
    parser.add_argument("--a-min", type=float, default=DynamicMOEAConfig.a_min)
    parser.add_argument("--a-max", type=float, default=DynamicMOEAConfig.a_max)
    parser.add_argument("--b-min", type=float, default=DynamicMOEAConfig.b_min)
    parser.add_argument("--b-max", type=float, default=DynamicMOEAConfig.b_max)
    parser.add_argument("--max-size-min", type=int, default=DynamicMOEAConfig.max_size_min)
    parser.add_argument("--max-size-max", type=int, default=DynamicMOEAConfig.max_size_max)
    parser.add_argument("--drop-threshold", type=float, default=DynamicMOEAConfig.drop_threshold)
    parser.add_argument("--pre-window", type=int, default=DynamicMOEAConfig.pre_window)
    parser.add_argument("--recovery-ratio", type=float, default=DynamicMOEAConfig.recovery_ratio)
    parser.add_argument("--local-min-window", type=int, default=DynamicMOEAConfig.local_min_window)
    parser.add_argument("--output-dir", default=DynamicMOEAConfig.output_dir)
    parser.add_argument("--plots-dir", default=DynamicMOEAConfig.plots_dir)
    parser.add_argument(
        "--use-elapsed-time-objective",
        action="store_true",
        help="Usa tiempo real de evaluación como tercer objetivo en lugar de max_size * bloques.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce la salida por consola.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = DynamicMOEAConfig(
        dataset_path=args.dataset,
        window_size=args.window_size,
        reopt_frequency=args.reopt_frequency,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        seed=args.seed,
        initial_a=args.initial_a,
        initial_b=args.initial_b,
        initial_max_size=args.initial_max_size,
        baseline_a=args.baseline_a,
        baseline_b=args.baseline_b,
        baseline_max_size=args.baseline_max_size,
        a_min=args.a_min,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
        max_size_min=args.max_size_min,
        max_size_max=args.max_size_max,
        drop_threshold=args.drop_threshold,
        pre_window=args.pre_window,
        recovery_ratio=args.recovery_ratio,
        local_min_window=args.local_min_window,
        use_elapsed_time_objective=args.use_elapsed_time_objective,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        verbose=not args.quiet,
    )
    run_dynamic_moea_experiment(config)


if __name__ == "__main__":
    main()