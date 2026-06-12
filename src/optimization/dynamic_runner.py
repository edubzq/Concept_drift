import argparse
import os

import numpy as np
import pandas as pd

from src.data.loading import load_blocks
from src.optimization.dynamic_config import (
    DynamicMOEAConfig,
    validate_dynamic_config,
)
from src.optimization.dynamic_moea import (
    evaluate_dynamic_moea_learnpp,
    evaluate_fixed_learnpp,
)
from src.utils.plotting import plot_results


def _summary_to_df(results, dataset_name, config):
    rows = []

    for model_name, summary in results.items():
        is_moea = "MOEA" in model_name

        final_delta = float(summary["final_delta"])
        final_log_delta = summary.get("final_log_delta")
        if final_log_delta is None:
            final_log_delta = np.log10(final_delta)
        rows.append({
            "dataset": dataset_name,
            "model": model_name,
            "window_size": config.window_size if is_moea else np.nan,
            "pop_size": config.pop_size if is_moea else np.nan,
            "n_gen": config.n_gen if is_moea else np.nan,
            "accuracy_mean": summary["accuracy_mean"],
            "accuracy_min": summary["accuracy_min"],
            "kappa_mean": summary["kappa_mean"],
            "diversity_mean": summary["diversity_mean"],
            "cost_mean": summary["cost_mean"],
            "stream_time": summary["stream_time"],
            "optimization_time": summary["optimization_time"],
            "total_time": summary["total_time"],
            "final_a": summary["final_a"],
            "final_b": summary["final_b"],
            "final_grace_period": summary["final_grace_period"],
            "final_log_delta": float(final_log_delta),
            "final_delta": final_delta,
            "final_recency_lambda": summary.get("final_recency_lambda", np.nan),
            "final_weight_power": summary.get("final_weight_power", np.nan),
        })

    return pd.DataFrame(rows)


def _curves_to_df(results):
    max_len = max(len(summary["curve"]) for summary in results.values())
    rows = []

    for curve_position in range(max_len):
        row = {"block_index": curve_position + 1}

        for model_name, summary in results.items():
            curve = summary["curve"]
            row[model_name] = (
                curve[curve_position]
                if curve_position < len(curve)
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _print_comparison(summary_df):
    print("\n=== Comparación Learn++NSE fijo vs Learn++NSE + MOEA pasivo ===")

    printable_cols = [
        "model",
        "accuracy_mean",
        "accuracy_min",
        "kappa_mean",
        "diversity_mean",
        "cost_mean",
        "stream_time",
        "optimization_time",
        "total_time",
        "final_a",
        "final_b",
        "final_grace_period",
        "final_log_delta",
        "final_delta",
        "final_recency_lambda",
        "final_weight_power",
    ]

    print(summary_df[printable_cols].to_string(index=False))


def run_dynamic_moea_experiment(config=None, **kwargs):
    if config is None:
        config = DynamicMOEAConfig(**kwargs)
    elif kwargs:
        raise ValueError("Pasa config o kwargs, pero no ambos.")

    validate_dynamic_config(config)

    chunks = load_blocks(config.dataset_path)
    dataset_name = os.path.basename(config.dataset_path).replace(".csv", "")

    if config.verbose:
        print(f"Dataset: {config.dataset_path}")
        print(f"Bloques: {len(chunks)}")
        print("Cromosoma MOEA: a, b, grace_period, delta, recency_lambda, weigght_power")
        print(f"NSGA-II: pop_size={config.pop_size}, n_gen={config.n_gen}")
        print(
            "Objetivos MOEA: maximizar recent_accuracy y diversity; "
            "minimizar tiempo de ejecución"
        )

    baseline = evaluate_fixed_learnpp(chunks, config)
    dynamic = evaluate_dynamic_moea_learnpp(chunks, config)

    results = {
        "Learn++NSE fijo": baseline,
        "Learn++NSE + MOEA pasivo": dynamic,
    }

    os.makedirs(config.output_dir, exist_ok=True)

    summary_df = _summary_to_df(results, dataset_name, config)
    curves_df = _curves_to_df(results)

    summary_path = os.path.join(
        config.output_dir,
        f"dynamic_moea_{dataset_name}_summary.csv",
    )
    curves_path = os.path.join(
        config.output_dir,
        f"dynamic_moea_{dataset_name}_accuracy_curve.csv",
    )
    reopt_path = os.path.join(
        config.output_dir,
        f"dynamic_moea_{dataset_name}_reoptimizations.csv",
    )
    pareto_path = os.path.join(
        config.output_dir,
        f"dynamic_moea_{dataset_name}_pareto_history.csv",
    )
    config_curve_path = os.path.join(
        config.output_dir,
        f"dynamic_moea_{dataset_name}_config_curve.csv",
    )

    summary_df.to_csv(summary_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    dynamic["reoptimizations"].to_csv(reopt_path, index=False)
    dynamic["pareto_history"].to_csv(pareto_path, index=False)
    dynamic["config_curve"].to_csv(config_curve_path, index=False)

    plot_title = f"dynamic_moea_{dataset_name}"
    plot_results(results, plot_title, plots_dir=config.plots_dir)
    plot_path = os.path.join(config.plots_dir, f"{plot_title}.png")
    
    if config.verbose:
        _print_comparison(summary_df)
        print("\nArchivos guardados:")
        print(f"  Resumen: {summary_path}")
        print(f"  Curvas de accuracy: {curves_path}")
        print(f"  Reoptimizaciones: {reopt_path}")
        print(f"  Historial Pareto: {pareto_path}")
        print(f"  Configuración por bloque: {config_curve_path}")
        print(f"  Plot: {plot_path}")

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
            "Learn++NSE fijo vs Learn++NSE con MOEA pasivo "
            "orientado a precisión reciente, diversidad y tiempo de ejecución."
        )
    )

    parser.add_argument("--dataset", default=DynamicMOEAConfig.dataset_path)
    parser.add_argument("--window-size", type=int, default=DynamicMOEAConfig.window_size)
    parser.add_argument("--pop-size", type=int, default=DynamicMOEAConfig.pop_size)
    parser.add_argument("--n-gen", type=int, default=DynamicMOEAConfig.n_gen)
    parser.add_argument("--seed", type=int, default=DynamicMOEAConfig.seed)
    parser.add_argument("--n-jobs", type=int, default=DynamicMOEAConfig.n_jobs)
    parser.add_argument("--initial-a", type=float, default=DynamicMOEAConfig.initial_a)
    parser.add_argument("--initial-b", type=float, default=DynamicMOEAConfig.initial_b)
    parser.add_argument(
        "--initial-grace-period",
        type=int,
        default=DynamicMOEAConfig.initial_grace_period,
    )
    parser.add_argument(
        "--initial-delta",
        type=float,
        default=DynamicMOEAConfig.initial_delta,
    )
    parser.add_argument(
        "--initial-weight-power",
        type=float,
        default=DynamicMOEAConfig.initial_weight_power,
    )
    parser.add_argument("--max-size", type=int, default=DynamicMOEAConfig.max_size)

    parser.add_argument("--baseline-a", type=float, default=DynamicMOEAConfig.baseline_a)
    parser.add_argument("--baseline-b", type=float, default=DynamicMOEAConfig.baseline_b)
    parser.add_argument(
        "--baseline-grace-period",
        type=int,
        default=DynamicMOEAConfig.baseline_grace_period,
    )
    parser.add_argument(
        "--baseline-delta",
        type=float,
        default=DynamicMOEAConfig.baseline_delta,
    )
    parser.add_argument(
        "--baseline-weight-power",
        type=float,
        default=DynamicMOEAConfig.baseline_weight_power,
    )
    parser.add_argument(
        "--baseline-max-size",
        type=int,
        default=DynamicMOEAConfig.baseline_max_size,
    )

    parser.add_argument("--a-min", type=float, default=DynamicMOEAConfig.a_min)
    parser.add_argument("--a-max", type=float, default=DynamicMOEAConfig.a_max)
    parser.add_argument("--b-min", type=float, default=DynamicMOEAConfig.b_min)
    parser.add_argument("--b-max", type=float, default=DynamicMOEAConfig.b_max)
    parser.add_argument(
        "--grace-period-min",
        type=int,
        default=DynamicMOEAConfig.grace_period_min,
    )
    parser.add_argument(
        "--grace-period-max",
        type=int,
        default=DynamicMOEAConfig.grace_period_max,
    )
    parser.add_argument(
        "--log-delta-min",
        type=float,
        default=DynamicMOEAConfig.log_delta_min,
    )
    parser.add_argument(
        "--log-delta-max",
        type=float,
        default=DynamicMOEAConfig.log_delta_max,
    )
    parser.add_argument(
        "--recency-lambda-min",
        type=float,
        default=DynamicMOEAConfig.recency_lambda_min,
    )
    parser.add_argument(
        "--recency-lambda-max",
        type=float,
        default=DynamicMOEAConfig.recency_lambda_max,
    )
    parser.add_argument(
        "--weight-power-min",
        type=float,
        default=DynamicMOEAConfig.weight_power_min,
    )
    parser.add_argument(
        "--weight-power-max",
        type=float,
        default=DynamicMOEAConfig.weight_power_max,
    )
    parser.add_argument("--output-dir", default=DynamicMOEAConfig.output_dir)
    parser.add_argument("--plots-dir", default=DynamicMOEAConfig.plots_dir)
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
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        seed=args.seed,
        n_jobs=args.n_jobs,
        initial_a=args.initial_a,
        initial_b=args.initial_b,
        initial_grace_period=args.initial_grace_period,
        initial_delta=args.initial_delta,
        initial_weight_power=args.initial_weight_power,
        max_size=args.max_size,
        baseline_a=args.baseline_a,
        baseline_b=args.baseline_b,
        baseline_grace_period=args.baseline_grace_period,
        baseline_delta=args.baseline_delta,
        baseline_weight_power=args.baseline_weight_power,
        baseline_max_size=args.baseline_max_size,
        a_min=args.a_min,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
        grace_period_min=args.grace_period_min,
        grace_period_max=args.grace_period_max,
        log_delta_min=args.log_delta_min,
        log_delta_max=args.log_delta_max,
        recency_lambda_min=args.recency_lambda_min,
        recency_lambda_max=args.recency_lambda_max,
        weight_power_min=args.weight_power_min,
        weight_power_max=args.weight_power_max,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        verbose=not args.quiet,
    )

    run_dynamic_moea_experiment(config)


if __name__ == "__main__":
    main()