import argparse
import os

import numpy as np
import pandas as pd

from src.data.loading import load_blocks
from src.optimization.dynamic_config import DynamicMOEAConfig, validate_dynamic_config
from src.optimization.dynamic_moea import evaluate_dynamic_moea_learnpp, evaluate_fixed_learnpp
from src.utils.plotting import plot_results


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

    validate_dynamic_config(config)
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