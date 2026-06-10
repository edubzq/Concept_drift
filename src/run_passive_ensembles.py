import argparse
import fnmatch
import os
import time

import numpy as np
import pandas as pd

from river import metrics

from src.ensembles.sea_ensemble import SEAEnsemble
from src.ensembles.aue2_ensemble import AUE2Ensemble
from src.ensembles.wae_ensemble import WAEEnsemble
from src.ensembles.learnpp_nse import LearnPPNSE

from src.data.loading import load_blocks
from src.utils.metrics import (
    compute_ensemble_diversity,
    compute_multiple_recoveries,
    _safe_nanmean,
    _safe_nanstd,
)
from src.utils.plotting import plot_results


DATASET_DIR = "datasets_controlled"
RESULTS_CSV = "results/results_summary.csv"
PLOTS_DIR = "plots"
DATASET_PATTERN = "agrawal_*.csv"

N_RUNS = 1
ENSEMBLE_SIZE = 20

DROP_THRESHOLD = 0.15
PRE_WINDOW = 3
RECOVERY_RATIO = 0.60
LOCAL_MIN_WINDOW = 5


def evaluate_model(model, chunks):
    accuracies = []
    kappas = []
    diversities = []

    kappa_metric = metrics.CohenKappa()

    for i, (X, y) in enumerate(chunks):
        if i > 0:
            base_predictions = None

            if hasattr(model, "predict_with_base_predictions"):
                preds, base_predictions = model.predict_with_base_predictions(X)
            else:
                preds = np.array(model.predict(X))

            acc = np.mean(preds == y)
            accuracies.append(acc)

            for yt, yp in zip(y, preds):
                kappa_metric.update(yt, yp)

            kappas.append(kappa_metric.get())

            diversity = compute_ensemble_diversity(
                model,
                X=X,
                base_predictions=base_predictions,
            )
            diversities.append(diversity)

        model.fit_chunk(X, y)

    return accuracies, kappas, diversities


def evaluate_multiple_runs(model_class, chunks, n_runs=N_RUNS):
    if n_runs < 1:
        raise ValueError("n_runs debe ser >= 1.")

    run_results = []

    for _ in range(n_runs):
        model = model_class(max_size=ENSEMBLE_SIZE)

        start = time.perf_counter()
        accs, kappas, diversities = evaluate_model(model, chunks)
        elapsed = time.perf_counter() - start

        min_acc, recovery_mean, _, n_drops = compute_multiple_recoveries(
            accs,
            drop_threshold=DROP_THRESHOLD,
            pre_window=PRE_WINDOW,
            recovery_ratio=RECOVERY_RATIO,
            local_min_window=LOCAL_MIN_WINDOW,
        )

        run_results.append({
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_min": float(min_acc),
            "recovery_mean": float(recovery_mean) if not np.isnan(recovery_mean) else np.nan,
            "kappa_mean": _safe_nanmean(kappas),
            "diversity_mean": _safe_nanmean(diversities),
            "time": float(elapsed),
            "num_drops": float(n_drops),
            "curve": np.array(accs, dtype=float),
        })

    curves = [result["curve"] for result in run_results]
    min_curve_len = min(len(curve) for curve in curves)
    curve_matrix = np.vstack([curve[:min_curve_len] for curve in curves])

    metric_names = [
        "accuracy_mean",
        "accuracy_min",
        "recovery_mean",
        "kappa_mean",
        "diversity_mean",
        "time",
        "num_drops",
    ]

    summary = {}
    for metric_name in metric_names:
        values = [result[metric_name] for result in run_results]
        summary[f"{metric_name}_mean"] = _safe_nanmean(values)
        summary[f"{metric_name}_std"] = _safe_nanstd(values)

    summary["curve"] = np.mean(curve_matrix, axis=0)
    summary["curve_std"] = np.std(curve_matrix, axis=0)
    summary["n_runs"] = n_runs

    return summary


def save_results_csv(all_results, output_path=RESULTS_CSV):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []

    for dataset_name, models_dict in all_results.items():
        for model_name, metrics_dict in models_dict.items():
            rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "n_runs": metrics_dict["n_runs"],
                "ensemble_size": ENSEMBLE_SIZE,
                "accuracy_mean": metrics_dict["accuracy_mean_mean"],
                "accuracy_std": metrics_dict["accuracy_mean_std"],
                "accuracy_min_mean": metrics_dict["accuracy_min_mean"],
                "accuracy_min_std": metrics_dict["accuracy_min_std"],
                "recovery_mean": metrics_dict["recovery_mean_mean"],
                "recovery_std": metrics_dict["recovery_mean_std"],
                "kappa_mean": metrics_dict["kappa_mean_mean"],
                "kappa_std": metrics_dict["kappa_mean_std"],
                "diversity_mean": metrics_dict["diversity_mean_mean"],
                "diversity_std": metrics_dict["diversity_mean_std"],
                "time_mean": metrics_dict["time_mean"],
                "time_std": metrics_dict["time_std"],
                "num_drops_mean": metrics_dict["num_drops_mean"],
                "num_drops_std": metrics_dict["num_drops_std"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def print_results(results, dataset_name):
    print(f"\n=== Results for {dataset_name} ===")
    for model_name, m in results.items():
        print(f"\n{model_name}")
        print(f"  Runs               : {m['n_runs']}")
        print(f"  Accuracy mean      : {m['accuracy_mean_mean']:.4f} ± {m['accuracy_mean_std']:.4f}")
        print(f"  Accuracy min       : {m['accuracy_min_mean']:.4f} ± {m['accuracy_min_std']:.4f}")
        print(f"  Recovery mean      : {m['recovery_mean_mean']}")
        print(f"  Kappa mean         : {m['kappa_mean_mean']:.4f} ± {m['kappa_mean_std']:.4f}")
        print(f"  Diversity mean     : {m['diversity_mean_mean']:.4f} ± {m['diversity_mean_std']:.4f}")
        print(f"  Time mean (s)      : {m['time_mean']:.4f} ± {m['time_std']:.4f}")
        print(f"  Num drops          : {m['num_drops_mean']:.0f} ± {m['num_drops_std']:.2f}")


def run_experiment(dataset_path, plots_dir=PLOTS_DIR):
    chunks = load_blocks(dataset_path)

    models = {
        "SEA": SEAEnsemble,
        "AUE2": AUE2Ensemble,
        "WAE": WAEEnsemble,
        "Learn++NSE": LearnPPNSE,
    }

    results = {}

    for name, model_class in models.items():
        metrics_dict = evaluate_multiple_runs(model_class, chunks, n_runs=N_RUNS)
        results[name] = metrics_dict

    dataset_name = os.path.basename(dataset_path).replace(".csv", "")

    plot_results(results, dataset_name, plots_dir=plots_dir)
    print_results(results, dataset_name)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalúa ensembles pasivos sobre datasets Agrawal por bloques."
    )
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--results-csv", default=RESULTS_CSV)
    parser.add_argument("--plots-dir", default=PLOTS_DIR)
    parser.add_argument("--dataset-pattern", default=DATASET_PATTERN)
    return parser.parse_args()


def main():
    args = parse_args()
    all_results = {}

    for file in sorted(os.listdir(args.dataset_dir)):
        if file.endswith(".csv") and fnmatch.fnmatch(file, args.dataset_pattern):
            dataset_path = os.path.join(args.dataset_dir, file)
            results = run_experiment(dataset_path, plots_dir=args.plots_dir)
            all_results[file] = results

    save_results_csv(all_results, args.results_csv)
    print(f"\nResumen guardado en: {args.results_csv}")


if __name__ == "__main__":
    main()