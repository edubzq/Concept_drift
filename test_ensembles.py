
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sea_ensemble import SEAEnsemble
from aue2_ensemble import AUE2Ensemble
from wae_ensemble import WAEEnsemble
from learnpp_nse import LearnPPNSE

from river import metrics

DATASET_DIR = "datasets"
PLOTS_DIR = "plots"
RESULTS_CSV = "results_summary.csv"

N_RUNS = 1
ENSEMBLE_SIZE = 20

# Parámetros para recovery automático sin conocer bloques de drift
DROP_THRESHOLD = 0.15   # caída mínima fuerte entre bloques
PRE_WINDOW = 3          # bloques previos para estimar baseline local
RECOVERY_RATIO = 0.60   # % del daño recuperado para considerar recuperación
LOCAL_MIN_WINDOW = 5    # ventana local tras la caída para estimar el mínimo

os.makedirs(PLOTS_DIR, exist_ok=True)


# ------------------------------------------------
# Cargar bloques
# ------------------------------------------------

def load_blocks(csv_path):
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c not in ["target", "block", "timestamp", "concept"]]

    chunks = []

    for block in sorted(df["block"].unique()):
        block_df = df[df["block"] == block]

        X = block_df[feature_cols]
        y = block_df["target"].values

        chunks.append((X, y))

    return chunks


# ------------------------------------------------
# Diversidad del ensemble
# ------------------------------------------------

def compute_ensemble_diversity(model, X):
    """
    Diversidad media por pares basada en disagreement.
    Devuelve 0 si el ensemble tiene menos de 2 modelos.
    """
    if not hasattr(model, "models") or len(model.models) < 2:
        return 0.0

    X_dict = X.to_dict(orient="records")
    preds = []

    for base_model in model.models:
        model_preds = []
        for xi in X_dict:
            pred = base_model.predict_one(xi)
            model_preds.append(pred)
        preds.append(model_preds)

    preds = np.array(preds)
    n_models = preds.shape[0]

    total_disagreement = 0.0
    total_pairs = 0

    for i in range(n_models):
        for j in range(i + 1, n_models):
            disagreement = np.mean(preds[i] != preds[j])
            total_disagreement += disagreement
            total_pairs += 1

    if total_pairs == 0:
        return 0.0

    return float(total_disagreement / total_pairs)


# ------------------------------------------------
# Prequential evaluation
# ------------------------------------------------

def evaluate_model(model, chunks):
    accuracies = []
    kappas = []
    diversities = []

    kappa_metric = metrics.CohenKappa()

    for i, (X, y) in enumerate(chunks):
        if i > 0:
            preds = np.array(model.predict(X))

            acc = np.mean(preds == y)
            accuracies.append(acc)

            for yt, yp in zip(y, preds):
                kappa_metric.update(yt, yp)

            kappas.append(kappa_metric.get())

            diversity = compute_ensemble_diversity(model, X)
            diversities.append(diversity)

        model.fit_chunk(X, y)

    return accuracies, kappas, diversities


# ------------------------------------------------
# Recovery basado en caídas fuertes y recuperación relativa
# ------------------------------------------------

def compute_multiple_recoveries(curve,
                                drop_threshold=DROP_THRESHOLD,
                                pre_window=PRE_WINDOW,
                                recovery_ratio=RECOVERY_RATIO,
                                local_min_window=LOCAL_MIN_WINDOW):
    curve = np.array(curve, dtype=float)

    if len(curve) == 0:
        return np.nan, np.nan, [], 0

    min_acc = np.min(curve)
    recoveries = []

    i = 1
    while i < len(curve):
        drop = curve[i] - curve[i - 1]

        if drop <= -drop_threshold:
            start_pre = max(0, i - pre_window)
            baseline_segment = curve[start_pre:i]

            if len(baseline_segment) == 0:
                baseline = curve[i - 1]
            else:
                baseline = np.mean(baseline_segment)

            end_local = min(i + local_min_window, len(curve))
            local_segment = curve[i:end_local]

            if len(local_segment) == 0:
                min_point = curve[i]
            else:
                min_point = np.min(local_segment)

            recovery_target = min_point + recovery_ratio * (baseline - min_point)

            recovered = False
            for j in range(i, len(curve)):
                if curve[j] >= recovery_target:
                    recoveries.append(j - i)
                    i = j + 1
                    recovered = True
                    break

            if not recovered:
                recoveries.append(len(curve) - i)
                break
        else:
            i += 1

    if len(recoveries) == 0:
        mean_recovery = np.nan
    else:
        mean_recovery = float(np.mean(recoveries))

    return min_acc, mean_recovery, recoveries, len(recoveries)


# ------------------------------------------------
# Una sola run
# ------------------------------------------------

def evaluate_multiple_runs(model_class, chunks):
    model = model_class(max_size=ENSEMBLE_SIZE)

    start = time.perf_counter()
    accs, kappas, diversities = evaluate_model(model, chunks)
    elapsed = time.perf_counter() - start

    min_acc, recovery_mean, recovery_list, n_drops = compute_multiple_recoveries(accs)

    mean_curve = np.array(accs, dtype=float)
    std_curve = np.zeros_like(mean_curve)

    mean_acc = float(np.mean(accs))
    std_acc = 0.0

    if len(kappas) == 0:
        mean_kappa = np.nan
    else:
        mean_kappa = float(np.mean(kappas))
    std_kappa = 0.0

    if len(diversities) == 0:
        mean_diversity = np.nan
    else:
        mean_diversity = float(np.mean(diversities))
    std_diversity = 0.0

    return {
        "accuracy_mean": mean_acc,
        "accuracy_std": std_acc,
        "accuracy_min_mean": float(min_acc),
        "accuracy_min_std": 0.0,
        "recovery_mean": float(recovery_mean) if not np.isnan(recovery_mean) else np.nan,
        "recovery_std": 0.0,
        "kappa_mean": mean_kappa,
        "kappa_std": std_kappa,
        "diversity_mean": mean_diversity,
        "diversity_std": std_diversity,
        "time_mean": float(elapsed),
        "time_std": 0.0,
        "num_drops_mean": float(n_drops),
        "num_drops_std": 0.0,
        "curve": mean_curve,
        "curve_std": std_curve
    }


# ------------------------------------------------
# Plot results
# ------------------------------------------------

def plot_results(results_dict, title):
    plt.figure(figsize=(10, 6))

    for name, metrics_dict in results_dict.items():
        plt.plot(metrics_dict["curve"], label=name)

    plt.xlabel("Block")
    plt.ylabel("Accuracy")
    plt.title(title)

    plt.legend()
    plt.grid(True)

    save_path = os.path.join(PLOTS_DIR, f"{title}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ------------------------------------------------
# Guardar resultados en CSV
# ------------------------------------------------

def save_results_csv(all_results, output_path=RESULTS_CSV):
    rows = []

    for dataset_name, models_dict in all_results.items():
        for model_name, metrics_dict in models_dict.items():
            rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "accuracy_mean": metrics_dict["accuracy_mean"],
                "accuracy_min_mean": metrics_dict["accuracy_min_mean"],
                "recovery_mean": metrics_dict["recovery_mean"],
                "kappa_mean": metrics_dict["kappa_mean"],
                "diversity_mean": metrics_dict["diversity_mean"],
                "time_mean": metrics_dict["time_mean"],
                "num_drops_mean": metrics_dict["num_drops_mean"]
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


# ------------------------------------------------
# Mostrar resumen por pantalla
# ------------------------------------------------

def print_results(results, dataset_name):
    print(f"\n=== Results for {dataset_name} ===")
    for model_name, m in results.items():
        print(f"\n{model_name}")
        print(f"  Accuracy mean      : {m['accuracy_mean']:.4f}")
        print(f"  Accuracy min       : {m['accuracy_min_mean']:.4f}")
        print(f"  Recovery mean      : {m['recovery_mean']}")
        print(f"  Kappa mean         : {m['kappa_mean']:.4f}")
        print(f"  Diversity mean     : {m['diversity_mean']:.4f}")
        print(f"  Time mean (s)      : {m['time_mean']:.4f}")
        print(f"  Num drops          : {m['num_drops_mean']:.0f}")


# ------------------------------------------------
# Ejecutar experimento
# ------------------------------------------------

def run_experiment(dataset_path):
    chunks = load_blocks(dataset_path)

    models = {
        "SEA": SEAEnsemble,
        "AUE2": AUE2Ensemble,
        "WAE": WAEEnsemble,
        "Learn++NSE": LearnPPNSE,
    }

    results = {}

    for name, model_class in models.items():
        metrics_dict = evaluate_multiple_runs(model_class, chunks)
        results[name] = metrics_dict

    dataset_name = os.path.basename(dataset_path).replace(".csv", "")

    plot_results(results, dataset_name)
    print_results(results, dataset_name)

    return results


# ------------------------------------------------
# MAIN
# ------------------------------------------------

if __name__ == "__main__":
    all_results = {}

    for file in os.listdir(DATASET_DIR):
        if file.endswith(".csv"):
            dataset_path = os.path.join(DATASET_DIR, file)
            results = run_experiment(dataset_path)
            all_results[file] = results

    save_results_csv(all_results, RESULTS_CSV)
    print(f"\nResumen guardado en: {RESULTS_CSV}")