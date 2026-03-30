
import os
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

N_RUNS = 5
ENSEMBLE_SIZE = 20

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
# Prequential evaluation
# ------------------------------------------------

def evaluate_model(model, chunks):

    accuracies = []
    kappas = []

    kappa_metric = metrics.CohenKappa()

    for i, (X, y) in enumerate(chunks):

        if i > 0:

            preds = model.predict(X)

            acc = np.mean(preds == y)
            accuracies.append(acc)

            # actualizar kappa
            for yt, yp in zip(y, preds):
                kappa_metric.update(yt, yp)

            kappas.append(kappa_metric.get())

        model.fit_chunk(X, y)

    return accuracies, kappas


# ------------------------------------------------
# Métricas adicionales
# ------------------------------------------------

def compute_additional_metrics(curve):

    curve = np.array(curve)

    min_index = np.argmin(curve)
    min_acc = curve[min_index]

    baseline = np.mean(curve[:5])

    recovery_block = None

    for i in range(min_index, len(curve)):
        if curve[i] >= baseline:
            recovery_block = i - min_index
            break

    if recovery_block is None:
        recovery_block = len(curve) - min_index

    return min_acc, recovery_block


# ------------------------------------------------
# Multiple runs
# ------------------------------------------------

def evaluate_multiple_runs(model_class, chunks):

    all_runs_acc = []
    all_runs_kappa = []

    for _ in range(N_RUNS):

        if model_class.__name__ == "LearnPPNSE":
            model = model_class()
        else:
            model = model_class(max_size=ENSEMBLE_SIZE)

        accs, kappas = evaluate_model(model, chunks)

        all_runs_acc.append(accs)
        all_runs_kappa.append(kappas)

    all_runs_acc = np.array(all_runs_acc)
    all_runs_kappa = np.array(all_runs_kappa)

    mean_curve = np.mean(all_runs_acc, axis=0)
    std_curve = np.std(all_runs_acc, axis=0)

    mean_acc = np.mean(mean_curve)
    std_acc = np.std(mean_curve)

    mean_kappa = np.mean(all_runs_kappa)

    # métricas adicionales
    min_acc, recovery = compute_additional_metrics(mean_curve)

    return {
        "accuracy_mean": mean_acc,
        "accuracy_std": std_acc,
        "accuracy_min": min_acc,
        "recovery": recovery,
        "kappa_mean": mean_kappa,
        "curve": mean_curve
    }


# ------------------------------------------------
# Plot results
# ------------------------------------------------

def plot_results(results_dict, title):

    plt.figure(figsize=(10,6))

    for name, metrics_dict in results_dict.items():
        plt.plot(metrics_dict["curve"], label=name)

    plt.xlabel("Block")
    plt.ylabel("Accuracy")
    plt.title(title)

    for x in [5,10,15,20]:
        plt.axvline(x=x, linestyle="--", color="gray", alpha=0.5)

    plt.legend()
    plt.grid(True)

    save_path = os.path.join(PLOTS_DIR, f"{title}.png")

    plt.savefig(save_path)
    plt.close()


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

    dataset_name = os.path.basename(dataset_path).replace(".csv","")

    plot_results(results, dataset_name)

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