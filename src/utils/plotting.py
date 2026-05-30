import os
import numpy as np
import matplotlib.pyplot as plt


def plot_results(results_dict, title, plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for name, metrics_dict in results_dict.items():
        x = np.arange(len(metrics_dict["curve"]))
        plt.plot(x, metrics_dict["curve"], label=name)

        if metrics_dict.get("n_runs", 1) > 1:
            lower = metrics_dict["curve"] - metrics_dict["curve_std"]
            upper = metrics_dict["curve"] + metrics_dict["curve_std"]
            plt.fill_between(x, lower, upper, alpha=0.15)

    plt.xlabel("Block")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(plots_dir, f"{title}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()