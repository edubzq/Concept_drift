import os

import numpy as np
import matplotlib.pyplot as plt


def _drift_markers(title):
    """Devuelve marcadores de drift para los escenarios Agrawal conocidos."""
    normalized = title.lower()

    if "real_concept" in normalized or "real concept" in normalized:
        if "gradual" in normalized:
            return {
                "spans": [(11, 30, "Transición gradual")],
                "lines": [],
            }

        if "abrupt" in normalized:
            return {
                "spans": [],
                "lines": [(21, "Drift abrupto")],
            }

        if "recurrent" in normalized:
            return {
                "spans": [
                    (9, 14, "Drift recurrente"),
                    (19, 24, None),
                    (29, 34, None),
                ],
                "lines": [],
            }


    if "gradual" in normalized:
        return {
            "spans": [(13, 28, "Transición gradual")],
            "lines": [],
        }

    if "abrupt" in normalized:
        return {
            "spans": [],
            "lines": [(21, "Drift abrupto")],
        }

    if "recurrent" in normalized:
        return {
            "spans": [
                (1, 10, "Drift recurrente"),
                (11, 20, None),
                (21, 30, None),
                (31, 40, None),
            ],
            "lines": [],
        }

    return {"spans": [], "lines": []}


def plot_results(results_dict, title, plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    markers = _drift_markers(title)
    for start_block, end_block, label in markers["spans"]:
        plt.axvspan(
            start_block,
            end_block,
            color="tab:orange",
            alpha=0.08,
            label=label,
        )

    for block, label in markers["lines"]:
        plt.axvline(
            block,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label=label,
        )

    for name, metrics_dict in results_dict.items():
        # La primera accuracy corresponde a la evaluación prequential del bloque 2,
        # porque el bloque 1 solo se usa para entrenar el primer modelo.
        x = np.arange(2, len(metrics_dict["curve"]) + 2)
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
    plt.ylim(0.0, 1.0)

    save_path = os.path.join(plots_dir, f"{title}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()