"""
Genera datasets sintéticos con distintos tipos de concept drift
usando el generador Hyperplane.
"""

import os
from calm_data_generator.generators.stream.StreamBlockGenerator import SyntheticBlockGenerator

OUTPUT_DIR = "datasets"

TOTAL_SAMPLES = 20000
N_BLOCKS = 20


def generate_dataset(filename, params):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gen = SyntheticBlockGenerator()

    path = gen.generate_blocks_simple(
        output_dir=OUTPUT_DIR,
        filename=filename,
        n_blocks=len(params),
        total_samples=TOTAL_SAMPLES,
        methods="hyperplane",
        method_params=params,
        generate_report=False,
    )

    print(f"Dataset generado: {path}")


# ------------------------------------------------
# Abrupt drift
# ------------------------------------------------

def generate_abrupt():

    params = []

    for i in range(N_BLOCKS):

        # cambio brusco en la velocidad de rotación
        if i < N_BLOCKS // 2:
            mag = 0.01
        else:
            mag = 0.08

        params.append({
            "n_features": 10,
            "mag_change": mag,
            "sigma": 0.1
        })

    generate_dataset("hyperplane_abrupt.csv", params)


# ------------------------------------------------
# Gradual drift
# ------------------------------------------------

def generate_gradual():

    params = []

    for i in range(N_BLOCKS):

        # drift progresivo
        mag = 0.005 + (i / N_BLOCKS) * 0.05

        params.append({
            "n_features": 10,
            "mag_change": mag,
            "sigma": 0.1
        })

    generate_dataset("hyperplane_gradual.csv", params)


# ------------------------------------------------
# Recurrent drift
# ------------------------------------------------

def generate_recurrent():

    params = []

    concept_length = 5

    for i in range(N_BLOCKS):

        concept = (i // concept_length) % 2

        if concept == 0:
            mag = 0.01
        else:
            mag = 0.07

        params.append({
            "n_features": 10,
            "mag_change": mag,
            "sigma": 0.1
        })

    generate_dataset("hyperplane_recurrent.csv", params)


# ------------------------------------------------
# MAIN
# ------------------------------------------------

if __name__ == "__main__":

    generate_abrupt()
    generate_gradual()
    generate_recurrent()