
import os
import pandas as pd
import numpy as np
import random
from river.datasets import synth
from calm_data_generator.generators.stream.StreamBlockGenerator import SyntheticBlockGenerator

OUTPUT_DIR = "datasets"

TOTAL_SAMPLES = 20000
N_BLOCKS = 25
SAMPLES_PER_BLOCK = TOTAL_SAMPLES // N_BLOCKS


# ------------------------------------------------
# BASE
# ------------------------------------------------

def generate_dataset(filename, generators, concepts):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gen = SyntheticBlockGenerator()

    path = gen.generate(
        output_dir=OUTPUT_DIR,
        filename=filename,
        n_blocks=len(generators),
        total_samples=TOTAL_SAMPLES,
        n_samples_block=[SAMPLES_PER_BLOCK] * len(generators),
        generators=generators,
        generate_report=False,
    )

    df = pd.read_csv(path)

    concept_list = []
    for block in sorted(df["block"].unique()):
        concept_list += [concepts[block - 1]] * len(df[df["block"] == block])

    df["concept"] = concept_list
    df.to_csv(path, index=False)

    print(f"Dataset generado: {path}")


# ------------------------------------------------
# ABRUPT
# ------------------------------------------------

def generate_abrupt():

    rng = np.random.RandomState(42)

    generators = []
    concepts = []

    for i in range(N_BLOCKS):

        concept = 0 if i < N_BLOCKS // 2 else 2

        block_seed = rng.randint(0, 100000)

        generators.append(
            synth.SEA(
                variant=concept,
                noise=0.05,
                seed=block_seed
            )
        )

        concepts.append(concept)

    generate_dataset("sea_abrupt.csv", generators, concepts)


# ------------------------------------------------
# GRADUAL (CONTROL FINO)
# ------------------------------------------------

def generate_gradual():

    rng = np.random.RandomState(42)
    all_rows = []

    for i in range(N_BLOCKS):

        block_seed = rng.randint(0, 100000)

        gen_A = synth.SEA(variant=0, noise=0.05, seed=block_seed)
        gen_B = synth.SEA(variant=2, noise=0.05, seed=block_seed)

        stream_A = list(gen_A.take(SAMPLES_PER_BLOCK))
        stream_B = list(gen_B.take(SAMPLES_PER_BLOCK))

        for j, ((xA, yA), (xB, yB)) in enumerate(zip(stream_A, stream_B)):

            # -------------------------
            # BEFORE DRIFT
            # -------------------------
            if i < 10:
                x, y = xA, yA
                concept = 0

            # -------------------------
            # GRADUAL TRANSITION
            # -------------------------
            elif 10 <= i < 18:

                alpha_block = (i - 10) / (18 - 10)
                alpha_sample = j / SAMPLES_PER_BLOCK

                # transición suave
                p = alpha_block * 0.7 + alpha_sample * 0.3

                if random.random() < p:
                    x, y = xB, yB
                    concept = 2
                else:
                    x, y = xA, yA
                    concept = 0

            # -------------------------
            # AFTER DRIFT
            # -------------------------
            else:
                x, y = xB, yB
                concept = 2

            row = dict(x)
            row["target"] = y
            row["block"] = i + 1
            row["concept"] = concept

            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    path = os.path.join(OUTPUT_DIR, "sea_gradual.csv")
    df.to_csv(path, index=False)

    print(f"Dataset generado: {path}")


# ------------------------------------------------
# RECURRENT
# ------------------------------------------------

def generate_recurrent():

    rng = np.random.RandomState(42)

    generators = []
    concepts = []

    pattern = (
        [0]*5 +
        [2]*5 +
        [0]*4 +
        [2]*4 +
        [0]*4 +
        [2]*3
    )

    for concept in pattern:

        block_seed = rng.randint(0, 100000)

        generators.append(
            synth.SEA(
                variant=concept,
                noise=0.05,
                seed=block_seed
            )
        )

        concepts.append(concept)

    generate_dataset("sea_recurrent.csv", generators, concepts)


# ------------------------------------------------
# MAIN
# ------------------------------------------------

if __name__ == "__main__":
    generate_abrupt()
    generate_gradual()
    generate_recurrent()