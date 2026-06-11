from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from river.datasets import synth

from calm_data_generator.generators.stream.StreamBlockGenerator import (
    SyntheticBlockGenerator,
)


def classification_function(x: dict) -> int:
    age = x["age"]
    elevel = x["elevel"]

    if age < 40:
        # En función 0 todos los menores de 40 eran clase 1.
        # Ahora excluimos dos niveles educativos, no solo uno.
        return int(elevel in [0, 1, 2])

    elif age < 60:
        # En función 0 este grupo era clase 0.
        # Ahora algunos casos pasan a clase 1.
        return int(elevel in [1, 2])

    else:
        # En función 0 todos los mayores de 60 eran clase 1.
        # Ahora excluimos dos niveles educativos.
        return int(elevel in [2, 3, 4])


def replace_concept_b_targets(df: pd.DataFrame, concept_b: int = 2) -> pd.DataFrame:
    concept_b_mask = df["concept"] == concept_b
    df.loc[concept_b_mask, "target"] = df.loc[concept_b_mask].apply(
        classification_function, axis=1
    )
    return df


@dataclass(frozen=True)
class DriftDatasetConfig:
    output_dir: str = "datasets"
    total_samples: int = 25_000
    chunk_size: int = 500
    seed: int = 42

    @property
    def n_blocks(self) -> int:
        if self.total_samples % self.chunk_size != 0:
            raise ValueError(
                "total_samples debe ser múltiplo de chunk_size para bloques homogéneos."
            )
        return self.total_samples // self.chunk_size


def _write_dataset(
    filename: str,
    generators: list,
    concepts: list[int],
    cfg: DriftDatasetConfig,
) -> str:
    """Genera dataset por bloques, añade columna concept y guarda CSV final."""
    if len(generators) != cfg.n_blocks:
        raise ValueError("El número de generadores debe coincidir con n_blocks.")
    if len(concepts) != cfg.n_blocks:
        raise ValueError("La longitud de concepts debe coincidir con n_blocks.")

    os.makedirs(cfg.output_dir, exist_ok=True)

    gen = SyntheticBlockGenerator()
    path = gen.generate(
        output_dir=cfg.output_dir,
        filename=filename,
        n_blocks=cfg.n_blocks,
        total_samples=cfg.total_samples,
        n_samples_block=[cfg.chunk_size] * cfg.n_blocks,
        generators=generators,
        generate_report=False,
    )

    df = pd.read_csv(path)

    # block generado por SyntheticBlockGenerator empieza en 1
    concept_by_block = {block_idx + 1: concept for block_idx, concept in enumerate(concepts)}
    df["concept"] = df["block"].map(concept_by_block).astype(int)
    replace_concept_b_targets(df)
    df.to_csv(path, index=False)

    print(f"Dataset generado: {path} ({len(df)} filas, {cfg.n_blocks} bloques)")
    return path


def generate_abrupt(cfg: DriftDatasetConfig) -> str:
    
    concept_a, concept_b = 0, 2
    drift_start = 20
    concepts = [concept_a] * drift_start + [concept_b] * (cfg.n_blocks - drift_start)
    generators = [
        synth.Agrawal(classification_function=concept, seed=cfg.seed + i)
        for i, concept in enumerate(concepts)
    ]

    return _write_dataset("agrawal_abrupt.csv", generators, concepts, cfg)


def generate_gradual(cfg: DriftDatasetConfig) -> str:
  
    os.makedirs(cfg.output_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    concept_a, concept_b = 0, 2

    # Ventana de transición sobre bloques
    start_transition_block = 15
    end_transition_block = 30

    rows: list[dict] = []

    for block_idx in range(cfg.n_blocks):
        seed_block = cfg.seed + block_idx

        gen_a = synth.Agrawal(classification_function=concept_a, seed=seed_block)
        gen_b = synth.Agrawal(classification_function=concept_b, seed=seed_block)

        stream_a = list(gen_a.take(cfg.chunk_size))
        stream_b = list(gen_b.take(cfg.chunk_size))

        for j, ((x_a, y_a), (x_b, y_b)) in enumerate(zip(stream_a, stream_b)):
            # Fase estable A
            if block_idx < start_transition_block:
                x, y, concept = x_a, y_a, concept_a
            # Fase de transición
            elif start_transition_block <= block_idx < end_transition_block:
                alpha_block = (block_idx - start_transition_block) / (
                    end_transition_block - start_transition_block
                )
                alpha_sample = j / (cfg.chunk_size - 1)

                # 70% cambio entre bloques + 30% cambio dentro del bloque
                p_switch_to_b = 0.7 * alpha_block + 0.3 * alpha_sample

                if rng.random() < p_switch_to_b:
                    x, y, concept = x_b, y_b, concept_b
                else:
                    x, y, concept = x_a, y_a, concept_a
            # Fase estable B
            else:
                x, y, concept = x_b, y_b, concept_b

            row = dict(x)
            row["target"] = y
            row["block"] = block_idx + 1
            row["concept"] = concept
            rows.append(row)

    df = pd.DataFrame(rows)
    replace_concept_b_targets(df)
    path = os.path.join(cfg.output_dir, "agrawal_gradual.csv")
    df.to_csv(path, index=False)

    print(f"Dataset generado: {path} ({len(df)} filas, {cfg.n_blocks} bloques)")
    return path


def generate_recurrent(cfg: DriftDatasetConfig) -> str:

    concept_a, concept_b = 0, 2

    pattern = (
        [concept_a] * 10
        + [concept_b] * 8
        + [concept_a] * 6
        + [concept_b] * 6
        + [concept_a] * 6
        + [concept_b] * 6
        + [concept_a] * 8

    )

    concepts = pattern[: cfg.n_blocks]
    if len(concepts) < cfg.n_blocks:
        concepts += [concept_b] * (cfg.n_blocks - len(concepts))

    rng = np.random.default_rng(cfg.seed)
    generators = [
        synth.Agrawal(classification_function=concept, seed=int(rng.integers(0, 1_000_000)))
        for concept in concepts
    ]

    return _write_dataset("agrawal_recurrent.csv", generators, concepts, cfg)


def split_in_chunks(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    """Helper útil para evaluación incremental chunk-by-chunk."""
    return [df.iloc[i : i + chunk_size].reset_index(drop=True) for i in range(0, len(df), chunk_size)]


def _quick_validate(path: str, cfg: DriftDatasetConfig) -> None:
    df = pd.read_csv(path)
    chunks = split_in_chunks(df, cfg.chunk_size)
    assert len(df) == cfg.total_samples
    assert len(chunks) == cfg.n_blocks
    assert "concept" in df.columns


def main() -> None:
    cfg = DriftDatasetConfig()

    abrupt_path = generate_abrupt(cfg)
    gradual_path = generate_gradual(cfg)
    recurrent_path = generate_recurrent(cfg)

    for path in [abrupt_path, gradual_path, recurrent_path]:
        _quick_validate(path, cfg)

    print("Generación completada y validada para abrupt, gradual y recurrent.")


if __name__ == "__main__":
    main()