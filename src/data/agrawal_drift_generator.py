from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from river.datasets import synth

from calm_data_generator.generators.configs import DriftConfig
from calm_data_generator.generators.stream.StreamBlockGenerator import (
    SyntheticBlockGenerator,
)


@dataclass(frozen=True)
class DriftDatasetConfig:
    output_dir: str = "datasets_controlled"
    total_samples: int = 20_000
    chunk_size: int = 500
    seed: int = 42

    # Ruido base suave del propio Agrawal.
    # No es el drift experimental principal.
    base_perturbation: float = 0.01

    @property
    def n_blocks(self) -> int:
        if self.total_samples % self.chunk_size != 0:
            raise ValueError(
                "total_samples debe ser múltiplo de chunk_size para bloques homogéneos."
            )
        return self.total_samples // self.chunk_size


def _base_agrawal_generators(cfg: DriftDatasetConfig) -> list:
    """
    Generadores base homogéneos.

    Importante:
    - Mantenemos classification_function=0 en todos los bloques.
    - Así evitamos convertir el problema en una clase/concepto completamente distinto.
    - El drift experimental lo introduce DriftConfig.
    """

    return [
        synth.Agrawal(
            classification_function=0,
            perturbation=cfg.base_perturbation,
            seed=cfg.seed + block_idx,
        )
        for block_idx in range(cfg.n_blocks)
    ]


def _write_dataset(
    filename: str,
    cfg: DriftDatasetConfig,
    drift_config: list[DriftConfig],
    scenario_name: str,
) -> str:
    """
    Genera un dataset stream por bloques usando SyntheticBlockGenerator
    y aplica drift mediante DriftConfig.
    """

    os.makedirs(cfg.output_dir, exist_ok=True)

    block_gen = SyntheticBlockGenerator()

    path = block_gen.generate(
        output_dir=cfg.output_dir,
        filename=filename,
        n_blocks=cfg.n_blocks,
        total_samples=cfg.total_samples,
        n_samples_block=[cfg.chunk_size] * cfg.n_blocks,
        generators=_base_agrawal_generators(cfg),
        target_col="target",
        drift_config=drift_config,
        generate_report=False,
    )

    # Añadimos metadata útil para tus evaluaciones.
    # La generación y el drift ya los ha hecho la librería.
    df = pd.read_csv(path)
    df["scenario"] = scenario_name
    df["base_concept"] = 0
    df["chunk_size"] = cfg.chunk_size
    df.to_csv(path, index=False)

    print(
        f"Dataset generado: {path} "
        f"({len(df)} filas, {df['block'].nunique()} bloques)"
    )

    return path


def generate_abrupt(cfg: DriftDatasetConfig) -> str:
    """
    Drift abrupto controlado.

    20_000 muestras, 40 bloques, 500 muestras/bloque.

    El cambio empieza en la fila 10_000, que equivale al inicio del bloque 21:
    - bloques 1-20: distribución base
    - bloques 21-40: distribución con ruido gaussiano en salary, age y loan
    """

    drift_config = [
        DriftConfig(
            method="inject_feature_drift",
            params={
                "feature_cols": ["salary", "age", "loan"],
                "drift_type": "gaussian_noise",
                "drift_magnitude": 0.05,
                "start_index": 10_000,
            },
        )
    ]

    return _write_dataset(
        filename="agrawal_abrupt_feature_drift.csv",
        cfg=cfg,
        drift_config=drift_config,
        scenario_name="abrupt_feature_drift",
    )


def generate_gradual(cfg: DriftDatasetConfig) -> str:
    """
    Drift gradual controlado.

    Transición entre la fila 6_000 y la fila 14_000:
    - bloque 1-12: distribución base
    - bloque 13-28: transición gradual
    - bloque 29-40: distribución drifted

    Como cada bloque tiene 500 muestras:
    - fila 6_000  = inicio aproximado del bloque 13
    - fila 14_000 = inicio aproximado del bloque 29
    """

    drift_config = [
        DriftConfig(
            method="inject_feature_drift_gradual",
            params={
                "feature_cols": ["salary", "age", "loan"],
                "drift_type": "gaussian_noise",
                "drift_magnitude": 0.05,
                "start_index": 6_000,
                "end_index": 14_000,

                # Ojo: en el método gradual, center y width se aplican sobre
                # las filas seleccionadas por start_index/end_index.
                # Aquí seleccionamos 8_000 filas, así que center=4_000
                # queda en el centro de la transición.
                "center": 4_000,
                "width": 8_000,
                "profile": "sigmoid",
                "speed_k": 0.7,
            },
        )
    ]

    return _write_dataset(
        filename="agrawal_gradual_feature_drift.csv",
        cfg=cfg,
        drift_config=drift_config,
        scenario_name="gradual_feature_drift",
    )


def generate_recurrent(cfg: DriftDatasetConfig) -> str:
    """
    Drift recurrente controlado.

    Se aplica drift en 4 recurrencias sobre el stream.
    No cambia la classification_function de Agrawal.
    """

    drift_config = [
        DriftConfig(
            method="inject_feature_drift_recurrent",
            params={
                "feature_cols": ["salary", "loan", "age"],
                "drift_type": "gaussian_noise",
                "drift_magnitude": 0.04,
                "start_index": 0,
                "repeats": 4,
                "profile": "cosine",
                "speed_k": 0.7,
            },
        )
    ]

    return _write_dataset(
        filename="agrawal_recurrent_feature_drift.csv",
        cfg=cfg,
        drift_config=drift_config,
        scenario_name="recurrent_feature_drift",
    )


def split_in_chunks(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    """Helper útil para evaluación incremental chunk-by-chunk."""

    return [
        df.iloc[i : i + chunk_size].reset_index(drop=True)
        for i in range(0, len(df), chunk_size)
    ]


def _quick_validate(path: str, cfg: DriftDatasetConfig) -> None:
    df = pd.read_csv(path)
    chunks = split_in_chunks(df, cfg.chunk_size)

    assert len(df) == cfg.total_samples
    assert len(chunks) == cfg.n_blocks
    assert "block" in df.columns
    assert "target" in df.columns
    assert "scenario" in df.columns
    assert df["block"].nunique() == cfg.n_blocks

    block_sizes = df["block"].value_counts().sort_index()
    assert (block_sizes == cfg.chunk_size).all()

    print(f"\nValidado: {path}")
    print(f"Filas: {len(df)}")
    print(f"Bloques: {df['block'].nunique()}")
    print("Tamaño de los primeros bloques:")
    print(block_sizes.head())

    # Resumen rápido para ver si el drift está afectando features.
    numeric_cols = [
        col
        for col in ["salary", "age", "loan"]
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if numeric_cols:
        print("\nMedia por bloque de features afectadas:")
        print(df.groupby("block")[numeric_cols].mean().head(10))
        print("...")
        print(df.groupby("block")[numeric_cols].mean().tail(10))


def main() -> None:
    cfg = DriftDatasetConfig(
        output_dir="datasets_controlled",
        total_samples=20_000,
        chunk_size=500,
        seed=42,
        base_perturbation=0.01,
    )

    abrupt_path = generate_abrupt(cfg)
    gradual_path = generate_gradual(cfg)
    recurrent_path = generate_recurrent(cfg)

    for path in [abrupt_path, gradual_path, recurrent_path]:
        _quick_validate(path, cfg)

    print("\nGeneración completada y validada.")
    print(f"Datasets guardados en: {cfg.output_dir}")


if __name__ == "__main__":
    main()