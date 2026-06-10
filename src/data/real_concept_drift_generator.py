from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from river.datasets import synth

from calm_data_generator.generators.stream.StreamBlockGenerator import (
    SyntheticBlockGenerator,
)


@dataclass(frozen=True)
class DriftDatasetConfig:
    output_dir: str = "datasets_real"
    total_samples: int = 20_000
    chunk_size: int = 500
    seed: int = 42

    # Ruido base suave de Agrawal. No es el drift principal.
    base_perturbation: float = 0.01

    # Severidad máxima del concept drift.
    # 0.25 significa que, como máximo, se flippea el 25% de la subpoblación afectada.
    max_flip_prob: float = 0.50

    # Regla basada en percentiles.
    # salary_quantile=0.70 => salary por encima del percentil 70.
    # loan_quantile=0.70 => loan por encima del percentil 70.
    salary_quantile: float = 0.60
    loan_quantile: float = 0.60

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

    Mantenemos classification_function=0 en todos los bloques.
    El real concept drift lo introducimos después modificando target
    de forma condicionada y probabilística.
    """

    return [
        synth.Agrawal(
            classification_function=0,
            perturbation=cfg.base_perturbation,
            seed=cfg.seed + block_idx,
        )
        for block_idx in range(cfg.n_blocks)
    ]


def generate_base_dataset(cfg: DriftDatasetConfig) -> str:
    """
    Genera el dataset base por bloques usando SyntheticBlockGenerator.
    """

    os.makedirs(cfg.output_dir, exist_ok=True)

    block_gen = SyntheticBlockGenerator()

    path = block_gen.generate(
        output_dir=cfg.output_dir,
        filename="agrawal_base.csv",
        n_blocks=cfg.n_blocks,
        total_samples=cfg.total_samples,
        n_samples_block=[cfg.chunk_size] * cfg.n_blocks,
        generators=_base_agrawal_generators(cfg),
        target_col="target",
        generate_report=False,
    )

    df = pd.read_csv(path)
    df["scenario"] = "base"
    df["base_concept"] = 0
    df["chunk_size"] = cfg.chunk_size
    df.to_csv(path, index=False)

    print(
        f"Dataset base generado: {path} "
        f"({len(df)} filas, {df['block'].nunique()} bloques)"
    )

    return path


def compute_rule_thresholds(df: pd.DataFrame, cfg: DriftDatasetConfig) -> dict[str, float]:
    """
    Calcula umbrales basados en percentiles del dataset base.

    Esto evita fijar a mano valores como salary > 100000 o loan > 300000
    sin saber previamente qué porcentaje de filas afectan.
    """

    return {
        "salary_threshold": float(df["salary"].quantile(cfg.salary_quantile)),
        "loan_threshold": float(df["loan"].quantile(cfg.loan_quantile)),
    }


def affected_subpopulation_mask(
    df: pd.DataFrame,
    salary_threshold: float,
    loan_threshold: float,
) -> pd.Series:
    """
    Regla de subpoblación afectada.

    Real concept drift local:
    cambiaremos P(y|X) solo para clientes con salario alto y préstamo alto.
    """

    return (
        (df["salary"] > salary_threshold)
        & (df["loan"] > loan_threshold)
    )


def build_abrupt_probabilities(cfg: DriftDatasetConfig) -> np.ndarray:
    """
    Abrupt real concept drift.

    Bloques 1-20: p = 0
    Bloques 21-40: p = max_flip_prob
    """

    probs = np.zeros(cfg.total_samples, dtype=float)

    # 20 bloques * 500 filas = 10_000
    start_index = 10_000
    probs[start_index:] = cfg.max_flip_prob

    return probs


def build_gradual_probabilities(cfg: DriftDatasetConfig) -> np.ndarray:
    """
    Gradual real concept drift.

    Bloques 1-10: p = 0
    Bloques 11-30: p sube linealmente de 0 a max_flip_prob
    Bloques 31-40: p = max_flip_prob
    """

    probs = np.zeros(cfg.total_samples, dtype=float)

    start = 5_000    # inicio bloque 11
    end = 15_000     # inicio bloque 31

    for idx in range(cfg.total_samples):
        if idx < start:
            probs[idx] = 0.0
        elif start <= idx < end:
            alpha = (idx - start) / max(end - start, 1)
            probs[idx] = alpha * cfg.max_flip_prob
        else:
            probs[idx] = cfg.max_flip_prob

    return probs


def build_recurrent_probabilities(cfg: DriftDatasetConfig) -> np.ndarray:
    """
    Recurrent real concept drift.

    El drift aparece en episodios recurrentes.
    Dentro de cada episodio, la probabilidad sube y baja suavemente.

    Ventanas:
    - bloques 9-14
    - bloques 19-24
    - bloques 29-34
    """

    probs = np.zeros(cfg.total_samples, dtype=float)

    recurrent_windows_blocks = [
        (9, 14),
        (19, 24),
        (29, 34),
    ]

    for start_block, end_block in recurrent_windows_blocks:
        start_idx = (start_block - 1) * cfg.chunk_size
        end_idx = end_block * cfg.chunk_size
        length = end_idx - start_idx

        for local_i, idx in enumerate(range(start_idx, end_idx)):
            alpha = local_i / max(length - 1, 1)

            # Forma triangular:
            # 0 al inicio, 1 en el centro, 0 al final.
            triangular = 1.0 - abs(2.0 * alpha - 1.0)

            probs[idx] = max(
                probs[idx],
                triangular * cfg.max_flip_prob,
            )

    return probs


def apply_probabilistic_real_concept_drift(
    df: pd.DataFrame,
    scenario_name: str,
    flip_prob_by_row: np.ndarray,
    salary_threshold: float,
    loan_threshold: float,
    seed: int,
) -> pd.DataFrame:
    """
    Aplica real concept drift modificando el target.

    Como no asumimos qué significa target=1, usamos flip simétrico:

        target = 1 - target

    Esto cambia P(y|X) en la subpoblación afectada sin asumir que la clase 1
    sea necesariamente "riesgo", "default", "positivo", etc.
    """

    drifted = df.copy()
    rng = np.random.default_rng(seed)

    affected_mask = affected_subpopulation_mask(
        drifted,
        salary_threshold=salary_threshold,
        loan_threshold=loan_threshold,
    )

    candidate_indices = drifted.index[affected_mask].to_numpy()

    drifted["concept_drift_affected"] = False
    drifted.loc[candidate_indices, "concept_drift_affected"] = True

    drifted["flip_probability"] = 0.0
    drifted.loc[candidate_indices, "flip_probability"] = flip_prob_by_row[candidate_indices]

    drifted["concept_drift_flipped"] = False

    if len(candidate_indices) > 0:
        candidate_probs = flip_prob_by_row[candidate_indices]
        random_values = rng.random(len(candidate_indices))
        flip_indices = candidate_indices[random_values < candidate_probs]

        # Flip simétrico: no asumimos semántica de la clase.
        drifted.loc[flip_indices, "target"] = 1 - drifted.loc[flip_indices, "target"]
        drifted.loc[flip_indices, "concept_drift_flipped"] = True

    drifted["scenario"] = scenario_name
    drifted["salary_threshold"] = salary_threshold
    drifted["loan_threshold"] = loan_threshold

    return drifted


def save_scenario(
    df: pd.DataFrame,
    output_dir: str,
    filename: str,
) -> str:
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)

    print(
        f"Dataset generado: {path} "
        f"({len(df)} filas, {df['block'].nunique()} bloques)"
    )

    return path


def generate_abrupt_real_concept(
    base_df: pd.DataFrame,
    thresholds: dict[str, float],
    cfg: DriftDatasetConfig,
) -> str:
    probs = build_abrupt_probabilities(cfg)

    drifted = apply_probabilistic_real_concept_drift(
        df=base_df,
        scenario_name="abrupt_real_concept_drift",
        flip_prob_by_row=probs,
        salary_threshold=thresholds["salary_threshold"],
        loan_threshold=thresholds["loan_threshold"],
        seed=cfg.seed + 100,
    )

    return save_scenario(
        drifted,
        cfg.output_dir,
        "agrawal_abrupt_real_concept_drift.csv",
    )


def generate_gradual_real_concept(
    base_df: pd.DataFrame,
    thresholds: dict[str, float],
    cfg: DriftDatasetConfig,
) -> str:
    probs = build_gradual_probabilities(cfg)

    drifted = apply_probabilistic_real_concept_drift(
        df=base_df,
        scenario_name="gradual_real_concept_drift",
        flip_prob_by_row=probs,
        salary_threshold=thresholds["salary_threshold"],
        loan_threshold=thresholds["loan_threshold"],
        seed=cfg.seed + 200,
    )

    return save_scenario(
        drifted,
        cfg.output_dir,
        "agrawal_gradual_real_concept_drift.csv",
    )


def generate_recurrent_real_concept(
    base_df: pd.DataFrame,
    thresholds: dict[str, float],
    cfg: DriftDatasetConfig,
) -> str:
    probs = build_recurrent_probabilities(cfg)

    drifted = apply_probabilistic_real_concept_drift(
        df=base_df,
        scenario_name="recurrent_real_concept_drift",
        flip_prob_by_row=probs,
        salary_threshold=thresholds["salary_threshold"],
        loan_threshold=thresholds["loan_threshold"],
        seed=cfg.seed + 300,
    )

    return save_scenario(
        drifted,
        cfg.output_dir,
        "agrawal_recurrent_real_concept_drift.csv",
    )


def quick_validate(path: str, cfg: DriftDatasetConfig) -> None:
    df = pd.read_csv(path)

    assert len(df) == cfg.total_samples
    assert df["block"].nunique() == cfg.n_blocks
    assert "target" in df.columns
    assert "scenario" in df.columns

    if df["scenario"].iloc[0] != "base":
        assert "concept_drift_affected" in df.columns
        assert "concept_drift_flipped" in df.columns
        assert "flip_probability" in df.columns

    print(f"\nValidado: {path}")
    print("Filas:", len(df))
    print("Bloques:", df["block"].nunique())

    if "concept_drift_affected" in df.columns:
        print("Subpoblación afectada:", df["concept_drift_affected"].mean())
        print("Targets fliplados:", df["concept_drift_flipped"].mean())
        print("Probabilidad media de flip:", df["flip_probability"].mean())


def print_scenario_summary(path: str) -> None:
    df = pd.read_csv(path)

    print(f"\nResumen por bloque: {path}")

    cols = ["target"]
    if "flip_probability" in df.columns:
        cols += ["flip_probability", "concept_drift_flipped"]

    print(
        df.groupby("block")[cols]
        .mean()
        .head(12)
    )

    print("...")

    print(
        df.groupby("block")[cols]
        .mean()
        .tail(12)
    )


def main() -> None:
    cfg = DriftDatasetConfig(
        output_dir="datasets_real_concept_drift",
        total_samples=20_000,
        chunk_size=500,
        seed=42,
        base_perturbation=0.01,
        max_flip_prob=0.60,
        salary_quantile=0.30,
        loan_quantile=0.40,
    )

    base_path = generate_base_dataset(cfg)
    base_df = pd.read_csv(base_path)

    thresholds = compute_rule_thresholds(base_df, cfg)

    print("\nUmbrales de regla:")
    print(thresholds)

    affected = affected_subpopulation_mask(
        base_df,
        salary_threshold=thresholds["salary_threshold"],
        loan_threshold=thresholds["loan_threshold"],
    )

    print(f"Porcentaje afectado por la regla: {affected.mean():.2%}")

    abrupt_path = generate_abrupt_real_concept(base_df, thresholds, cfg)
    gradual_path = generate_gradual_real_concept(base_df, thresholds, cfg)
    recurrent_path = generate_recurrent_real_concept(base_df, thresholds, cfg)

    for path in [base_path, abrupt_path, gradual_path, recurrent_path]:
        quick_validate(path, cfg)

    for path in [abrupt_path, gradual_path, recurrent_path]:
        print_scenario_summary(path)

    print("\nGeneración de real concept drift completada.")
    print(f"Datasets guardados en: {cfg.output_dir}")


if __name__ == "__main__":
    main()