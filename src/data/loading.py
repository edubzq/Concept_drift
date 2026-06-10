from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

TARGET_COL = "target"
BLOCK_COL = "block"

# Columnas añadidas por los generadores para trazabilidad/evaluación. No deben
# entrar como variables predictoras porque filtran información del escenario de
# drift o son constantes técnicas del dataset.
DEFAULT_METADATA_COLUMNS = {
    TARGET_COL,
    BLOCK_COL,
    "timestamp",
    # Metadata del generador antiguo basado en mezcla de conceptos.
    "concept",
    "p_concept_b",
    # Metadata del generador actual basado en DriftConfig.
    "scenario",
    "base_concept",
    "chunk_size",
        # Metadata de los datasets de real concept drift. Son útiles para auditar
    # la generación, pero no deben entrar en X porque revelarían información
    # experimental/oráculo sobre dónde y con qué intensidad se aplicó el drift.
    "concept_drift_affected",
    "concept_drift_flipped",
    "flip_probability",
    "salary_threshold",
    "loan_threshold",
}


def _feature_columns(
    df: pd.DataFrame,
    metadata_cols: Iterable[str] = DEFAULT_METADATA_COLUMNS,
) -> list[str]:
    metadata = set(metadata_cols)
    return [col for col in df.columns if col not in metadata]


def load_blocks(csv_path, metadata_cols: Iterable[str] = DEFAULT_METADATA_COLUMNS):
    """
    Carga un CSV generado por SyntheticBlockGenerator como lista de bloques.

    La función soporta tanto los datasets antiguos (con ``concept`` y
    ``p_concept_b``) como los nuevos datasets controlados (con ``scenario``,
    ``base_concept`` y ``chunk_size``). Todas esas columnas son metadata y se
    excluyen de X para evitar leakage y para que los experimentos comparen los
    ensembles usando solo las features Agrawal reales.
    """
    df = pd.read_csv(csv_path)

    missing_required = {TARGET_COL, BLOCK_COL} - set(df.columns)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise ValueError(
            f"El dataset {csv_path!r} no contiene columnas requeridas: {missing}."
        )

    feature_cols = _feature_columns(df, metadata_cols=metadata_cols)
    if not feature_cols:
        raise ValueError(f"El dataset {csv_path!r} no contiene columnas de features.")

    chunks = []
    for block in sorted(df[BLOCK_COL].unique()):
        block_df = df[df[BLOCK_COL] == block]
        X = block_df[feature_cols]
        y = block_df[TARGET_COL].values
        chunks.append((X, y))

    return chunks