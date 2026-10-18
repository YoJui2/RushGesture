"""Carga, validacion, featurizacion y division del dataset de landmarks.

El dataset se guarda en CSV con:
- 63 columnas de landmarks crudos (`lm_{i}_x/y/z`).
- Metadatos: ``label``, ``participant``, ``session``, ``handedness``, ``source``.

Las features se calculan en el momento de cargar, con el mismo pipeline que en
tiempo real (`features.normalize_landmarks`), evitando discrepancias.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features

METADATA_COLUMNS = ["label", "participant", "session", "handedness", "source"]


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Carga el dataset desde CSV."""
    path = Path(path or config.DATASET_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el dataset en {path}. "
            "Genera datos con scripts/run_generate_synthetic.py o recolecta con "
            "scripts/run_collect.py."
        )
    return pd.read_csv(path)


def save_dataset(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Guarda el dataset en CSV, creando el directorio si hace falta."""
    path = Path(path or config.DATASET_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def validate_dataset(df: pd.DataFrame, gestures: tuple[str, ...] | None = None) -> dict:
    """Valida el dataset y devuelve un informe.

    Comprueba columnas requeridas, etiquetas conocidas, valores finitos y
    balance de clases. Lanza ``ValueError`` ante problemas graves.
    """
    gestures = tuple(gestures or config.GESTURES)
    lm_cols = features.landmark_columns()
    missing = [c for c in lm_cols + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    unknown = sorted(set(df["label"].unique()) - set(gestures))
    if unknown:
        raise ValueError(f"Etiquetas desconocidas en el dataset: {unknown}")

    if df.empty:
        raise ValueError("El dataset esta vacio")

    values = df[lm_cols].to_numpy(dtype=np.float64)
    n_nonfinite = int(np.size(values) - np.isfinite(values).sum())
    if n_nonfinite:
        raise ValueError(f"El dataset contiene {n_nonfinite} valores no finitos")

    counts = {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}
    report = {
        "n_samples": int(len(df)),
        "class_counts": counts,
        "n_participants": int(df["participant"].nunique())
        if "participant" in df.columns
        else 0,
        "n_sessions": int(df["session"].nunique()) if "session" in df.columns else 0,
        "sources": {str(k): int(v) for k, v in df["source"].value_counts().to_dict().items()}
        if "source" in df.columns
        else {},
        "n_nonfinite": n_nonfinite,
    }
    return report


def check_class_balance(df: pd.DataFrame, min_per_class: int = 10) -> dict:
    """Comprueba que cada clase tenga un minimo de muestras."""
    counts = df["label"].value_counts()
    insufficient = {
        label: int(counts.get(label, 0)) for label in config.GESTURES
        if int(counts.get(label, 0)) < min_per_class
    }
    return {"insufficient": insufficient, "counts": counts.to_dict()}


def featurize(
    df: pd.DataFrame, use_z: bool | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convierte el ``DataFrame`` en ``(X, y, groups)``.

    ``X``: matriz (n, dim) de features normalizadas.
    ``y``: etiquetas (str).
    ``groups``: identificador de grupo (participante) para division agrupada.
    """
    if use_z is None:
        use_z = config.USE_Z
    lm_cols = features.landmark_columns()
    raw = df[lm_cols].to_numpy(dtype=np.float64).reshape(-1, features.N_LANDMARKS, 3)
    X = np.stack([features.normalize_landmarks(sample, use_z=use_z) for sample in raw])
    y = df["label"].to_numpy()
    groups = (
        df["participant"].to_numpy()
        if "participant" in df.columns
        else np.array(["_"] * len(df))
    )
    return X, y, groups


def group_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = config.TEST_SIZE,
    validation_size: float = config.VALIDATION_SIZE,
    seed: int = config.RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """Divide por grupo (participante) en train/val/test.

    Reserva ``test_size`` de los grupos para prueba y, del resto,
    ``validation_size`` para validacion. Nunca mezcla muestras de un mismo
    participante entre particiones.
    """
    from sklearn.model_selection import GroupShuffleSplit

    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError(
            "Se necesitan al menos 3 participantes/grupos para separar "
            f"train/val/test de forma agrupada; hay {len(unique_groups)}. "
            "Recolecta datos de mas personas o usa el generador sintetico."
        )

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(X, y, groups))

    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=seed
    )
    g_tv, y_tv, g_tv_groups = X[trainval_idx], y[trainval_idx], groups[trainval_idx]
    rel_train_idx, rel_val_idx = next(gss_val.split(g_tv, y_tv, g_tv_groups))

    train_idx = trainval_idx[rel_train_idx]
    val_idx = trainval_idx[rel_val_idx]
    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "test_groups": np.unique(groups[test_idx]),
        "train_groups": np.unique(groups[train_idx]),
        "val_groups": np.unique(groups[val_idx]),
    }
