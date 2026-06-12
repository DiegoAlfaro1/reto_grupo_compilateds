"""
pipeline/build_snippets.py
──────────────────────────
Adaptación de `build_pairs.py` al nuevo dataset en parquet (new_data/):
lee un parquet con columnas [code, generator, label, language] y produce
otro parquet con las 16 características de `snippet_features.py` + label.

La tarea ya no es comparar pares de archivos sino clasificar cada
snippet individual (0 = humano, 1 = generado por máquina), así que cada
fila del parquet de entrada produce una fila de características.

Uso
---
    python -m pipeline.build_snippets \
        --input new_data/task_a_training_set_1.parquet \
        --output new_data_features/train_features.parquet \
        --k 15 --w 4 --masking medium --workers 8

    # Para experimentar rápido con una muestra:
    python -m pipeline.build_snippets --input ... --output ... --sample 50000
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from pipeline.snippet_features import (
    SNIPPET_FEATURE_COLUMNS,
    compute_snippet_features,
)

RANDOM_SEED = 42


def _featurize_row(row: tuple[str, str], k: int, w: int, masking: str) -> dict:
    """Worker: (code, language) → dict de características."""
    code, language = row
    try:
        return compute_snippet_features(code, language=language, k=k, w=w, masking=masking)
    except Exception:
        # Snippet patológico: vector neutro en lugar de tirar todo el lote
        return {col: 0.0 for col in SNIPPET_FEATURE_COLUMNS}


def build_features(
    input_path: str | Path,
    output_path: str | Path,
    k: int = 15,
    w: int = 4,
    masking: str = "medium",
    sample: int | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Calcula las características de cada snippet del parquet de entrada."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_parquet(input_path)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=RANDOM_SEED).reset_index(drop=True)

    rows = list(zip(df["code"].fillna(""), df["language"].fillna("")))
    job = partial(_featurize_row, k=k, w=w, masking=masking)

    start = time.time()
    if workers > 1:
        with Pool(processes=workers) as pool:
            feats = pool.map(job, rows, chunksize=512)
    else:
        feats = [job(r) for r in rows]
    elapsed = time.time() - start

    out = pd.DataFrame(feats, columns=SNIPPET_FEATURE_COLUMNS)
    out["label"] = df["label"].values
    out["language"] = df["language"].values
    out["generator"] = df["generator"].values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(
        f"{input_path.name}: {len(out)} snippets → {output_path} "
        f"({elapsed:.1f}s, {len(out) / max(elapsed, 1e-9):.0f} snippets/s)"
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Genera características por snippet desde un parquet."
    )
    parser.add_argument("--input", required=True, help="parquet de entrada")
    parser.add_argument("--output", required=True, help="parquet de salida")
    parser.add_argument("--k", type=int, default=15, help="tamaño de k-grama")
    parser.add_argument("--w", type=int, default=4, help="ventana de Winnowing")
    parser.add_argument(
        "--masking", default="medium", choices=["low", "medium", "high"]
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="muestrear N filas (None = todas)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="procesos en paralelo (recomendado: nº de núcleos)",
    )
    args = parser.parse_args(argv)

    build_features(
        args.input,
        args.output,
        k=args.k,
        w=args.w,
        masking=args.masking,
        sample=args.sample,
        workers=args.workers,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
