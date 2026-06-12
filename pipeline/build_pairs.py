"""
pipeline/build_pairs.py
───────────────────────
Genera el archivo pairs_features.csv que consume model.ipynb.

Recorre el dataset de Kaggle, construye pares etiquetados y calcula
las 5 características por par usando el pipeline completo:
    ast_tokenizer → winnowing → features

Estructura esperada del dataset
--------------------------------
    data_dir/
        problema_1/
            snippets/
                snip_01.py
                snip_02.py
                ...
        problema_2/
            snippets/
                ...

Etiquetas
---------
    1  →  par positivo: dos .py del MISMO problema
    0  →  par negativo: dos .py de problemas DISTINTOS

Uso desde línea de comandos
---------------------------
    python -m pipeline.build_pairs \\
        --data-dir ./data/python_variants \\
        --output   pairs_features.csv \\
        --k        23 \\
        --w        4  \\
        --masking  medium

Uso desde Python
----------------
    from pipeline.build_pairs import build_pairs_csv

    build_pairs_csv(
        data_dir="./data/python_variants",
        output="pairs_features.csv",
        k=23, w=4, masking="medium",
    )
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
from pathlib import Path

from pipeline.ast_tokenizer import tokenize_file
from pipeline.winnowing import winnow
from pipeline.features import compute_features, FEATURE_COLUMNS


# ──────────────────────────────────────────────────────────────────────────────
# Recolección de archivos por problema
# ──────────────────────────────────────────────────────────────────────────────

def _collect_problems(data_dir: Path) -> dict[str, list[Path]]:
    """
    Recorre data_dir y devuelve un dict:
        { nombre_problema: [lista de archivos .py] }

    Soporta dos layouts:
      A) data_dir/problema/snippets/*.py   (layout Kaggle)
      B) data_dir/problema/*.py            (layout plano)
    """
    problems: dict[str, list[Path]] = {}

    for problem_dir in sorted(data_dir.iterdir()):
        if not problem_dir.is_dir():
            continue

        # Layout A: carpeta snippets/
        snippets_dir = problem_dir / "snippets"
        if snippets_dir.is_dir():
            files = sorted(snippets_dir.glob("*.py"))
        else:
            # Layout B: .py directamente en la carpeta del problema
            files = sorted(problem_dir.glob("*.py"))

        if files:
            problems[problem_dir.name] = files

    return problems


# ──────────────────────────────────────────────────────────────────────────────
# Preprocesamiento de un archivo
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess(path: Path, masking: str, k: int, w: int) -> dict | None:
    """
    Tokeniza un .py y genera sus fingerprints.
    Retorna None si el archivo tiene errores de sintaxis o está vacío.
    """
    result = tokenize_file(path, masking=masking)
    if result["error"] or not result["tokens"]:
        return None

    fps = winnow(result["tokens"], k=k, w=w)
    return {
        "tokens":    result["tokens"],
        "depth":     result["max_depth"],
        "fps":       fps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Construcción de pares
# ──────────────────────────────────────────────────────────────────────────────

def _build_positive_pairs(
    files: list[Path],
    cache: dict[Path, dict],
    masking: str,
    k: int,
    w: int,
) -> list[dict]:
    """
    Genera todos los pares C(n,2) dentro de un mismo problema (label=1).
    """
    pairs = []
    for path_a, path_b in itertools.combinations(files, 2):
        # Preprocesar y cachear
        if path_a not in cache:
            cache[path_a] = _preprocess(path_a, masking, k, w)
        if path_b not in cache:
            cache[path_b] = _preprocess(path_b, masking, k, w)

        data_a = cache[path_a]
        data_b = cache[path_b]

        if data_a is None or data_b is None:
            continue

        feats = compute_features(
            data_a["tokens"], data_a["depth"], data_a["fps"],
            data_b["tokens"], data_b["depth"], data_b["fps"],
            k=k,
        )
        feats["label"] = 1
        feats["file_a"] = str(path_a)
        feats["file_b"] = str(path_b)
        pairs.append(feats)

    return pairs


def _build_negative_pairs(
    problems: dict[str, list[Path]],
    n_negatives: int,
    cache: dict[Path, dict],
    masking: str,
    k: int,
    w: int,
    seed: int,
) -> list[dict]:
    """
    Muestrea n_negatives pares entre problemas DISTINTOS (label=0).
    """
    rng = random.Random(seed)
    problem_names = list(problems.keys())
    pairs = []
    attempts = 0
    max_attempts = n_negatives * 10

    while len(pairs) < n_negatives and attempts < max_attempts:
        attempts += 1

        # Elegir dos problemas distintos al azar
        prob_a, prob_b = rng.sample(problem_names, 2)
        path_a = rng.choice(problems[prob_a])
        path_b = rng.choice(problems[prob_b])

        # Preprocesar y cachear
        if path_a not in cache:
            cache[path_a] = _preprocess(path_a, masking, k, w)
        if path_b not in cache:
            cache[path_b] = _preprocess(path_b, masking, k, w)

        data_a = cache[path_a]
        data_b = cache[path_b]

        if data_a is None or data_b is None:
            continue

        feats = compute_features(
            data_a["tokens"], data_a["depth"], data_a["fps"],
            data_b["tokens"], data_b["depth"], data_b["fps"],
            k=k,
        )
        feats["label"] = 0
        feats["file_a"] = str(path_a)
        feats["file_b"] = str(path_b)
        pairs.append(feats)

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Función principal pública
# ──────────────────────────────────────────────────────────────────────────────

def build_pairs_csv(
    data_dir: str | Path,
    output: str | Path = "pairs_features.csv",
    k: int = 23,
    w: int = 4,
    masking: str = "medium",
    seed: int = 42,
    verbose: bool = True,
) -> Path:
    """
    Genera pairs_features.csv a partir del dataset de Kaggle.

    Parámetros
    ----------
    data_dir : directorio raíz del dataset
    output   : ruta del CSV de salida
    k        : tamaño del k-grama para Winnowing
    w        : tamaño de ventana para Winnowing
    masking  : nivel de enmascaramiento ("low" / "medium" / "high")
    seed     : semilla para el muestreo de pares negativos
    verbose  : imprimir progreso en consola

    Retorna
    -------
    Path del CSV generado
    """
    data_dir = Path(data_dir)
    output   = Path(output)

    if not data_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio: {data_dir}")

    # ── 1. Recolectar problemas ───────────────────────────────────────────────
    problems = _collect_problems(data_dir)
    if not problems:
        raise ValueError(f"No se encontraron archivos .py en: {data_dir}")

    if verbose:
        total_files = sum(len(v) for v in problems.values())
        print(f"Problemas encontrados : {len(problems)}")
        print(f"Archivos .py totales  : {total_files}")

    # ── 2. Pares positivos (todos los C(n,2) por problema) ───────────────────
    cache: dict[Path, dict] = {}
    positive_pairs: list[dict] = []

    for i, (prob_name, files) in enumerate(problems.items(), 1):
        if verbose:
            print(f"  [{i:3d}/{len(problems)}] {prob_name} — {len(files)} archivos", end="\r")

        pos = _build_positive_pairs(files, cache, masking, k, w)
        positive_pairs.extend(pos)

    if verbose:
        print(f"\nPares positivos (label=1): {len(positive_pairs)}")

    # ── 3. Pares negativos (balanceados con los positivos) ───────────────────
    n_neg = len(positive_pairs)
    negative_pairs = _build_negative_pairs(problems, n_neg, cache, masking, k, w, seed)

    if verbose:
        print(f"Pares negativos (label=0): {len(negative_pairs)}")

    # ── 4. Mezclar y guardar CSV ──────────────────────────────────────────────
    all_pairs = positive_pairs + negative_pairs
    rng = random.Random(seed)
    rng.shuffle(all_pairs)

    columns = FEATURE_COLUMNS + ["label", "file_a", "file_b"]
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_pairs)

    if verbose:
        print(f"\nCSV guardado en : {output}")
        print(f"Total de filas  : {len(all_pairs)}")

    return output


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera pairs_features.csv para el detector de plagio."
    )
    p.add_argument("--data-dir",  required=True,         help="Directorio raíz del dataset")
    p.add_argument("--output",    default="pairs_features.csv", help="Ruta del CSV de salida")
    p.add_argument("--k",         type=int, default=23,  help="Tamaño del k-grama (default: 23)")
    p.add_argument("--w",         type=int, default=4,   help="Ventana Winnowing  (default: 4)")
    p.add_argument("--masking",   default="medium",
                   choices=["low", "medium", "high"],    help="Nivel de enmascaramiento")
    p.add_argument("--seed",      type=int, default=42,  help="Semilla aleatoria")
    p.add_argument("--quiet",     action="store_true",   help="Suprimir salida de progreso")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_pairs_csv(
        data_dir = args.data_dir,
        output   = args.output,
        k        = args.k,
        w        = args.w,
        masking  = args.masking,
        seed     = args.seed,
        verbose  = not args.quiet,
    )