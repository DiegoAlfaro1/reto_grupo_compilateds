"""Construye el dataset de pares etiquetados y escribe ``pairs_features.csv``.

Implementa la "Construccion de pares etiquetados" descrita en el marco de
referencia (seccion 8.1) sobre el dataset *Code Similarity Dataset - Python
Variants* de Kaggle:

* Par POSITIVO (label 1): dos variantes del MISMO problema.
* Par NEGATIVO (label 0): dos soluciones de problemas DISTINTOS.

Las clases se balancean (igual numero de pares positivos y negativos).

Uso tipico:

    python -m pipeline.build_pairs --data-dir ./data/python_variants \
        --output pairs_features.csv

Estructura de carpetas esperada por defecto: cada problema es una subcarpeta
que contiene sus archivos ``.py`` (una por variante)::

    data/python_variants/
        problem_001/
            variant_01.py
            variant_02.py
            ...
        problem_002/
            ...

Si tu copia del dataset usa otra organizacion, ajusta ``--group-by`` (carpeta
padre, abuela, o un patron de nombre de archivo).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from .ast_tokenizer import MaskingLevel
from .features import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    CodeProfile,
    build_profile,
    pair_features,
)
from .winnowing import DEFAULT_K, DEFAULT_W


def find_python_files(data_dir: str) -> List[str]:
    """Lista recursivamente todos los archivos ``.py`` bajo ``data_dir``."""
    paths: List[str] = []
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith(".py"):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def problem_id_for(path: str, data_dir: str, group_by: str) -> str:
    """Determina a que problema pertenece un archivo.

    * ``parent``      -> nombre de la carpeta que contiene el archivo (default).
    * ``grandparent`` -> nombre de la carpeta abuela.
    * ``filename``    -> prefijo del nombre del archivo antes del primer '_' o
                         digito (p. ej. ``problem12_v3.py`` -> ``problem``).
    """
    if group_by == "parent":
        return os.path.basename(os.path.dirname(path))
    if group_by == "grandparent":
        return os.path.basename(os.path.dirname(os.path.dirname(path)))
    if group_by == "filename":
        base = os.path.splitext(os.path.basename(path))[0]
        m = re.match(r"([A-Za-z]+)", base)
        return m.group(1) if m else base
    raise ValueError(f"group_by desconocido: {group_by}")


def group_files_by_problem(
    paths: List[str], data_dir: str, group_by: str
) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        groups[problem_id_for(p, data_dir, group_by)].append(p)
    # Solo conservamos problemas con >= 2 variantes (necesarias para un par +).
    return {pid: files for pid, files in groups.items() if len(files) >= 2}


def _load_profiles(
    paths: List[str], k: int, w: int, level: MaskingLevel
) -> Dict[str, CodeProfile]:
    """Tokeniza y genera huellas de cada archivo una sola vez (cacheado)."""
    profiles: Dict[str, CodeProfile] = {}
    skipped = 0
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                source = fh.read()
            profiles[p] = build_profile(source, k, w, level)
        except SyntaxError:
            skipped += 1  # archivo no parseable; lo ignoramos.
    if skipped:
        print(f"  Aviso: {skipped} archivo(s) omitido(s) por errores de sintaxis.")
    return profiles


def generate_pairs(
    groups: Dict[str, List[str]],
    profiles: Dict[str, CodeProfile],
    max_pos_per_problem: Optional[int],
    rng: random.Random,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Genera pares positivos (intra-problema) y negativos (inter-problema).

    Los negativos se muestrean aleatoriamente y se balancean al numero de
    positivos.
    """
    # Positivos: combinaciones dentro de cada problema.
    positives: List[Tuple[str, str]] = []
    for files in groups.values():
        files = [f for f in files if f in profiles]
        pairs = list(combinations(files, 2))
        if max_pos_per_problem is not None and len(pairs) > max_pos_per_problem:
            pairs = rng.sample(pairs, max_pos_per_problem)
        positives.extend(pairs)

    # Negativos: pares entre problemas distintos, hasta igualar a los positivos.
    problem_ids = list(groups.keys())
    negatives: List[Tuple[str, str]] = []
    seen: set = set()
    target = len(positives)
    attempts = 0
    max_attempts = target * 50 + 1000
    while len(negatives) < target and attempts < max_attempts:
        attempts += 1
        if len(problem_ids) < 2:
            break
        pid_a, pid_b = rng.sample(problem_ids, 2)
        fa = [f for f in groups[pid_a] if f in profiles]
        fb = [f for f in groups[pid_b] if f in profiles]
        if not fa or not fb:
            continue
        a = rng.choice(fa)
        b = rng.choice(fb)
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        negatives.append((a, b))

    return positives, negatives


def write_csv(
    output: str,
    positives: List[Tuple[str, str]],
    negatives: List[Tuple[str, str]],
    profiles: Dict[str, CodeProfile],
) -> int:
    """Calcula caracteristicas de cada par y escribe el CSV. Devuelve # filas."""
    header = FEATURE_COLUMNS + [LABEL_COLUMN]
    rows = 0
    with open(output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for label, pairs in ((1, positives), (0, negatives)):
            for a, b in pairs:
                feats = pair_features(profiles[a], profiles[b])
                feats[LABEL_COLUMN] = label
                writer.writerow(feats)
                rows += 1
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Genera pairs_features.csv para el modelo de plagio."
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Carpeta raiz del dataset (con los .py organizados por problema).",
    )
    parser.add_argument(
        "--output", default="pairs_features.csv",
        help="Ruta del CSV de salida (default: pairs_features.csv).",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Tamano de k-grama.")
    parser.add_argument("--w", type=int, default=DEFAULT_W, help="Ventana Winnowing.")
    parser.add_argument(
        "--masking", choices=[m.value for m in MaskingLevel],
        default=MaskingLevel.MEDIUM.value, help="Nivel de enmascaramiento.",
    )
    parser.add_argument(
        "--group-by", choices=["parent", "grandparent", "filename"],
        default="parent", help="Como agrupar archivos por problema.",
    )
    parser.add_argument(
        "--max-pos-per-problem", type=int, default=None,
        help="Limite de pares positivos por problema (muestreo). Default: todos.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    level = MaskingLevel(args.masking)

    print(f"Buscando archivos .py en {args.data_dir} ...")
    paths = find_python_files(args.data_dir)
    print(f"  {len(paths)} archivo(s) encontrado(s).")

    groups = group_files_by_problem(paths, args.data_dir, args.group_by)
    print(f"  {len(groups)} problema(s) con >= 2 variantes.")
    if not groups:
        raise SystemExit(
            "No se encontraron problemas con multiples variantes. "
            "Revisa --data-dir y --group-by."
        )

    print(f"Tokenizando y generando huellas (k={args.k}, w={args.w}, "
          f"masking={args.masking}) ...")
    profiles = _load_profiles(paths, args.k, args.w, level)

    positives, negatives = generate_pairs(
        groups, profiles, args.max_pos_per_problem, rng
    )
    print(f"  {len(positives)} pares positivos, {len(negatives)} pares negativos.")

    rows = write_csv(args.output, positives, negatives, profiles)
    print(f"Listo: {rows} filas escritas en {args.output}")


if __name__ == "__main__":
    main()
