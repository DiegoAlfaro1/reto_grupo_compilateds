"""
pipeline/generate_dataset.py
────────────────────────────
Construye el dataset de plagio "ad hoc" (sección 5.3 del marco de
referencia) a partir del Code Similarity Dataset de Kaggle.

Motivación (hallazgo de la iteración 1)
---------------------------------------
Las 20 variantes por problema del dataset de Kaggle NO son ofuscaciones
unas de otras: son ALGORITMOS DISTINTOS (recursivo vs. iterativo vs.
fórmula de Binet, etc.). Detectar esa equivalencia semántica está
explícitamente FUERA del alcance del proyecto (sección 9 del marco).
Por eso la iteración 1 obtuvo F1 ≈ 0.20: se le pedía al modelo una
tarea imposible para AST + Winnowing.

Solución (iteración 2)
----------------------
Cada snippet original se trata como una FUENTE, y el plagio se simula
aplicando las transformaciones de ofuscación documentadas en
pipeline/transforms.py (las mismas que lista el marco en la sección 8.1:
renombrado, reordenamiento, código muerto, reescrituras equivalentes).

Layout de salida
----------------
    data/
        fibonacci__snip_01/
            original.py          ← snippet fuente
            variant_1.py         ← plagio con intensity baja
            variant_2.py
            variant_3.py
            variant_4.py         ← plagio con intensity alta
        fibonacci__snip_02/
            ...

Cada carpeta es un "grupo de plagio": todos sus archivos derivan de la
misma fuente. El prefijo antes de `__` conserva el problema original,
lo que permite muestrear negativos difíciles (mismo problema, distinto
algoritmo) en build_pairs.py.

Uso
---
    python -m pipeline.generate_dataset --output-dir data --variants 4
"""

from __future__ import annotations

import argparse
import csv
import zlib
from pathlib import Path

from pipeline.transforms import make_variant

DEFAULT_KAGGLE_DIR = (
    Path.home()
    / ".cache" / "kagglehub" / "datasets" / "hemajitpatel"
    / "code-similarity-dataset-python-variants" / "versions" / "1"
    / "CodeSimilarityDataset"
)


def _download_if_needed() -> Path:
    """Descarga el dataset de Kaggle con kagglehub si no está en caché."""
    if DEFAULT_KAGGLE_DIR.exists():
        return DEFAULT_KAGGLE_DIR
    import kagglehub
    path = kagglehub.dataset_download("hemajitpatel/code-similarity-dataset-python-variants")
    return Path(path) / "CodeSimilarityDataset"


def generate_dataset(
    source_dir: str | Path | None = None,
    output_dir: str | Path = "data",
    n_variants: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> Path:
    """
    Genera el dataset ad hoc de grupos de plagio.

    Parámetros
    ----------
    source_dir : raíz del Code Similarity Dataset (None → descargar/caché)
    output_dir : carpeta de salida (layout descrito arriba)
    n_variants : variantes plagiadas por snippet original
    seed       : semilla base para reproducibilidad

    Retorna
    -------
    Path de la carpeta generada.
    """
    source_dir = Path(source_dir) if source_dir else _download_if_needed()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    n_groups = 0
    n_files = 0

    for problem_dir in sorted(source_dir.iterdir()):
        snippets = problem_dir / "snippets"
        if not snippets.is_dir():
            continue

        for snippet in sorted(snippets.glob("*.py")):
            source = snippet.read_text(encoding="utf-8")
            group_name = f"{problem_dir.name}__{snippet.stem}"
            group_dir = output_dir / group_name

            # Generar las variantes ANTES de escribir nada: si el original
            # no parsea, se omite el grupo completo.
            variants = []
            for v in range(1, n_variants + 1):
                # intensity crece con v: variant_1 = ofuscación ligera
                # (solo renombrado, casi seguro), variant_4 = agresiva.
                intensity = v / n_variants
                # crc32 (no hash()): estable entre procesos → reproducible
                var_seed = seed * 100_000 + zlib.crc32(group_name.encode()) % 50_000 + v
                code = make_variant(source, seed=var_seed, intensity=intensity)
                if code is None:
                    break
                variants.append((f"variant_{v}.py", code, round(intensity, 2)))

            if len(variants) < n_variants:
                if verbose:
                    print(f"  [skip] {group_name}: el original no parsea")
                continue

            group_dir.mkdir(exist_ok=True)
            (group_dir / "original.py").write_text(source, encoding="utf-8")
            manifest_rows.append([group_name, "original.py", problem_dir.name, ""])
            for filename, code, intensity in variants:
                (group_dir / filename).write_text(code, encoding="utf-8")
                manifest_rows.append([group_name, filename, problem_dir.name, intensity])

            n_groups += 1
            n_files += 1 + n_variants

    # Manifest con la procedencia de cada archivo (transformaciones documentadas)
    with open(output_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "file", "problem", "intensity"])
        writer.writerows(manifest_rows)

    if verbose:
        print(f"Grupos de plagio generados : {n_groups}")
        print(f"Archivos .py totales       : {n_files}")
        print(f"Dataset en                 : {output_dir}")

    return output_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera el dataset ad hoc de plagio.")
    p.add_argument("--source-dir", default=None,
                   help="Raíz del CodeSimilarityDataset (default: descargar con kagglehub)")
    p.add_argument("--output-dir", default="data", help="Carpeta de salida")
    p.add_argument("--variants", type=int, default=4, help="Variantes por snippet")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        n_variants=args.variants,
        seed=args.seed,
    )
