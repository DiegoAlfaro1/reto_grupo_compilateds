"""
pipeline/winnowing.py
─────────────────────
Implementa el algoritmo Winnowing para generar huellas digitales (fingerprints)
a partir de una secuencia de tokens normalizados.

Referencia
----------
Schleimer, S., Wilkerson, D. S., & Aiken, A. (2003).
Winnowing: Local algorithms for document fingerprinting.
ACM SIGMOD, 76–85. https://doi.org/10.1145/872757.872770

Pipeline interno
----------------
  tokens  →  k-gramas  →  hashes  →  ventana deslizante  →  fingerprints (set)

Garantía del algoritmo
-----------------------
  Cualquier subcadena de longitud ≥ k compartida entre dos documentos
  quedará representada por al menos un hash en común.
  Se retiene aprox. 1 de cada w hashes → compresión configurable.

Uso rápido
----------
    from pipeline.winnowing import winnow, kgrams, rolling_hashes

    tokens = ["Module", "FunctionDef", "VAR", "For", "VAR", "LIT", "Return", "VAR"]
    fps = winnow(tokens, k=5, w=4)
    # → frozenset de enteros (huellas digitales)
"""

from __future__ import annotations

import hashlib
from typing import Iterator


# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────

# Separador interno para unir tokens en un k-grama antes de hashear
_SEP = "\x00"

# Módulo para el hash rodante de Rabin–Karp (primo grande)
_MOD = (1 << 61) - 1   # primo de Mersenne; evita colisiones frecuentes
_BASE = 131


# ──────────────────────────────────────────────────────────────────────────────
# Paso 1 — K-gramas
# ──────────────────────────────────────────────────────────────────────────────

def kgrams(tokens: list[str], k: int) -> Iterator[tuple[str, ...]]:
    """
    Genera todas las sub-secuencias contiguas de longitud k.

    Parámetros
    ----------
    tokens : lista de tokens normalizados
    k      : tamaño del k-grama (≥ 1)

    Yields
    ------
    tupla de k strings
    """
    if k < 1:
        raise ValueError(f"k debe ser ≥ 1; recibido: {k}")
    n = len(tokens)
    for i in range(n - k + 1):
        yield tuple(tokens[i : i + k])


# ──────────────────────────────────────────────────────────────────────────────
# Paso 2 — Hashing
# ──────────────────────────────────────────────────────────────────────────────

def _hash_kgram(gram: tuple[str, ...]) -> int:
    """
    Convierte un k-grama en un entero hash usando SHA-256 truncado.
    Más estable que el hash incorporado de Python (que varía entre procesos).
    """
    text = _SEP.join(gram).encode()
    digest = hashlib.sha256(text).digest()
    # Tomamos los primeros 8 bytes → entero de 64 bits sin signo
    return int.from_bytes(digest[:8], byteorder="little")


def rolling_hashes(tokens: list[str], k: int) -> list[int]:
    """
    Calcula el hash de cada k-grama de la secuencia de tokens.

    Retorna
    -------
    list[int] — un hash por cada k-grama, en orden
    """
    return [_hash_kgram(g) for g in kgrams(tokens, k)]


# ──────────────────────────────────────────────────────────────────────────────
# Paso 3 — Ventana deslizante (Winnowing)
# ──────────────────────────────────────────────────────────────────────────────

def _sliding_window_min(hashes: list[int], w: int) -> frozenset[int]:
    """
    Aplica la ventana deslizante de tamaño w y retiene el hash mínimo
    de cada ventana (desempate: posición más a la derecha, como en el paper).

    Retorna el conjunto de hashes seleccionados (sin duplicados).
    """
    if not hashes:
        return frozenset()

    selected: set[int] = set()
    n = len(hashes)

    # Mínimo previo y su posición
    prev_min_val = -1
    prev_min_pos = -1

    for i in range(n - w + 1):
        window = hashes[i : i + w]

        # Buscar el mínimo de la ventana (posición más a la derecha en empate)
        min_val = window[-1]
        min_pos = i + w - 1
        for j in range(w - 2, -1, -1):
            if window[j] < min_val:
                min_val = window[j]
                min_pos = i + j

        # Solo agregar si cambió el mínimo seleccionado
        if min_pos != prev_min_pos:
            selected.add(min_val)
            prev_min_val = min_val
            prev_min_pos = min_pos

    return frozenset(selected)


# ──────────────────────────────────────────────────────────────────────────────
# Función principal pública
# ──────────────────────────────────────────────────────────────────────────────

def winnow(
    tokens: list[str],
    k: int = 23,
    w: int = 4,
) -> frozenset[int]:
    """
    Genera las huellas digitales (fingerprints) de una secuencia de tokens
    mediante el algoritmo Winnowing.

    Parámetros
    ----------
    tokens : list[str]
        Secuencia de tokens normalizados (salida de ast_tokenizer).
    k : int
        Tamaño del k-grama. Default 23 (valor por defecto de Dolos).
    w : int
        Tamaño de la ventana deslizante. Default 4.

    Retorna
    -------
    frozenset[int]
        Conjunto de hashes que representan la huella digital del documento.
        Vacío si el documento tiene menos de k tokens.
    """
    if k < 1:
        raise ValueError(f"k debe ser ≥ 1; recibido: {k}")
    if w < 1:
        raise ValueError(f"w debe ser ≥ 1; recibido: {w}")

    hashes = rolling_hashes(tokens, k)

    if not hashes:
        return frozenset()

    # Si hay menos hashes que el tamaño de ventana, usamos todos
    if len(hashes) < w:
        return frozenset(hashes)

    return _sliding_window_min(hashes, w)


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de similitud (usadas por features.py)
# ──────────────────────────────────────────────────────────────────────────────

def winnowing_similarity(fps_a: frozenset[int], fps_b: frozenset[int]) -> float:
    """
    Similitud estilo Dolos:
        |A ∩ B| / min(|A|, |B|)

    Sesgada hacia detectar si uno es subconjunto del otro
    (útil cuando un archivo es mucho más corto).
    """
    if not fps_a or not fps_b:
        return 0.0
    return len(fps_a & fps_b) / min(len(fps_a), len(fps_b))


def fingerprint_jaccard(fps_a: frozenset[int], fps_b: frozenset[int]) -> float:
    """
    Similitud de Jaccard sobre los conjuntos de huellas:
        |A ∩ B| / |A ∪ B|

    Simétrica; penaliza más cuando los archivos tienen tamaños distintos.
    """
    union = fps_a | fps_b
    if not union:
        return 0.0
    return len(fps_a & fps_b) / len(union)