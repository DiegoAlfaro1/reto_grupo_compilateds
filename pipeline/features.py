"""
pipeline/features.py
────────────────────
Calcula el vector de 5 características para un par de archivos Python.
Este vector es la entrada al modelo de ML (model.ipynb).

Características calculadas
--------------------------
  1. winnowing_similarity      |A∩B| / min(|A|,|B|)          estilo Dolos
  2. fingerprint_jaccard       |A∩B| / |A∪B|                 sobre fingerprints
  3. token_overlap_ratio       Jaccard sobre conjuntos de k-gramas de tokens
  4. shared_fragment_coverage  cobertura promedio de tokens cubiertos por
                               fragmentos compartidos entre A y B
  5. ast_depth_difference      diferencia normalizada de profundidad del AST

Todas las métricas están en [0, 1].

Uso rápido
----------
    from pipeline.features import compute_features

    feats = compute_features(
        tokens_a, depth_a, fingerprints_a,
        tokens_b, depth_b, fingerprints_b,
        k=23,
    )
    # feats es un dict con las 5 claves del modelo
"""

from __future__ import annotations

from pipeline.winnowing import (
    winnowing_similarity,
    fingerprint_jaccard,
    kgrams,
)


# ──────────────────────────────────────────────────────────────────────────────
# Columnas que espera model.ipynb  (FEATURE_COLUMNS del notebook)
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "winnowing_similarity",
    "shared_fragment_coverage",
    "token_overlap_ratio",
    "ast_depth_difference",
    "fingerprint_jaccard",
]


# ──────────────────────────────────────────────────────────────────────────────
# Métricas individuales
# ──────────────────────────────────────────────────────────────────────────────

def token_overlap_ratio(tokens_a: list[str], tokens_b: list[str], k: int) -> float:
    """
    Jaccard sobre los *conjuntos* de k-gramas de tokens (sin contar frecuencia).

        |kgrams(A) ∩ kgrams(B)| / |kgrams(A) ∪ kgrams(B)|

    Mide cuántas sub-secuencias estructurales comparten los dos archivos,
    independientemente del orden global.
    """
    set_a = set(kgrams(tokens_a, k))
    set_b = set(kgrams(tokens_b, k))
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def shared_fragment_coverage(
    tokens_a: list[str],
    tokens_b: list[str],
    k: int,
) -> float:
    """
    Fracción promedio de tokens de cada archivo que están cubiertos por
    al menos un k-grama compartido.

    Algoritmo
    ---------
    1. Encontrar los k-gramas comunes entre A y B.
    2. Para cada archivo, marcar las posiciones de tokens que pertenecen
       a al menos un k-grama compartido.
    3. coverage_X = posiciones_marcadas_X / len(tokens_X)
    4. Retornar (coverage_A + coverage_B) / 2
    """
    common = set(kgrams(tokens_a, k)) & set(kgrams(tokens_b, k))
    if not common:
        return 0.0

    def _covered_positions(tokens: list[str]) -> int:
        n = len(tokens)
        covered = [False] * n
        for i in range(n - k + 1):
            gram = tuple(tokens[i : i + k])
            if gram in common:
                for j in range(i, i + k):
                    covered[j] = True
        return sum(covered)

    cov_a = _covered_positions(tokens_a) / len(tokens_a) if tokens_a else 0.0
    cov_b = _covered_positions(tokens_b) / len(tokens_b) if tokens_b else 0.0
    return (cov_a + cov_b) / 2.0


def ast_depth_difference(depth_a: int, depth_b: int) -> float:
    """
    Diferencia normalizada de profundidad máxima del AST.

        |depth_A - depth_B| / max(depth_A, depth_B)

    0 → misma profundidad   1 → profundidades muy distintas
    """
    max_d = max(depth_a, depth_b)
    if max_d == 0:
        return 0.0
    return abs(depth_a - depth_b) / max_d


# ──────────────────────────────────────────────────────────────────────────────
# Función principal pública
# ──────────────────────────────────────────────────────────────────────────────

def compute_features(
    tokens_a: list[str],
    depth_a: int,
    fingerprints_a: frozenset[int],
    tokens_b: list[str],
    depth_b: int,
    fingerprints_b: frozenset[int],
    k: int = 23,
) -> dict[str, float]:
    """
    Calcula las 5 características para un par de archivos.

    Parámetros
    ----------
    tokens_a / tokens_b         : salida de ast_tokenizer (campo "tokens")
    depth_a  / depth_b          : salida de ast_tokenizer (campo "max_depth")
    fingerprints_a / _b         : salida de winnow()
    k                           : tamaño de k-grama (debe coincidir con el usado
                                  al generar los fingerprints)

    Retorna
    -------
    dict con exactamente las claves de FEATURE_COLUMNS, valores en [0, 1].
    """
    return {
        "winnowing_similarity": winnowing_similarity(fingerprints_a, fingerprints_b),
        "shared_fragment_coverage": shared_fragment_coverage(tokens_a, tokens_b, k),
        "token_overlap_ratio": token_overlap_ratio(tokens_a, tokens_b, k),
        "ast_depth_difference": ast_depth_difference(depth_a, depth_b),
        "fingerprint_jaccard": fingerprint_jaccard(fingerprints_a, fingerprints_b),
    }