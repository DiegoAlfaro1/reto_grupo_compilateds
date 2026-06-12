"""
pipeline/snippet_features.py
────────────────────────────
Calcula el vector de características para UN snippet de código.
Es la adaptación de `pipeline/features.py` (que opera sobre *pares*) al
nuevo dataset en parquet, donde la tarea es clasificar cada snippet
individual: 0 = escrito por humano, 1 = generado por máquina.

Se reutiliza la misma maquinaria de la iteración 2 (tokenización
normalizada + k-gramas + Winnowing), pero las métricas son intrínsecas
al snippet en lugar de comparativas entre dos archivos:

Estructurales (tokens normalizados + Winnowing)
  1. token_count_log         log(1 + nº de tokens normalizados)
  2. token_diversity         tokens únicos / tokens totales
  3. kgram_uniqueness        k-gramas únicos / k-gramas totales (k = K)
  4. small_kgram_uniqueness  ídem con k = 4 (robusto en snippets cortos)
  5. fingerprint_density     |winnow(tokens)| / nº de k-gramas
  6. nesting_depth           profundidad del AST (Python) o anidamiento
  7. var_ratio               proporción de tokens VAR
  8. lit_ratio               proporción de tokens LIT
  9. parsed_python_ast       1 si el AST de Python parseó, 0 si fue léxico

Estilísticas (texto crudo — donde vive la señal humano vs. máquina)
 10. comment_ratio           líneas con comentario / líneas no vacías
 11. blank_line_ratio        líneas vacías / líneas totales
 12. avg_line_length         longitud media de las líneas no vacías
 13. line_length_cv          coef. de variación de la longitud de línea
 14. avg_identifier_length   longitud media de los identificadores crudos
 15. identifier_diversity    identificadores únicos / ocurrencias
 16. whitespace_ratio        caracteres de espacio / caracteres totales

Uso rápido
----------
    from pipeline.snippet_features import compute_snippet_features

    feats = compute_snippet_features(code, language="Python", k=15, w=4)
    # → dict con exactamente las claves de SNIPPET_FEATURE_COLUMNS
"""

from __future__ import annotations

import math

from pipeline.ast_tokenizer import MASK_VAR, MASK_LIT
from pipeline.snippet_tokenizer import tokenize_snippet
from pipeline.winnowing import winnow, kgrams


# ──────────────────────────────────────────────────────────────────────────────
# Columnas que espera el modelo (mismo rol que FEATURE_COLUMNS en features.py)
# ──────────────────────────────────────────────────────────────────────────────

SNIPPET_FEATURE_COLUMNS = [
    "token_count_log",
    "token_diversity",
    "kgram_uniqueness",
    "small_kgram_uniqueness",
    "fingerprint_density",
    "nesting_depth",
    "var_ratio",
    "lit_ratio",
    "parsed_python_ast",
    "comment_ratio",
    "blank_line_ratio",
    "avg_line_length",
    "line_length_cv",
    "avg_identifier_length",
    "identifier_diversity",
    "whitespace_ratio",
]

SMALL_K = 4

# Topes para que valores extremos no dominen el StandardScaler
_MAX_LINE_LEN = 300.0
_MAX_IDENT_LEN = 40.0
_MAX_NESTING = 30.0
_MAX_CV = 3.0


# ──────────────────────────────────────────────────────────────────────────────
# Métricas individuales
# ──────────────────────────────────────────────────────────────────────────────

def _kgram_uniqueness(tokens: list[str], k: int) -> float:
    """K-gramas únicos / k-gramas totales. 1 → nada se repite."""
    grams = list(kgrams(tokens, k))
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def _fingerprint_density(tokens: list[str], k: int, w: int) -> float:
    """Huellas Winnowing retenidas por k-grama: ~1/w si nada se repite,
    menor cuando el snippet es repetitivo (huellas duplicadas se funden)."""
    n_grams = max(0, len(tokens) - k + 1)
    if n_grams == 0:
        return 0.0
    return len(winnow(tokens, k=k, w=w)) / n_grams


def _line_stats(code: str) -> tuple[float, float, float, float]:
    """(comment_ratio*, blank_ratio, avg_len, cv_len) — *comment se calcula
    aparte con el tokenizador; aquí solo líneas vacías y longitudes."""
    lines = code.splitlines() or [""]
    non_blank = [ln for ln in lines if ln.strip()]
    blank_ratio = 1.0 - len(non_blank) / len(lines)
    if not non_blank:
        return blank_ratio, 0.0, 0.0, 0.0
    lengths = [len(ln) for ln in non_blank]
    mean = sum(lengths) / len(lengths)
    var = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    cv = math.sqrt(var) / mean if mean > 0 else 0.0
    return blank_ratio, min(mean, _MAX_LINE_LEN), min(cv, _MAX_CV), len(non_blank)


# ──────────────────────────────────────────────────────────────────────────────
# Función principal pública
# ──────────────────────────────────────────────────────────────────────────────

def compute_snippet_features(
    code: str,
    language: str = "",
    k: int = 15,
    w: int = 4,
    masking: str = "medium",
) -> dict[str, float]:
    """
    Calcula las 16 características para un snippet.

    Parámetros
    ----------
    code     : código fuente del snippet
    language : lenguaje declarado en el dataset ("Python", "C++", ...)
    k, w     : parámetros de Winnowing (mismos de la iteración 2)
    masking  : nivel de enmascaramiento del tokenizador AST

    Retorna
    -------
    dict con exactamente las claves de SNIPPET_FEATURE_COLUMNS.
    """
    code = code or ""
    tk = tokenize_snippet(code, language=language, masking=masking)
    tokens = tk["tokens"]
    n = len(tokens)

    blank_ratio, avg_len, cv_len, n_non_blank = _line_stats(code)
    comment_ratio = tk["n_comment_lines"] / n_non_blank if n_non_blank else 0.0

    identifiers = tk["identifiers"]
    if identifiers:
        avg_ident = sum(len(i) for i in identifiers) / len(identifiers)
        ident_diversity = len(set(identifiers)) / len(identifiers)
    else:
        avg_ident = 0.0
        ident_diversity = 0.0

    whitespace_ratio = (
        sum(1 for c in code if c in " \t") / len(code) if code else 0.0
    )

    return {
        "token_count_log": math.log1p(n),
        "token_diversity": len(set(tokens)) / n if n else 0.0,
        "kgram_uniqueness": _kgram_uniqueness(tokens, k),
        "small_kgram_uniqueness": _kgram_uniqueness(tokens, SMALL_K),
        "fingerprint_density": _fingerprint_density(tokens, k, w),
        "nesting_depth": min(float(tk["max_depth"]), _MAX_NESTING),
        "var_ratio": tokens.count(MASK_VAR) / n if n else 0.0,
        "lit_ratio": tokens.count(MASK_LIT) / n if n else 0.0,
        "parsed_python_ast": 1.0 if tk["method"] == "python_ast" else 0.0,
        "comment_ratio": min(comment_ratio, 1.0),
        "blank_line_ratio": blank_ratio,
        "avg_line_length": avg_len,
        "line_length_cv": cv_len,
        "avg_identifier_length": min(avg_ident, _MAX_IDENT_LEN),
        "identifier_diversity": ident_diversity,
        "whitespace_ratio": whitespace_ratio,
    }
