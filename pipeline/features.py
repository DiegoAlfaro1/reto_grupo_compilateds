"""Extraccion del vector de caracteristicas por par de archivos.

Produce exactamente las columnas que consume ``model.ipynb``:

    winnowing_similarity, shared_fragment_coverage, token_overlap_ratio,
    ast_depth_difference, fingerprint_jaccard

Todas las caracteristicas estan normalizadas al rango [0, 1]. Las cuatro
primeras y la ultima son "similitudes" (mas alto = mas parecido); la cuarta,
``ast_depth_difference``, es una diferencia normalizada (mas alto = mas
distinto). El clasificador aprende el signo de cada relacion durante el
entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from .ast_tokenizer import MaskingLevel, ast_depth, tokenize_source
from .winnowing import DEFAULT_K, DEFAULT_W, fingerprint_source, kgram_hashes

# Orden EXACTO esperado por el notebook (FEATURE_COLUMNS en model.ipynb).
FEATURE_COLUMNS: List[str] = [
    "winnowing_similarity",
    "shared_fragment_coverage",
    "token_overlap_ratio",
    "ast_depth_difference",
    "fingerprint_jaccard",
]
LABEL_COLUMN = "label"


@dataclass
class CodeProfile:
    """Representacion preprocesada de un archivo, reutilizable en muchos pares.

    Calcular esto una sola vez por archivo evita re-tokenizar O(n^2) veces al
    comparar todos los pares de un problema.
    """

    tokens: List[str]
    kgrams: Set[int]                       # conjunto de hashes de k-gramas
    fingerprints: Set[Tuple[int, int]]     # huellas Winnowing (hash, posicion)
    fp_hashes: Set[int]                    # solo los valores de hash
    depth: int

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


def build_profile(
    source: str,
    k: int = DEFAULT_K,
    w: int = DEFAULT_W,
    level: MaskingLevel = MaskingLevel.MEDIUM,
) -> CodeProfile:
    """Preprocesa una cadena de codigo en un ``CodeProfile``."""
    tokens = tokenize_source(source, level)
    kgrams = set(kgram_hashes(tokens, k))
    fingerprints = fingerprint_source(tokens, k, w)
    fp_hashes = {h for h, _ in fingerprints}
    depth = ast_depth(source)
    return CodeProfile(
        tokens=tokens,
        kgrams=kgrams,
        fingerprints=fingerprints,
        fp_hashes=fp_hashes,
        depth=depth,
    )


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def pair_features(a: CodeProfile, b: CodeProfile) -> Dict[str, float]:
    """Calcula el vector de caracteristicas entre dos perfiles de codigo."""
    shared_fp = a.fp_hashes & b.fp_hashes
    union_fp = a.fp_hashes | b.fp_hashes

    # 1. Similitud Winnowing (coeficiente de solapamiento, estilo Dolos):
    #    fraccion de huellas compartidas respecto al archivo con menos huellas.
    winnowing_similarity = _safe_div(
        len(shared_fp), min(len(a.fp_hashes), len(b.fp_hashes))
    )

    # 2. Cobertura de fragmentos compartidos: que proporcion de las huellas de
    #    cada archivo participa en el solapamiento, promediada entre ambos.
    cov_a = _safe_div(len(a.fp_hashes & shared_fp), len(a.fp_hashes))
    cov_b = _safe_div(len(b.fp_hashes & shared_fp), len(b.fp_hashes))
    shared_fragment_coverage = (cov_a + cov_b) / 2.0

    # 3. Solapamiento de tokens (a nivel de k-gramas, antes de winnowing):
    #    Jaccard de los conjuntos de k-gramas.
    token_overlap_ratio = _safe_div(
        len(a.kgrams & b.kgrams), len(a.kgrams | b.kgrams)
    )

    # 4. Diferencia de profundidad del AST, normalizada a [0, 1].
    ast_depth_difference = _safe_div(
        abs(a.depth - b.depth), max(a.depth, b.depth)
    )

    # 5. Jaccard de las huellas digitales.
    fingerprint_jaccard = _safe_div(len(shared_fp), len(union_fp))

    return {
        "winnowing_similarity": winnowing_similarity,
        "shared_fragment_coverage": shared_fragment_coverage,
        "token_overlap_ratio": token_overlap_ratio,
        "ast_depth_difference": ast_depth_difference,
        "fingerprint_jaccard": fingerprint_jaccard,
    }


def features_from_sources(
    source_a: str,
    source_b: str,
    k: int = DEFAULT_K,
    w: int = DEFAULT_W,
    level: MaskingLevel = MaskingLevel.MEDIUM,
) -> Dict[str, float]:
    """Conveniencia: par de cadenas de codigo -> vector de caracteristicas."""
    return pair_features(
        build_profile(source_a, k, w, level),
        build_profile(source_b, k, w, level),
    )
