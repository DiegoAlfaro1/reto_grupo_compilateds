"""Pipeline de preprocesamiento para deteccion de plagio en codigo Python.

Este paquete implementa la Etapa 2 y 3 del marco de referencia:

1. ``ast_tokenizer``  -> Parser de AST (modulo ``ast`` de Python) que convierte
   archivos ``.py`` en secuencias de tokens normalizados, enmascarando
   identificadores y (opcionalmente) literales.
2. ``winnowing``      -> Algoritmo Winnowing: k-gramas, hash rodante y seleccion
   de huellas digitales (fingerprints).
3. ``features``       -> Calculo del vector de caracteristicas por par de archivos
   que consume el modelo de ``model.ipynb``.
4. ``build_pairs``    -> Construye los pares etiquetados a partir del dataset de
   Kaggle y escribe ``pairs_features.csv``.

El CSV de salida tiene exactamente las columnas que espera el notebook:

    winnowing_similarity, shared_fragment_coverage, token_overlap_ratio,
    ast_depth_difference, fingerprint_jaccard, label
"""

from .ast_tokenizer import MaskingLevel, tokenize_source, tokenize_file, ast_depth
from .winnowing import kgram_hashes, winnow, fingerprint_source
from .features import FEATURE_COLUMNS, LABEL_COLUMN, pair_features

__all__ = [
    "MaskingLevel",
    "tokenize_source",
    "tokenize_file",
    "ast_depth",
    "kgram_hashes",
    "winnow",
    "fingerprint_source",
    "FEATURE_COLUMNS",
    "LABEL_COLUMN",
    "pair_features",
]
