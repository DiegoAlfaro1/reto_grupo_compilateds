"""Algoritmo Winnowing para generacion de huellas digitales (fingerprints).

Referencia: Schleimer, Wilkerson & Aiken (2003), "Winnowing: Local algorithms
for document fingerprinting".

Dado un flujo de tokens:

1. Se calcula el hash de cada k-grama (subsecuencia de ``k`` tokens consecutivos).
2. Se desliza una ventana de tamano ``w`` sobre la secuencia de hashes.
3. De cada ventana se retiene el hash minimo (en empates, el mas a la derecha;
   asi ventanas solapadas que comparten ese minimo no lo registran dos veces).
4. El conjunto de hashes retenidos es la huella digital del documento.

Propiedad clave: cualquier subcadena compartida de longitud >= k entre dos
documentos queda garantizada en la huella, manteniendo ~1 de cada w hashes.
"""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple

# Parametros por defecto tomados del marco de referencia (default de Dolos).
DEFAULT_K = 23
DEFAULT_W = 4

# Hash rodante polinomial determinista (independiente de PYTHONHASHSEED, a
# diferencia de hash() para str, que esta "salteado" entre ejecuciones).
_BASE = 257
_MOD = (1 << 61) - 1  # primo de Mersenne, rapido y con baja tasa de colisiones


def _token_ids(tokens: Sequence[str]) -> List[int]:
    """Mapea cada token a un entero estable mediante un hash determinista."""
    ids = []
    for tok in tokens:
        h = 0
        for ch in tok:
            h = (h * _BASE + ord(ch)) % _MOD
        ids.append(h)
    return ids


def kgram_hashes(tokens: Sequence[str], k: int = DEFAULT_K) -> List[int]:
    """Hash rodante de cada k-grama de la secuencia de tokens.

    Si hay menos de ``k`` tokens se usa un unico k-grama con todo el contenido
    disponible, de modo que los archivos cortos no queden sin huella.
    """
    ids = _token_ids(tokens)
    n = len(ids)
    if n == 0:
        return []
    if n < k:
        # Un solo hash con toda la secuencia disponible.
        h = 0
        for x in ids:
            h = (h * _BASE + x) % _MOD
        return [h]

    high = pow(_BASE, k - 1, _MOD)
    hashes: List[int] = []

    # Hash del primer k-grama.
    h = 0
    for i in range(k):
        h = (h * _BASE + ids[i]) % _MOD
    hashes.append(h)

    # Hashes rodantes: quitar el token saliente, agregar el entrante.
    for i in range(k, n):
        h = (h - ids[i - k] * high) % _MOD
        h = (h * _BASE + ids[i]) % _MOD
        hashes.append(h)

    return hashes


def winnow(hashes: Sequence[int], w: int = DEFAULT_W) -> Set[Tuple[int, int]]:
    """Selecciona las huellas con ventana deslizante de tamano ``w``.

    Devuelve un conjunto de tuplas ``(hash, posicion)``. La posicion permite
    estimar la cobertura de fragmentos compartidos; si solo interesan los
    valores, basta con ``{h for h, _ in fingerprints}``.
    """
    n = len(hashes)
    if n == 0:
        return set()
    if n <= w:
        # Ventana mayor que la secuencia: una sola ventana global.
        min_pos = min(range(n), key=lambda i: (hashes[i], -i))
        return {(hashes[min_pos], min_pos)}

    fingerprints: Set[Tuple[int, int]] = set()
    last_selected = -1
    for start in range(0, n - w + 1):
        window = range(start, start + w)
        # Minimo de la ventana; en empate se elige el mas a la derecha.
        min_pos = min(window, key=lambda i: (hashes[i], -i))
        if min_pos != last_selected:
            fingerprints.add((hashes[min_pos], min_pos))
            last_selected = min_pos
    return fingerprints


def fingerprint_source(
    tokens: Sequence[str],
    k: int = DEFAULT_K,
    w: int = DEFAULT_W,
) -> Set[Tuple[int, int]]:
    """Atajo: de tokens directamente a huellas ``(hash, posicion)``."""
    return winnow(kgram_hashes(tokens, k), w)
