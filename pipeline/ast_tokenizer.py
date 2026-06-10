"""Tokenizador basado en el Arbol de Sintaxis Abstracta (AST) de Python.

Convierte el codigo fuente de un archivo ``.py`` en una secuencia de tokens
normalizados. La normalizacion (enmascaramiento) hace que el analisis sea
insensible a las transformaciones de plagio mas comunes:

* Renombrado de variables / funciones  -> los identificadores se enmascaran.
* Cambios de formato, comentarios, lineas vacias -> el AST los ignora por
  construccion (un comentario no produce ningun nodo).
* Reordenamiento de bloques -> se mitiga mas adelante por Winnowing, ya que las
  subcadenas compartidas se detectan sin importar su posicion.

El nivel de enmascaramiento es la variable independiente "Nivel de
enmascaramiento" (Bajo / Medio / Alto) del marco de referencia.
"""

from __future__ import annotations

import ast
from enum import Enum
from typing import List


class MaskingLevel(str, Enum):
    """Cuanta informacion del codigo se conserva en los tokens.

    * ``LOW``    -> conserva nombres de identificadores y valores de literales.
                    (Mas discriminativo, mas fragil ante renombrado.)
    * ``MEDIUM`` -> enmascara identificadores; conserva tipos de literales.
                    (Recomendado: insensible a renombrado de variables.)
    * ``HIGH``   -> enmascara identificadores y literales; solo queda la
                    estructura sintactica pura. (Mas robusto ante ofuscacion.)
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Tokens genericos usados al enmascarar.
_MASK_NAME = "<NAME>"
_MASK_ATTR = "<ATTR>"
_MASK_ARG = "<ARG>"


class _Tokenizer(ast.NodeVisitor):
    """Recorre el AST en pre-orden y emite un token por nodo.

    El recorrido en pre-orden preserva el orden estructural del codigo, que es
    justo lo que necesita el motor de Winnowing para detectar subcadenas
    compartidas.
    """

    def __init__(self, level: MaskingLevel) -> None:
        self.level = level
        self.tokens: List[str] = []

    # -- helpers ---------------------------------------------------------

    def _emit(self, token: str) -> None:
        self.tokens.append(token)

    def _ident(self, name: str, mask_token: str) -> str:
        """Devuelve el identificador o su mascara segun el nivel."""
        if self.level == MaskingLevel.LOW:
            return name
        return mask_token

    # -- nodos con tratamiento especial ----------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        # Variable / referencia a nombre.
        self._emit(self._ident(node.id, _MASK_NAME))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # obj.attr  -> el nombre del atributo tambien es un identificador.
        self._emit("Attribute:" + self._ident(node.attr, _MASK_ATTR))
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._emit("arg:" + self._ident(node.arg, _MASK_ARG))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._emit("FunctionDef:" + self._ident(node.name, _MASK_NAME))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # mismo tratamiento

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._emit("ClassDef:" + self._ident(node.name, _MASK_NAME))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # Literales: numeros, cadenas, booleanos, None, etc.
        if self.level == MaskingLevel.HIGH:
            # Solo conservamos el *tipo* del literal, no su valor.
            self._emit("Constant:" + type(node.value).__name__)
        elif self.level == MaskingLevel.MEDIUM:
            self._emit("Constant:" + type(node.value).__name__)
        else:  # LOW -> conservamos el valor.
            self._emit("Constant:" + repr(node.value))
        self.generic_visit(node)

    # -- caso general ----------------------------------------------------

    def generic_visit(self, node: ast.AST) -> None:
        # Para cualquier otro nodo emitimos su tipo. Esto cubre operadores
        # (Add, Sub, ...), estructuras de control (For, While, If, ...),
        # comparadores, etc., que son la "forma" del programa.
        self._emit(type(node).__name__)
        super().generic_visit(node)


def tokenize_source(source: str, level: MaskingLevel = MaskingLevel.MEDIUM) -> List[str]:
    """Tokeniza una cadena de codigo Python.

    Lanza ``SyntaxError`` si el codigo no es Python valido; el llamador decide
    como manejar archivos no parseables.
    """
    tree = ast.parse(source)
    tokenizer = _Tokenizer(level)
    # No emitimos el token del nodo Module raiz para no introducir ruido
    # constante; recorremos directamente sus hijos.
    for child in ast.iter_child_nodes(tree):
        tokenizer.visit(child)
    return tokenizer.tokens


def tokenize_file(path: str, level: MaskingLevel = MaskingLevel.MEDIUM) -> List[str]:
    """Lee y tokeniza un archivo ``.py`` (UTF-8, ignora errores de encoding)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        source = fh.read()
    return tokenize_source(source, level)


def ast_depth(source: str) -> int:
    """Profundidad maxima del AST (longitud del camino mas largo raiz->hoja).

    Es una medida estructural global del codigo; la diferencia de profundidad
    entre dos archivos es una de las caracteristicas del modelo.
    """

    def _depth(node: ast.AST) -> int:
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(_depth(c) for c in children)

    return _depth(ast.parse(source))
