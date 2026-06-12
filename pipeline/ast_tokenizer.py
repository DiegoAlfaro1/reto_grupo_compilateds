"""
pipeline/ast_tokenizer.py
─────────────────────────
Convierte un archivo .py en una secuencia de tokens normalizados
usando el módulo `ast` de la biblioteca estándar de Python.

Niveles de enmascaramiento
--------------------------
  low    → solo enmascara nombres de variables/parámetros (Name, arg)
  medium → low + literales numéricos y de cadena (Constant)
  high   → medium + nombres de atributos (Attribute) y de funciones llamadas (Call)

Uso rápido
----------
    from pipeline.ast_tokenizer import tokenize_file, tokenize_source

    tokens = tokenize_file("solucion.py", masking="medium")
    # → ["Module", "FunctionDef", "VAR", "For", "VAR", "VAR", "LIT", ...]
"""

import ast
from pathlib import Path
from typing import Union

# ──────────────────────────────────────────────────────────────────────────────
# Constantes de enmascaramiento
# ──────────────────────────────────────────────────────────────────────────────

MASK_VAR   = "VAR"    # identificador (variable, parámetro, función definida)
MASK_LIT   = "LIT"    # literal (número, cadena, booleano, None)
MASK_ATTR  = "ATTR"   # acceso a atributo  (obj.campo)
MASK_CALL  = "CALL"   # nombre de función llamada


# ──────────────────────────────────────────────────────────────────────────────
# Visitor principal
# ──────────────────────────────────────────────────────────────────────────────

class ASTTokenizer(ast.NodeVisitor):
    """
    Recorre el AST en orden de visita (pre-orden DFS) y emite
    una secuencia de tokens string para cada nodo.

    Parámetros
    ----------
    masking : {"low", "medium", "high"}
        Nivel de normalización de identificadores y literales.
    """

    def __init__(self, masking: str = "medium"):
        if masking not in ("low", "medium", "high"):
            raise ValueError(f"masking debe ser 'low', 'medium' o 'high'; recibido: {masking!r}")
        self.masking = masking
        self.tokens: list[str] = []
        self._depth = 0          # profundidad actual en el árbol
        self.max_depth = 0       # profundidad máxima alcanzada

    # ── Método central ────────────────────────────────────────────────────────

    def _emit(self, token: str) -> None:
        self.tokens.append(token)

    def generic_visit(self, node: ast.AST) -> None:
        """
        Llamado para cada nodo.  Emite el tipo del nodo y luego
        visita recursivamente sus hijos.
        """
        node_type = type(node).__name__

        # Actualizar profundidad
        self._depth += 1
        if self._depth > self.max_depth:
            self.max_depth = self._depth

        # ── Nodos que se procesan con lógica especial ──────────────────────

        if isinstance(node, ast.Name):
            self._handle_Name(node)

        elif isinstance(node, ast.arg):
            self._handle_arg(node)

        elif isinstance(node, ast.Constant):
            self._handle_Constant(node)

        elif isinstance(node, ast.Attribute):
            self._handle_Attribute(node)

        elif isinstance(node, ast.Call):
            self._handle_Call(node)

        # ── Nodos de definición de funciones/clases: emitir tipo + nombre ──
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._emit(node_type)
            self._emit(MASK_VAR)   # el nombre de la función siempre se enmascara

        elif isinstance(node, ast.ClassDef):
            self._emit(node_type)
            self._emit(MASK_VAR)

        # ── Resto de nodos: emitir solo el tipo ────────────────────────────
        else:
            self._emit(node_type)

        # Visitar hijos
        for child in ast.iter_child_nodes(node):
            self.visit(child)

        self._depth -= 1

    # ── Manejadores por tipo de nodo ──────────────────────────────────────────

    def _handle_Name(self, node: ast.Name) -> None:
        """Variables y nombres en expresiones."""
        # low, medium y high enmascaran todos los Name
        self._emit(MASK_VAR)

    def _handle_arg(self, node: ast.arg) -> None:
        """Parámetros de funciones."""
        self._emit(MASK_VAR)
        # Visitar anotación de tipo si existe (pero no el nombre)
        if node.annotation:
            self.visit(node.annotation)

    def _handle_Constant(self, node: ast.Constant) -> None:
        """Literales: números, cadenas, booleanos, None."""
        if self.masking in ("medium", "high"):
            self._emit(MASK_LIT)
        else:
            # low: distingue al menos el tipo del literal
            self._emit(f"Constant_{type(node.value).__name__}")

    def _handle_Attribute(self, node: ast.Attribute) -> None:
        """Accesos a atributos: obj.campo"""
        if self.masking == "high":
            # Enmascara el nombre del atributo
            self._emit("Attribute")
            self._emit(MASK_ATTR)
            # Visitar el objeto receptor (puede ser otra expresión)
            self.visit(node.value)
        else:
            # low/medium: emite el tipo, visita normalmente
            self._emit("Attribute")
            self.visit(node.value)

    def _handle_Call(self, node: ast.Call) -> None:
        """Llamadas a funciones/métodos."""
        self._emit("Call")
        if self.masking == "high":
            # Enmascara el nombre de la función llamada si es un Name simple
            if isinstance(node.func, ast.Name):
                self._emit(MASK_CALL)
            elif isinstance(node.func, ast.Attribute):
                # método: enmascara el atributo pero visita el receptor
                self._emit("Attribute")
                self._emit(MASK_CALL)
                self.visit(node.func.value)
            else:
                self.visit(node.func)
        else:
            self.visit(node.func)

        # Visitar argumentos posicionales y con nombre
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)


# ──────────────────────────────────────────────────────────────────────────────
# Funciones públicas
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_source(
    source: str,
    masking: str = "medium",
    filename: str = "<string>",
) -> dict:
    """
    Tokeniza código fuente Python dado como string.

    Retorna
    -------
    dict con:
      "tokens"    : list[str]  — secuencia de tokens normalizados
      "max_depth" : int        — profundidad máxima del AST
      "error"     : str | None — mensaje de error si el código no es válido
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return {"tokens": [], "max_depth": 0, "error": str(exc)}

    visitor = ASTTokenizer(masking=masking)
    visitor.visit(tree)

    return {
        "tokens": visitor.tokens,
        "max_depth": visitor.max_depth,
        "error": None,
    }


def tokenize_file(
    path: Union[str, Path],
    masking: str = "medium",
    encoding: str = "utf-8",
) -> dict:
    """
    Tokeniza un archivo .py en disco.

    Retorna el mismo dict que `tokenize_source`.
    """
    path = Path(path)
    try:
        source = path.read_text(encoding=encoding)
    except (OSError, UnicodeDecodeError) as exc:
        return {"tokens": [], "max_depth": 0, "error": str(exc)}

    return tokenize_source(source, masking=masking, filename=str(path))