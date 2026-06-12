"""
pipeline/transforms.py
──────────────────────
Transformaciones de ofuscación documentadas para generar variantes de
plagio a partir de un archivo .py (dataset ad hoc, sección 5.3 del
marco de referencia).

Cada transformación simula una técnica de ofuscación común en plagio
académico (sección 8.1 del marco de referencia):

  T1  rename_identifiers      → renombrado de variables, funciones y parámetros
  T2  insert_dead_code        → inserción de código muerto (asignaciones sin uso,
                                 bloques `if False`)
  T3  expand_aug_assign       → reescritura  x += y   ⇒   x = x + y
  T4  flip_comparisons        → reescritura  a < b    ⇒   b > a
  T5  reorder_functions       → reordenamiento de definiciones de funciones
                                 independientes a nivel de módulo

Todas operan sobre el AST (ast.NodeTransformer) y regeneran el código con
ast.unparse, por lo que el resultado siempre es Python válido.

Uso rápido
----------
    from pipeline.transforms import make_variant

    variante = make_variant(codigo_fuente, seed=7, intensity=0.8)
"""

from __future__ import annotations

import ast
import builtins
import random

_BUILTINS = frozenset(dir(builtins))


# ──────────────────────────────────────────────────────────────────────────────
# T1 — Renombrado de identificadores
# ──────────────────────────────────────────────────────────────────────────────

class _BoundNameCollector(ast.NodeVisitor):
    """Recolecta los nombres definidos EN el archivo (no los importados
    ni los builtins), que son los únicos seguros de renombrar."""

    def __init__(self):
        self.bound: set[str] = set()
        self.imported: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imported.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.imported.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self.bound.add(node.arg)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        self.generic_visit(node)
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        if node.name in self.mapping:
            node.name = self.mapping[node.name]
        return node

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        if node.name in self.mapping:
            node.name = self.mapping[node.name]
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        if node.name in self.mapping:
            node.name = self.mapping[node.name]
        return node

    def visit_Global(self, node: ast.Global) -> ast.Global:
        node.names = [self.mapping.get(n, n) for n in node.names]
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Nonlocal:
        node.names = [self.mapping.get(n, n) for n in node.names]
        return node


_NAME_POOL = [
    "data", "value", "result", "item", "temp", "aux", "elem", "buffer",
    "output", "entry", "holder", "obj", "thing", "node", "acc", "total",
    "count", "current", "state", "info", "payload", "chunk", "piece",
]


def rename_identifiers(tree: ast.Module, rng: random.Random) -> ast.Module:
    """T1 — Renombra de forma consistente todos los identificadores
    definidos en el archivo (excluye builtins e importados)."""
    collector = _BoundNameCollector()
    collector.visit(tree)
    renameable = sorted(collector.bound - collector.imported - _BUILTINS)

    pool = _NAME_POOL[:]
    rng.shuffle(pool)
    mapping = {}
    for i, name in enumerate(renameable):
        base = pool[i % len(pool)]
        new = f"{base}_{rng.randint(1, 99)}"
        while new in mapping.values() or new in collector.bound:
            new = f"{base}_{rng.randint(1, 999)}"
        mapping[name] = new

    return _Renamer(mapping).visit(tree)


# ──────────────────────────────────────────────────────────────────────────────
# T2 — Inserción de código muerto
# ──────────────────────────────────────────────────────────────────────────────

def _dead_statement(rng: random.Random) -> ast.stmt:
    """Genera una sentencia inocua: asignación sin uso o bloque if False."""
    kind = rng.choice(["assign", "if_false", "assign"])
    var = f"unused_{rng.randint(0, 999)}"
    if kind == "assign":
        return ast.Assign(
            targets=[ast.Name(id=var, ctx=ast.Store())],
            value=ast.Constant(value=rng.randint(0, 100)),
        )
    return ast.If(
        test=ast.Constant(value=False),
        body=[ast.Expr(value=ast.Constant(value="dead"))],
        orelse=[],
    )


class _DeadCodeInserter(ast.NodeTransformer):
    def __init__(self, rng: random.Random, rate: float):
        self.rng = rng
        self.rate = rate          # probabilidad de insertar tras cada sentencia

    def _inject(self, body: list[ast.stmt]) -> list[ast.stmt]:
        new_body: list[ast.stmt] = []
        for stmt in body:
            new_body.append(stmt)
            # No insertar después de return/raise (sería inalcanzable y obvio)
            if not isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
                if self.rng.random() < self.rate:
                    new_body.append(_dead_statement(self.rng))
        return new_body

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body = self._inject(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = self._inject(node.body)
        return node


def insert_dead_code(tree: ast.Module, rng: random.Random, rate: float = 0.3) -> ast.Module:
    """T2 — Inserta asignaciones sin uso y bloques `if False` entre sentencias."""
    return _DeadCodeInserter(rng, rate).visit(tree)


# ──────────────────────────────────────────────────────────────────────────────
# T3 — Expansión de asignaciones aumentadas
# ──────────────────────────────────────────────────────────────────────────────

class _AugAssignExpander(ast.NodeTransformer):
    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        self.generic_visit(node)
        # Solo expandir targets simples (Name); subscripts/atributos cambian
        # semántica si la expresión receptora tiene efectos secundarios.
        if not isinstance(node.target, ast.Name):
            return node
        load_target = ast.Name(id=node.target.id, ctx=ast.Load())
        return ast.Assign(
            targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
            value=ast.BinOp(left=load_target, op=node.op, right=node.value),
        )


def expand_aug_assign(tree: ast.Module) -> ast.Module:
    """T3 — Reescribe `x += y` como `x = x + y` (y análogos)."""
    return _AugAssignExpander().visit(tree)


# ──────────────────────────────────────────────────────────────────────────────
# T4 — Inversión de comparaciones
# ──────────────────────────────────────────────────────────────────────────────

_MIRROR = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE}


class _ComparisonFlipper(ast.NodeTransformer):
    def __init__(self, rng: random.Random):
        self.rng = rng

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        self.generic_visit(node)
        if len(node.ops) == 1 and type(node.ops[0]) in _MIRROR:
            if self.rng.random() < 0.7:
                node.left, node.comparators[0] = node.comparators[0], node.left
                node.ops[0] = _MIRROR[type(node.ops[0])]()
        return node


def flip_comparisons(tree: ast.Module, rng: random.Random) -> ast.Module:
    """T4 — Invierte el orden de los operandos: `a < b` ⇒ `b > a`."""
    return _ComparisonFlipper(rng).visit(tree)


# ──────────────────────────────────────────────────────────────────────────────
# T5 — Reordenamiento de funciones a nivel de módulo
# ──────────────────────────────────────────────────────────────────────────────

def reorder_functions(tree: ast.Module, rng: random.Random) -> ast.Module:
    """T5 — Baraja bloques contiguos de FunctionDef a nivel de módulo.
    Las definiciones de funciones son independientes entre sí mientras
    no se llamen al momento de la definición."""
    body = tree.body
    i = 0
    new_body: list[ast.stmt] = []
    while i < len(body):
        if isinstance(body[i], (ast.FunctionDef, ast.AsyncFunctionDef)):
            j = i
            while j < len(body) and isinstance(body[j], (ast.FunctionDef, ast.AsyncFunctionDef)):
                j += 1
            block = body[i:j]
            rng.shuffle(block)
            new_body.extend(block)
            i = j
        else:
            new_body.append(body[i])
            i += 1
    tree.body = new_body
    return tree


# ──────────────────────────────────────────────────────────────────────────────
# Función principal: generar una variante plagiada
# ──────────────────────────────────────────────────────────────────────────────

def make_variant(source: str, seed: int, intensity: float = 0.7) -> str | None:
    """
    Genera una variante "plagiada" del código fuente aplicando las
    transformaciones documentadas T1–T5.

    Parámetros
    ----------
    source    : código fuente Python original
    seed      : semilla del generador aleatorio (reproducibilidad)
    intensity : en [0, 1]; controla cuántas transformaciones opcionales
                se aplican y con qué agresividad.
                  - T1 (renombrado) se aplica SIEMPRE: es la técnica de
                    ofuscación más básica y universal.
                  - T2–T5 se aplican cada una con probabilidad `intensity`.

    Retorna
    -------
    str con el código transformado, o None si el original no parsea.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    rng = random.Random(seed)

    tree = rename_identifiers(tree, rng)

    if rng.random() < intensity:
        tree = expand_aug_assign(tree)
    if rng.random() < intensity:
        tree = flip_comparisons(tree, rng)
    if rng.random() < intensity:
        tree = reorder_functions(tree, rng)
    if rng.random() < intensity:
        tree = insert_dead_code(tree, rng, rate=0.15 + 0.25 * intensity)

    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return None
