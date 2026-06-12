"""
pipeline/snippet_tokenizer.py
─────────────────────────────
Tokeniza un *snippet* individual de código (cualquier lenguaje) y lo
convierte en una secuencia de tokens normalizados, compatible con el
mismo Winnowing de `pipeline/winnowing.py`.

Estrategia (estilo Dolos: un parser por lenguaje, con fallback genérico):

  1. Si el snippet es Python y parsea con `ast`, se reutiliza
     `pipeline/ast_tokenizer.tokenize_source` (mismo enmascaramiento
     que la iteración 2).
  2. En cualquier otro caso (C++, Java, C#, Go, JS, PHP, fragmentos de
     Python que no parsean...) se usa un tokenizador léxico genérico:
     - palabras clave del lenguaje  → se conservan tal cual
     - identificadores              → VAR
     - literales numéricos/cadenas  → LIT
     - operadores y puntuación      → se conservan tal cual
     - comentarios                  → se eliminan del stream

Uso rápido
----------
    from pipeline.snippet_tokenizer import tokenize_snippet

    res = tokenize_snippet(code, language="Python", masking="medium")
    # → {"tokens": [...], "max_depth": int, "method": "python_ast"|"lexical",
    #    "identifiers": [...], "n_comment_lines": int}
"""

from __future__ import annotations

import re
import textwrap
import warnings

from pipeline.ast_tokenizer import tokenize_source, MASK_VAR, MASK_LIT


# ──────────────────────────────────────────────────────────────────────────────
# Palabras clave por lenguaje (para el tokenizador léxico genérico)
# ──────────────────────────────────────────────────────────────────────────────

_KEYWORDS_COMMON = {
    "if", "else", "for", "while", "return", "break", "continue", "switch",
    "case", "default", "do", "new", "try", "catch", "finally", "throw",
    "class", "static", "void", "int", "float", "double", "char", "bool",
    "boolean", "long", "short", "string", "true", "false", "null", "public",
    "private", "protected", "const", "struct", "enum", "import", "package",
    "func", "var", "let", "def", "elif", "lambda", "yield", "pass", "in",
    "and", "or", "not", "is", "none", "with", "as", "from", "raise",
    "global", "nonlocal", "del", "assert", "async", "await", "this", "self",
    "super", "extends", "implements", "interface", "abstract", "final",
    "override", "virtual", "namespace", "using", "template", "typename",
    "auto", "unsigned", "signed", "sizeof", "typedef", "union", "goto",
    "register", "volatile", "extern", "inline", "operator", "friend",
    "delete", "function", "typeof", "instanceof", "export", "module",
    "echo", "foreach", "endif", "require", "include", "chan", "go", "defer",
    "select", "map", "range", "type", "fallthrough",
}

# Comentarios de línea por lenguaje (familia C usa //, Python/PHP usan #)
_LINE_COMMENT = {
    "python": ("#",),
    "php": ("//", "#"),
    "go": ("//",),
    "java": ("//",),
    "c": ("//",),
    "c++": ("//",),
    "c#": ("//",),
    "javascript": ("//",),
}
_DEFAULT_LINE_COMMENT = ("//", "#")


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizador léxico genérico
# ──────────────────────────────────────────────────────────────────────────────

# Orden importa: cadenas primero (para no confundir // dentro de un string),
# luego comentarios, números, identificadores y operadores.
_LEX_RE = re.compile(
    r"""
      (?P<string>   \"\"\"(?:.|\n)*?\"\"\" | '''(?:.|\n)*?'''
                  | "(?:\\.|[^"\\\n])*"? | '(?:\\.|[^'\\\n])*'? | `(?:\\.|[^`\\])*`? )
    | (?P<block_comment> /\*(?:.|\n)*?(?:\*/|$) )
    | (?P<line_comment>  (?://|\#)[^\n]* )
    | (?P<number>   \d(?:[\w.]|_)* )
    | (?P<ident>    [A-Za-z_]\w* )
    | (?P<op>       [{}()\[\]] | [;,.:?~!@$%^&*+=|/<>-]+ )
    """,
    re.VERBOSE,
)

_OPEN_BRACKETS = {"{": "}", "(": ")", "[": "]"}


def _lexical_tokenize(code: str, language: str) -> dict:
    """
    Tokenizador genérico independiente del lenguaje.

    Retorna tokens normalizados, identificadores crudos (para métricas de
    estilo), número de líneas con comentario y profundidad máxima de
    anidamiento de llaves/paréntesis (aproximación a la profundidad del AST).
    """
    lang = (language or "").strip().lower()
    line_comment_prefixes = _LINE_COMMENT.get(lang, _DEFAULT_LINE_COMMENT)

    tokens: list[str] = []
    identifiers: list[str] = []
    comment_lines: set[int] = set()

    depth = 0
    max_depth = 0
    stack: list[str] = []

    for match in _LEX_RE.finditer(code):
        kind = match.lastgroup
        text = match.group()
        line_no = code.count("\n", 0, match.start())

        if kind in ("block_comment", "line_comment"):
            # En Python "#" sí es comentario, pero "//" es división entera:
            # respetar los prefijos del lenguaje cuando se conoce.
            if kind == "line_comment" and not any(
                text.startswith(p) for p in line_comment_prefixes
            ):
                tokens.append(text[:2])
                continue
            first = line_no
            last = first + text.count("\n")
            comment_lines.update(range(first, last + 1))
            continue

        if kind == "string":
            tokens.append(MASK_LIT)
        elif kind == "number":
            tokens.append(MASK_LIT)
        elif kind == "ident":
            lowered = text.lower()
            if lowered in _KEYWORDS_COMMON:
                tokens.append(lowered)
            else:
                tokens.append(MASK_VAR)
                identifiers.append(text)
        else:  # op
            tokens.append(text)
            if text in _OPEN_BRACKETS:
                stack.append(text)
                depth += 1
                max_depth = max(max_depth, depth)
            elif stack and text == _OPEN_BRACKETS[stack[-1]]:
                stack.pop()
                depth -= 1

    # En Python (sin llaves) la profundidad se aproxima por indentación
    if max_depth == 0:
        max_depth = _indent_depth(code)

    return {
        "tokens": tokens,
        "max_depth": max_depth,
        "identifiers": identifiers,
        "n_comment_lines": len(comment_lines),
        "method": "lexical",
        "error": None,
    }


def _indent_depth(code: str, indent_unit: int = 4) -> int:
    """Niveles máximos de indentación (aprox. anidamiento en Python)."""
    max_indent = 0
    for line in code.splitlines():
        stripped = line.lstrip(" \t")
        if not stripped:
            continue
        spaces = len(line) - len(stripped) + line[: len(line) - len(stripped)].count("\t") * (indent_unit - 1)
        max_indent = max(max_indent, spaces)
    return max_indent // indent_unit


def _python_identifiers(code: str) -> list[str]:
    """Identificadores crudos de un snippet (para métricas de estilo)."""
    return [
        m.group()
        for m in re.finditer(r"[A-Za-z_]\w*", code)
        if m.group().lower() not in _KEYWORDS_COMMON
    ]


def _count_comment_lines(code: str, prefixes: tuple[str, ...]) -> int:
    """Líneas cuyo primer contenido es un comentario de línea."""
    count = 0
    for line in code.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(p) for p in prefixes):
            count += 1
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Función pública
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_snippet(code: str, language: str = "", masking: str = "medium") -> dict:
    """
    Tokeniza un snippet de código de cualquier lenguaje.

    Para Python intenta primero el AST real (mismo tokenizador de la
    iteración 2); si el fragmento no parsea (función suelta, indentación
    truncada...) cae al tokenizador léxico genérico.

    Retorna
    -------
    dict con:
      "tokens"          : list[str] — secuencia normalizada
      "max_depth"       : int       — profundidad AST o anidamiento aproximado
      "identifiers"     : list[str] — identificadores crudos (estilo)
      "n_comment_lines" : int       — líneas con comentario
      "method"          : "python_ast" | "lexical"
      "error"           : None
    """
    code = code or ""
    lang = (language or "").strip().lower()

    if lang == "python":
        # ast.parse emite SyntaxWarning con literales raros (p.ej. "1px");
        # aquí un fallo de parseo es esperado y cae al tokenizador léxico.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            result = tokenize_source(code, masking=masking)
            if result["error"] is not None:
                # Fragmentos con indentación inicial (p.ej. un método suelto)
                result = tokenize_source(textwrap.dedent(code), masking=masking)
        if result["error"] is None:
            return {
                "tokens": result["tokens"],
                "max_depth": result["max_depth"],
                "identifiers": _python_identifiers(code),
                "n_comment_lines": _count_comment_lines(code, ("#",)),
                "method": "python_ast",
                "error": None,
            }

    return _lexical_tokenize(code, language)
