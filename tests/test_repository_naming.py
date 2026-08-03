from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIRST_PARTY_ROOTS = (
    ".github",
    "app",
    "charts",
    "docs",
    "experiment",
    "scripts",
    "static",
    "templates",
    "tests",
)
IGNORED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "node_modules",
}


def _is_ascii(value: str) -> bool:
    return value.isascii()


def _first_party_paths():
    for root_name in FIRST_PARTY_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            relative = path.relative_to(ROOT)
            if any(part in IGNORED_PARTS for part in relative.parts):
                continue
            yield relative


def test_first_party_paths_use_ascii_names() -> None:
    invalid = sorted(
        path.as_posix()
        for path in _first_party_paths()
        if not _is_ascii(path.as_posix())
    )

    assert invalid == []


def test_python_identifiers_use_ascii_names() -> None:
    invalid: list[str] = []
    for relative in _first_party_paths():
        if relative.suffix != ".py":
            continue
        source = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(relative))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.alias):
                names.extend(name for name in (node.name, node.asname) if name)
            for name in names:
                if not _is_ascii(name):
                    invalid.append(f"{relative.as_posix()}: {name}")

    assert sorted(set(invalid)) == []
