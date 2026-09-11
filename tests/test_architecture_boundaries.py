from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "rsimem"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_package_code_does_not_import_repository_experiments() -> None:
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        if any(name in {"__pycache__"} for name in path.parts):
            continue
        if any(name == "experiments" or name.startswith("experiments.") for name in _imports(path)):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_stage_one_boundary_packages_exist() -> None:
    for name in ("hosts", "benchmarks", "evaluation", "cli"):
        assert (PACKAGE / name / "__init__.py").is_file()
    assert (ROOT / "experiments" / "README.md").is_file()
