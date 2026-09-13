from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "rsimem"
def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE).with_suffix("")
    return ".".join(("rsimem", *relative.parts))


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    package = _module_name(path).rsplit(".", 1)[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")[: len(package.split(".")) - node.level + 1]
                if node.module:
                    parts.extend(node.module.split("."))
                names.add(".".join(parts))
            elif node.module:
                names.add(node.module)
    return names


def test_package_code_does_not_import_repository_experiments() -> None:
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        if any(name in {"__pycache__"} for name in path.parts):
            continue
        if path.parent == PACKAGE / "cli":
            continue
        if any(name == "experiments" or name.startswith("experiments.") for name in _imports(path)):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_stage_one_boundary_packages_exist() -> None:
    for name in ("hosts", "benchmarks", "evaluation", "cli"):
        assert (PACKAGE / name / "__init__.py").is_file()
    assert (ROOT / "experiments" / "README.md").is_file()


def test_memory_systems_have_kind_specific_owners_and_capabilities() -> None:
    from rsimem.memory.contracts import MemoryKind
    from rsimem.memory_systems.registry import HERMES_NATIVE_CAPABILITY, MEM0_FLAT_CAPABILITY

    for relative in (
        "memory_systems/registry.py",
        "memory_systems/semantic/mem0_flat/__init__.py",
        "memory_systems/semantic/hermes_native.py",
        "memory_systems/episodic/hermes_native.py",
        "memory_systems/procedural/hermes_native.py",
    ):
        assert (PACKAGE / relative).is_file()
    assert MEM0_FLAT_CAPABILITY.memory_kinds == (MemoryKind.SEMANTIC,)
    assert HERMES_NATIVE_CAPABILITY.supports(MemoryKind.SEMANTIC)
    assert HERMES_NATIVE_CAPABILITY.supports(MemoryKind.EPISODIC)
    assert HERMES_NATIVE_CAPABILITY.supports(MemoryKind.PROCEDURAL)


def test_domain_code_does_not_import_integration_or_cli_packages() -> None:
    forbidden_prefixes = (
        "rsimem.hosts",
        "rsimem.benchmarks",
        "rsimem.cli",
    )
    offenders = []
    for scope in ("memory", "lifecycle"):
        for path in (PACKAGE / scope).rglob("*.py"):
            imports = _imports(path)
            forbidden = sorted(
                name for name in imports
                if name.startswith(forbidden_prefixes)
            )
            if forbidden:
                offenders.append(f"{path.relative_to(ROOT)}: {forbidden}")
    assert offenders == []


def test_active_experiment_modules_are_canonical() -> None:
    from experiments.adamem.adamem_experiment import AdaMemCondition
    from experiments.base_memory.base_memory_experiment import BaseMemoryCondition

    assert AdaMemCondition.__module__ == "experiments.adamem.adamem_experiment"
    assert BaseMemoryCondition.__module__ == "experiments.base_memory.base_memory_experiment"


def test_host_and_benchmark_code_do_not_import_experiment_protocols() -> None:
    offenders = []
    for scope in (PACKAGE / "hosts", PACKAGE / "benchmarks"):
        for path in scope.rglob("*.py"):
            imported = _imports(path)
            if any(name == "experiments" or name.startswith("experiments.") for name in imported):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_evaluation_code_does_not_import_memory_mutation_or_execution() -> None:
    forbidden_prefixes = (
        "rsimem.memory.executor",
        "rsimem.memory.live_writeback",
        "rsimem.memory.runtime",
        "rsimem.memory.semantic_loop",
    )
    offenders = []
    for path in (PACKAGE / "evaluation").rglob("*.py"):
        imported = _imports(path)
        forbidden = sorted(name for name in imported if name.startswith(forbidden_prefixes))
        if forbidden:
            offenders.append(f"{path.relative_to(ROOT)}: {forbidden}")
    assert offenders == []


def test_contract_layers_do_not_import_launch_or_analysis_layers() -> None:
    forbidden_prefixes = (
        "experiments",
        "rsimem.cli",
        "rsimem.evaluation",
        "rsimem.hosts",
        "rsimem.benchmarks",
    )
    contract_paths = (
        PACKAGE / "memory" / "contracts.py",
        PACKAGE / "memory" / "surface_policy.py",
        PACKAGE / "memory" / "hook_contract.py",
        PACKAGE / "lifecycle" / "contracts.py",
    )
    offenders = []
    for path in contract_paths:
        imported = _imports(path)
        forbidden = sorted(name for name in imported if name.startswith(forbidden_prefixes))
        if forbidden:
            offenders.append(f"{path.relative_to(ROOT)}: {forbidden}")
    assert offenders == []
