"""
FILE: tests/test_architecture.py
MODULE 9 — executable architecture rules.

Parses every source file (no imports executed) and fails if a package imports a layer it
must not know about. This turns the rules written in each package's __init__.py into a
regression test, so a careless `from app.messaging import ...` inside `ai/` breaks the build
instead of silently eroding the design.
"""
import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"

# package -> app packages it MAY import (itself is always allowed). None = unrestricted orchestrator.
ALLOWED: dict[str, set[str] | None] = {
    "config": set(),
    "models": set(),
    "utils": {"config"},  # cross-cutting layer (retry, run lock): imported BY others, imports almost nothing
    "collectors": {"models", "utils"},
    "database": {"models"},
    "processing": {"models", "database"},
    "ai": {"models", "config", "utils"},
    "messaging": {"models", "config", "utils"},
    "agents": None,
}


def app_packages_imported(source: str) -> set[str]:
    """Top-level `app.<pkg>` names a module imports, e.g. 'from app.ai.x import y' -> {'ai'}."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "app" and len(parts) > 1:
                found.add(parts[1])
            elif node.module == "app":  # `from app import ai`
                found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "app" and len(parts) > 1:
                    found.add(parts[1])
    return found


def test_detector_finds_all_import_styles():
    src = "import app.ai.x\nfrom app.database import y\nfrom app import messaging\nimport os\nfrom os import path"
    assert app_packages_imported(src) == {"ai", "database", "messaging"}


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_package_respects_its_dependency_rules(package):
    allowed = ALLOWED[package]
    if allowed is None:
        pytest.skip("orchestrator: may import every layer")
    violations = []
    for path in sorted((APP / package).rglob("*.py")):
        illegal = app_packages_imported(path.read_text(encoding="utf-8")) - allowed - {package}
        if illegal:
            violations.append(f"{path.relative_to(APP.parent)} imports {sorted(illegal)}")
    assert not violations, "layer rule broken:\n  " + "\n  ".join(violations)


def test_every_package_directory_is_covered_by_a_rule():
    on_disk = {p.name for p in APP.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert on_disk <= set(ALLOWED), f"add dependency rules for new package(s): {sorted(on_disk - set(ALLOWED))}"
