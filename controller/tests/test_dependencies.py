"""Regression guards for the dependency/CVE hardening work.

harden/next-items fixed two starlette CVEs found by pip-audit — PYSEC-2026-249
(request.form() resource limits silently ignored for urlencoded bodies, fixed in
1.3.1) and PYSEC-2026-248 (unvalidated request path shifts request.url's
authority boundary, spoofing request.url.hostname, fixed in 1.3.0) — by raising
the floor to ``starlette>=1.3.1``. An earlier CI fix (#1) declared httpx as a
dev dependency so TestClient-based tests actually run in CI. These tests pin all
of that in place so a future edit cannot silently regress it:

- the ENVIRONMENT actually running this suite has a non-vulnerable starlette;
- pyproject.toml still DECLARES the CVE floor (so fresh installs get it too);
- the CI workflow still runs pip-audit (the gate that found the CVEs);
- .github/dependabot.yml keeps automated update PRs enabled for both the pip
  and github-actions ecosystems (the automation counterpart to pip-audit).

No third-party TOML parser: Python 3.11+ has tomllib, and the repo floor is
``requires-python >= 3.11`` — but the dev venv may run 3.10 (see CLAUDE.md), so
fall back to a line-oriented scan of the dependency strings there.
"""
from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DEPENDABOT = REPO_ROOT / ".github" / "dependabot.yml"

# Minimum non-vulnerable starlette: PYSEC-2026-249 fixed in 1.3.1,
# PYSEC-2026-248 fixed in 1.3.0 -> floor is 1.3.1.
STARLETTE_CVE_FLOOR = (1, 3, 1)


def _version_tuple(raw: str) -> tuple[int, ...]:
    """Leading numeric release segments of a version string ("1.3.1" -> (1, 3, 1)).

    Ignores any non-numeric suffix (rc/dev/post segments) — good enough for
    floor comparisons here without pulling in the `packaging` dependency.
    """
    parts: list[int] = []
    for piece in raw.split("."):
        m = re.match(r"^(\d+)", piece)
        if not m:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


def _pyproject_dependency_strings() -> list[str]:
    """All declared dependency strings (main + every optional-dependency group)."""
    text = PYPROJECT.read_text()
    try:
        import tomllib  # Python 3.11+

        data = tomllib.loads(text)
        project = data.get("project", {})
        deps = list(project.get("dependencies", []))
        for group in project.get("optional-dependencies", {}).values():
            deps.extend(group)
        return deps
    except ModuleNotFoundError:
        # Python 3.10 fallback: pull every quoted "name..." requirement line.
        return re.findall(r'"([A-Za-z0-9_.\-]+\[?[^"]*)"', text)


# ---------------------------------------------------------------------------
# Installed environment (what this suite actually runs against)
# ---------------------------------------------------------------------------


def test_installed_starlette_meets_cve_floor() -> None:
    """The running environment must not contain the PYSEC-2026-248/249 versions."""
    installed = _version_tuple(version("starlette"))
    assert installed >= STARLETTE_CVE_FLOOR, (
        f"installed starlette {version('starlette')} is below the CVE floor "
        f"{'.'.join(map(str, STARLETTE_CVE_FLOOR))} "
        "(PYSEC-2026-249 fixed 1.3.1, PYSEC-2026-248 fixed 1.3.0)"
    )


def test_installed_httpx_available_for_testclient() -> None:
    """TestClient (controller/tests/test_api.py) needs httpx; a missing install
    hard-fails collection in CI — regression guard for fix #1."""
    assert _version_tuple(version("httpx")) >= (0, 28)


# ---------------------------------------------------------------------------
# Declared floors in pyproject.toml (what a fresh install would get)
# ---------------------------------------------------------------------------


def test_pyproject_declares_starlette_cve_floor() -> None:
    deps = _pyproject_dependency_strings()
    starlette_deps = [d for d in deps if re.match(r"^starlette\b", d.strip(), re.IGNORECASE)]
    assert starlette_deps, "pyproject.toml no longer declares starlette at all"
    spec = starlette_deps[0]
    m = re.search(r">=\s*([0-9][0-9.]*)", spec)
    assert m, f"starlette dependency {spec!r} lost its >= floor"
    assert _version_tuple(m.group(1)) >= STARLETTE_CVE_FLOOR, (
        f"declared starlette floor {m.group(1)} was lowered below the CVE-safe 1.3.1"
    )


def test_pyproject_declares_httpx_dev_dependency() -> None:
    deps = _pyproject_dependency_strings()
    assert any(re.match(r"^httpx\b", d.strip(), re.IGNORECASE) for d in deps), (
        "httpx dev dependency disappeared from pyproject.toml — TestClient tests "
        "will silently stop running in a fresh CI install (regression of fix #1)"
    )


def test_pyproject_declares_pip_audit_dev_dependency() -> None:
    deps = _pyproject_dependency_strings()
    assert any(
        re.match(r"^pip[-_]audit\b", d.strip(), re.IGNORECASE) for d in deps
    ), "pip-audit dev dependency disappeared from pyproject.toml"


# ---------------------------------------------------------------------------
# CI + Dependabot configuration files
# ---------------------------------------------------------------------------


def test_ci_workflow_runs_pip_audit() -> None:
    """The CI dependency-audit gate (the thing that caught the starlette CVEs)
    must stay in the workflow."""
    assert CI_WORKFLOW.exists(), ".github/workflows/ci.yml is missing"
    assert "pip-audit" in CI_WORKFLOW.read_text(), (
        "the 'Audit dependencies' pip-audit step was removed from CI"
    )


def test_dependabot_config_exists() -> None:
    assert DEPENDABOT.exists(), ".github/dependabot.yml is missing"


def test_dependabot_covers_pip_ecosystem() -> None:
    text = DEPENDABOT.read_text()
    assert re.search(r'package-ecosystem:\s*"?pip"?', text), (
        "dependabot.yml no longer watches the pip ecosystem"
    )


def test_dependabot_covers_github_actions_ecosystem() -> None:
    text = DEPENDABOT.read_text()
    assert re.search(r'package-ecosystem:\s*"?github-actions"?', text), (
        "dependabot.yml no longer watches the github-actions ecosystem"
    )


def test_dependabot_uses_supported_schema_version() -> None:
    text = DEPENDABOT.read_text()
    assert re.search(r"^version:\s*2\s*$", text, re.MULTILINE), (
        "dependabot.yml must declare schema `version: 2` (the only supported one)"
    )
