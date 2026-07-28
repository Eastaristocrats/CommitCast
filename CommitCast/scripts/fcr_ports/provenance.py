"""Pinned upstream identities and strict provenance checks."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Final


PINNED_UPSTREAMS: Final[dict[str, dict[str, str]]] = {
    "COSA": {
        "commit": "43a8c8da4de74d5745a8713f6130c523b7df2694",
        "module": "tta/cosa.py",
        "module_sha256": "a79ba62f509e2e4bbafb1590708ec851af56763431173e660db33e00ae4c3abf",
        "license_sha256": "09ee9e203ad9bb0548ce248e0906ba3215d6feb47e54f02c8fbf82349fe07f39",
    },
    "TAFAS": {
        "commit": "139bf980671da4daad728a0fc21d8df508b9203d",
        "module": "tta/tafas.py",
        "module_sha256": "7ff75bb08c6efadcc8dcedc3f9599b62ebb5397da77485c4254d78c3972b3932",
        "license_sha256": "5692a759b37b18116674d09e6a0dad678c512becbe744bb2e509da2664aeb10e",
    },
    "PETSA": {
        "commit": "87853d888e98311ac94e64be920d17b57143b20c",
        "module": "tta/petsa.py",
        "module_sha256": "7704007c72aa018ed5f838ff7bdb99a48056116a35f32a7482d15c30d8e3feb2",
        "license_sha256": "09ee9e203ad9bb0548ce248e0906ba3215d6feb47e54f02c8fbf82349fe07f39",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_upstream(
    repo: Path,
    *,
    project: str,
    expected_commit: str,
    expected_module_sha256: str,
    expected_license_sha256: str,
) -> dict[str, str]:
    """Reject an unpinned, modified, or content-mismatched upstream checkout."""
    project = str(project).upper()
    if project not in PINNED_UPSTREAMS:
        raise ValueError(f"unsupported upstream project: {project}")
    pinned = PINNED_UPSTREAMS[project]
    expected = {
        "commit": str(expected_commit).lower(),
        "module_sha256": str(expected_module_sha256).lower(),
        "license_sha256": str(expected_license_sha256).lower(),
    }
    for key, value in expected.items():
        if value != pinned[key]:
            raise ValueError(f"{project} {key} differs from the pinned identity")

    repo = repo.resolve()
    module_path = repo / pinned["module"]
    license_path = repo / "LICENSE"
    if not module_path.is_file():
        raise FileNotFoundError(f"missing upstream module: {module_path}")
    if not license_path.is_file():
        raise FileNotFoundError(f"missing upstream license: {license_path}")
    if not (repo / ".git").exists():
        raise ValueError(
            f"{project} must be a Git checkout so the pinned commit can be verified"
        )

    actual_commit = _git(repo, "rev-parse", "HEAD").lower()
    if actual_commit != expected["commit"]:
        raise ValueError(
            f"{project} commit mismatch: expected {expected['commit']}, got {actual_commit}"
        )
    tracked_changes = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if tracked_changes:
        raise ValueError(f"{project} checkout has modified tracked files")

    actual_module_sha256 = sha256_file(module_path)
    actual_license_sha256 = sha256_file(license_path)
    if actual_module_sha256 != expected["module_sha256"]:
        raise ValueError(f"{project} module SHA-256 mismatch")
    if actual_license_sha256 != expected["license_sha256"]:
        raise ValueError(f"{project} license SHA-256 mismatch")
    return {
        "upstream_project": project,
        "upstream_commit": actual_commit,
        "upstream_module": pinned["module"],
        "upstream_module_sha256": actual_module_sha256,
        "upstream_license": "LICENSE",
        "upstream_license_sha256": actual_license_sha256,
        "upstream_verification": "git_commit_clean_and_content_hash",
    }
