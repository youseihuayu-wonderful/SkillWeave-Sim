"""Startup-only access to the frozen, repository-owned G1 simulation source.

This module reads Git/files only. It never imports robot transports, publishers,
SDKs, the scene, or a policy. Call activate_source once before source imports;
do not call validation from a render or physics loop.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_SOURCE = Path(__file__).resolve().parent / "third_party" / "g1-vla-control"
SOURCE_URL = "https://github.com/youseihuayu-wonderful/g1-vla-control.git"
SOURCE_REVISION = "a5e40a25c5c080ae114a925dae94215c7cb6d793"
CONTRACT_SHA256 = "839e2a3576b3fbb7b09aa457838b4187e53b100f91fa5f48dc74458b21163374"
ASSETS = {
    "mujoco_menagerie": {
        "revision": "da76818e269b82289eba39808e2fb91d679d6994",
        "url": "https://github.com/google-deepmind/mujoco_menagerie.git",
        "paths": ("unitree_g1", "LICENSE"),
        "required": ("unitree_g1/scene.xml", "unitree_g1/g1.xml", "unitree_g1/LICENSE"),
    },
    "dex1_1_service": {
        "revision": "2986d26eefa4136d4777493e6fd0b8bac7a4c6ae",
        "url": "https://github.com/unitreerobotics/dex1_1_service.git",
        "paths": ("urdf", "LICENSE"),
        "required": ("urdf/dex1_1.urdf", "LICENSE"),
    },
}


def _git(root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args],
            capture_output=True, check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        raise RuntimeError(
            f"Cannot inspect frozen G1 checkout {root}: {detail.decode(errors='replace').strip() or exc}"
        ) from exc
    return result.stdout


def _checkout(root: Path, expected: str) -> None:
    if not (root / ".git").exists():
        raise FileNotFoundError(
            f"G1 checkout/submodule is not initialized: {root}. Initialize the source "
            "and only its mujoco_menagerie and dex1_1_service submodules."
        )
    top = Path(os.fsdecode(_git(root, "rev-parse", "--show-toplevel")).strip()).resolve()
    if top != root:
        raise RuntimeError(f"Expected a separate Git checkout at {root}, found {top}")
    actual = _git(root, "rev-parse", "HEAD").decode().strip()
    if actual != expected:
        raise RuntimeError(f"Frozen revision mismatch at {root}: expected {expected}, got {actual}")


def _tree(root: Path, revision: str, paths: tuple[str, ...] = ()) -> dict:
    records = _git(root, "ls-tree", "-r", "-z", revision, "--", *paths).split(b"\0")
    tree = {}
    for record in records:
        if record:
            metadata, name = record.split(b"\t", 1)
            mode, kind, oid = metadata.decode().split()
            tree[os.fsdecode(name)] = (mode, kind, oid)
    return tree


def _verify_files(root: Path, tree: dict) -> dict:
    count = 0
    for name, (mode, kind, expected) in tree.items():
        if kind != "blob":
            continue
        path = root / name
        if mode not in {"100644", "100755"} or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Missing or unsupported frozen tracked file: {path}")
        content = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen tracked file integrity mismatch: {path}")
        count += 1
    if not count:
        raise RuntimeError(f"No frozen tracked files found in {root}")
    return {"tracked_files_verified": count, "comparison": "working_bytes_against_frozen_git_blobs"}


def source_identity(source: Path) -> dict:
    """Validate the source and scene assets, returning local provenance.

    Every tracked source file is checked against the frozen Git tree, regardless
    of Git's index/stat cache. Asset validation covers the G1 model, Dex1 URDFs,
    meshes, and accompanying notices, not unrelated robot models or SDK code.
    """
    root = Path(source).expanduser().resolve()
    _checkout(root, SOURCE_REVISION)
    tree = _tree(root, SOURCE_REVISION)
    integrity = _verify_files(root, tree)
    for required in ("stack_scene.py", "dex1_gripper.py", "g1_policy_contract.yaml"):
        if required not in tree:
            raise RuntimeError(f"Required frozen source entry missing: {required}")
    # Extra Python files can silently shadow the frozen source's absolute imports.
    for path in root.glob("*.py"):
        if path.name not in tree:
            raise RuntimeError(f"Untracked Python module in frozen source: {path}")
    contract_path = root / "g1_policy_contract.yaml"
    contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if contract_hash != CONTRACT_SHA256:
        raise RuntimeError(f"Frozen policy contract SHA-256 mismatch: {contract_path}")
    assets = {}
    for name, spec in ASSETS.items():
        relative = f"third_party/{name}"
        expected = spec["revision"]
        if tree.get(relative) != ("160000", "commit", expected):
            raise RuntimeError(f"Frozen source gitlink mismatch: {relative}")
        asset_root = root / relative
        _checkout(asset_root, expected)
        asset_tree = _tree(asset_root, expected, spec["paths"])
        for required in spec["required"]:
            if required not in asset_tree:
                raise RuntimeError(f"Required frozen scene asset missing: {asset_root / required}")
        assets[name] = {
            "path": str(asset_root), "revision": expected, "url": spec["url"],
            "validated_paths": list(spec["paths"]),
            "integrity": _verify_files(asset_root, asset_tree),
        }
    modules = sorted(Path(name).stem for name in tree if "/" not in name and name.endswith(".py"))
    return {
        "path": str(root), "url": SOURCE_URL, "revision": SOURCE_REVISION,
        "integrity": integrity, "python_modules": modules,
        "contract": {"path": str(contract_path), "sha256": contract_hash},
        "assets": assets,
        "unitree_sdk": {"initialized": (root / "third_party/unitree_sdk2_python/.git").exists(),
                        "used": False, "required": False},
    }


def activate_source(source: Path | None = None) -> Path:
    """Validate and expose one frozen checkout for its existing absolute imports."""
    identity = source_identity(DEFAULT_SOURCE if source is None else source)
    root = Path(identity["path"])
    menagerie = root / "third_party" / "mujoco_menagerie"
    override = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if override is not None and Path(override).expanduser().resolve() != menagerie:
        raise RuntimeError(
            f"MUJOCO_MENAGERIE_PATH points outside the validated frozen assets: {override}. "
            "Unset it or use the selected source's third_party/mujoco_menagerie."
        )
    for name in identity["python_modules"]:
        module = sys.modules.get(name)
        if module is not None:
            loaded = getattr(module, "__file__", None)
            if loaded is None or Path(loaded).resolve() != root / f"{name}.py":
                raise RuntimeError(
                    f"Cannot mix G1 source checkouts: module {name!r} is already loaded "
                    f"from {loaded!r}, expected {root / (name + '.py')}. Start a fresh process."
                )
    # The upstream scene supports a home-directory fallback. Pin its environment
    # explicitly so neither that fallback nor an external asset override is used.
    os.environ["MUJOCO_MENAGERIE_PATH"] = str(menagerie)
    value = str(root)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)
    return root
