"""Inventory the local frozen G1 baseline without inference or robot access.

Usage: python g1_baseline_manifest.py --output results/g1_baseline_manifest.json
Use --input-root for locally authorized author-32 packages, or --checkpoint for
an existing checkpoint directory. Nothing is downloaded, installed, restored,
or executed. Existing evidence is never overwritten. This is not an E4 result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

from g1_source import DEFAULT_SOURCE, source_identity


ROOT = Path(__file__).resolve().parent
POLICY_REVISION = "cced7a7ff7b454fdcac555457a1a2a3dc262ac77"
DATA_REVISION = "c468ef259ff8bfad1b3b0e7b2c1f45efbb0b30ba"
# Historical artifact hashes are references, not evidence of local availability.
AUTHOR32_FILES = {
    "synchronized_samples": (
        "lgg100_author32_semantic_samples.npz",
        "e0d7e57c859aa6bbf53792fbf9e14df2cd0a7833319a914325d03174b3487d83",
    ),
    "sample_manifest": (
        "lgg100_author32_semantic_samples_manifest.json",
        "3e7f283b4b9cc79636a65d1857added17035b4b27562fa480248fcefd6009ee4",
    ),
    "semantic_outputs": (
        "lgg100_author32_semantic_outputs_quarantined.npz",
        "92f360ef059617c915d4dabcb90adafa88531528f7cde2d426ff2e5958882173",
    ),
    "realtime_raw_canonical_package": (
        "lgg100_author32_closed_loop_t0_offset008_baseline_realtime_quarantined.npz",
        "3f8d8eb0e48b5d950683cd78374f5210876c11cd0a7d50f1dcf1549e8c063386",
    ),
    "paused_raw_canonical_package": (
        "lgg100_author32_closed_loop_t0_offset008_baseline_paused_quarantined.npz",
        "a5adde9e18c506abed0fa3c3ef2f8430f4e753601c2346e7c6eaac392f7264cd",
    ),
}


def _file_inventory(path: Path, expected_hash: str | None = None) -> dict:
    item = {"path": str(path), "available": path.is_file(), "expected_sha256": expected_hash}
    if not item["available"]:
        item.update(sha256=None, bytes=None, hash_matches_reference=None)
        return item
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    item.update(
        sha256=digest, bytes=path.stat().st_size,
        hash_matches_reference=None if expected_hash is None else digest == expected_hash,
    )
    return item


def _find_input(roots: list[Path], filename: str, expected_hash: str | None) -> dict:
    candidates = [_file_inventory(root / filename, expected_hash) for root in roots]
    return {
        "filename": filename, "candidates": candidates,
        "available": any(item["available"] for item in candidates),
        "reference_hash_verified": any(item["hash_matches_reference"] is True for item in candidates),
        "content_schema_validated": False,
    }


def _hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser().resolve()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser().resolve() / "hub"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser().resolve() / "huggingface/hub"


def build_manifest(source: Path | None = None, *, input_roots: list[Path] | None = None,
                   checkpoint: Path | None = None, norm_stats: Path | None = None) -> dict:
    """Read local provenance; file presence never implies policy/task readiness."""
    identity = source_identity(DEFAULT_SOURCE if source is None else source)
    source_root = Path(identity["path"])
    contract = json.loads((source_root / "g1_policy_contract.yaml").read_text())
    policy = contract["production_policy"]
    roots = list(dict.fromkeys(
        path.expanduser().resolve() for path in [*(input_roots or []), ROOT / "results", source_root / "results"]
    ))
    cache = _hub_cache()
    checkpoint_path = (
        checkpoint.expanduser().resolve() if checkpoint is not None else
        cache / "models--LGG100--stack-cube-eef-24k" / "snapshots" / POLICY_REVISION
    )
    audit_path = source_root / "results/checkpoint_metadata_audit.json"
    audit = json.loads(audit_path.read_text())
    expected_files = audit["public_tree"]["files"]
    present_files = [name for name in expected_files if (checkpoint_path / name).is_file()]
    checkpoint_inventory = {
        "path": str(checkpoint_path), "selection": "explicit" if checkpoint else "pinned_huggingface_cache_snapshot",
        "directory_exists": checkpoint_path.is_dir(),
        "available": len(present_files) == len(expected_files),
        "expected_file_count": len(expected_files), "present_files": present_files,
        "missing_files": [name for name in expected_files if name not in present_files],
        "present_bytes": sum((checkpoint_path / name).stat().st_size for name in present_files),
        "historical_expected_bytes": audit["public_tree"]["listed_bytes"],
        "reference_file_listing": _file_inventory(audit_path),
        "content_hashes_verified": False, "strict_restoration_performed": False,
        "note": "Presence inventory only; directory name and byte counts do not verify checkpoint identity or completeness.",
    }
    norm_path = (
        norm_stats.expanduser().resolve() if norm_stats is not None else
        checkpoint_path / "assets/stack-cube-eef/norm_stats.json"
    )
    norm_inventory = _file_inventory(norm_path)
    norm_inventory["paired_checkpoint_provenance_verified"] = False
    inputs = {name: _find_input(roots, filename, digest)
              for name, (filename, digest) in AUTHOR32_FILES.items()}
    trace = source_root / "results/lgg100_author32_online_ik_convergence_trace.json"
    raw_caches = {}
    for name, path in {
        "huggingface_snapshot": cache / "datasets--LGG100--Stack-the-cubes" / "snapshots" / DATA_REVISION,
        "legacy_project_cache": Path.home() / ".cache/g1_vla_control/stack_the_cubes",
    }.items():
        files = sorted(path.rglob("*.parquet")) if path.is_dir() else []
        raw_caches[name] = {
            "path": str(path), "directory_exists": path.is_dir(),
            "parquet_file_count": len(files),
            "parquet_files": [_file_inventory(file) for file in files],
            "raw_dataset_norm_stats": _file_inventory(path / "norm_stats.json"),
            "pinned_revision_verified": False, "policy_ready_eef_samples": False,
            "note": "Raw joint-space data/statistics are not checkpoint EEF inputs/statistics.",
        }
    distributions = sorted(
        ({"name": dist.metadata.get("Name", "unknown"), "version": dist.version}
         for dist in metadata.distributions()),
        key=lambda item: (item["name"].lower(), item["version"]),
    )
    dependencies = {}
    for name in ("mujoco", "numpy", "Pillow", "aiohttp", "jax", "jaxlib", "openpi", "orbax-checkpoint"):
        try:
            dependencies[name] = {"installed": True, "version": metadata.version(name)}
        except metadata.PackageNotFoundError:
            dependencies[name] = {"installed": False, "version": None}
    missing = [name for name, item in inputs.items() if not item["available"]]
    if not checkpoint_inventory["available"]:
        missing.append("complete_checkpoint_file_set")
    if not norm_inventory["available"]:
        missing.append("checkpoint_normalization_statistics")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "E0 local source, asset, dependency and input inventory; not E4 success or task execution",
        "source": identity,
        "workspace_files": {
            name: _file_inventory(ROOT / name) for name in (
                ".gitmodules", "pyproject.toml", "uv.lock", "g1_source.py",
                "g1_baseline_manifest.py", "g1_preflight.py", "g1_first_target.py",
                "g1_scene_preview.py", "tests/test_g1_first_target.py",
                "g1_arm_skill.py", "tests/test_g1_arm_skill.py",
            )
        },
        "host": {"system": platform.system(), "release": platform.release(),
                 "version": platform.version(), "machine": platform.machine(),
                 "processor": platform.processor(), "node": platform.node()},
        "python": {"version": sys.version, "implementation": platform.python_implementation(),
                   "executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix},
        "dependencies": dependencies, "installed_distributions": distributions,
        "accelerator": {
            "device_probe_performed": False, "gpu_devices": None,
            "cuda_runtime_version": None, "nvidia_driver_version": None,
            "status": "no_current_device_or_GPU_execution_evidence",
            "note": "Package metadata is not a GPU probe. No JAX backend, CUDA device, SSH, or inference was initialized.",
        },
        "frozen_references": {
            "policy": {"repository": policy["model"], "revision": policy["revision"],
                       "url": f"https://huggingface.co/{policy['model']}/tree/{policy['revision']}"},
            "dataset": {"repository": "LGG100/Stack-the-cubes", "revision": DATA_REVISION,
                        "url": f"https://huggingface.co/datasets/LGG100/Stack-the-cubes/tree/{DATA_REVISION}"},
            "author_transform": {"url": policy["official_transform_repository"],
                                 "revision": policy["official_transform_commit"]},
            "author_deployment_reference_only": {"url": policy["deployment_repository"],
                                                  "revision": policy["deployment_commit"],
                                                  "code_copied_or_executed": False,
                                                  "license_established_for_reuse": False},
            "complete_author_training_config_available": policy["complete_author_train_config_available"],
            "exact_historical_openpi_commit_recovered": False,
        },
        "registered_configuration": {
            "contract_id": contract["contract_id"], "contract_version": contract["contract_version"],
            "contract_sha256": identity["contract"]["sha256"],
            "robot": contract["robot"], "timing": contract["timing"], "action": contract["action"],
            "model_loading_reference": {"model_config_name": policy["model_config_name"],
                                        "restore_mode": policy["framework"],
                                        "action_horizon": contract["timing"]["action_horizon"],
                                        "discrete_state_input": contract["observation"]["discrete_state_input"]},
            "actual_model_loading_parameters": None,
        },
        "inputs": {
            "searched_package_roots": [str(root) for root in roots],
            "huggingface_cache_root": str(cache), "checkpoint": checkpoint_inventory,
            "normalization_statistics": norm_inventory, "author32_packages": inputs,
            "saved_first_target_trace": {
                **_file_inventory(trace),
                "scope": "Saved numerical targets only; not synchronized multimodal observations or a complete raw/canonical chunk.",
            },
            "raw_dataset_caches": raw_caches,
        },
        "execution": {
            "hardware_execution_enabled": False, "robot_transport_initialized": False,
            "sdk_initialized": False, "publisher_initialized": False,
            "real_policy_execution_enabled": False, "real_policy_executed": False,
            "model_loaded": False, "inference_calls": 0,
            "kinematic_check_performed": False, "physical_simulation_performed": False,
            "neural_closed_loop_task_performed": False, "e4_success": False,
        },
        "limitations": {
            "missing_inputs_in_searched_locations": missing,
            "input_schema_and_cross_artifact_provenance_verified": False,
            "checkpoint_identity_and_restore_verified": False,
            "gpu_execution_verified": False, "complete_training_config_provenance": False,
            "scene_calibration_verified": False, "task_success_criteria_validated": False,
            "inventory_does_not_unlock_execution": True,
            "remote_artifacts": "Not inspected by this inventory; no successful remote artifact verification.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON evidence path; must not exist")
    parser.add_argument("--g1-source", type=Path, help="Explicit frozen checkout; defaults to the repository submodule")
    parser.add_argument("--input-root", type=Path, action="append", help="Additional local directory of original author-32 packages (repeatable)")
    parser.add_argument("--checkpoint", type=Path, help="Existing checkpoint directory; no downloading or loading")
    parser.add_argument("--norm-stats", type=Path, help="Existing paired norm_stats.json; pairing remains unverified")
    args = parser.parse_args()
    destination = args.output.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        parser.error(f"Refusing to overwrite evidence: {destination}")
    source_root = (DEFAULT_SOURCE if args.g1_source is None else args.g1_source).expanduser().resolve()
    if destination.resolve().is_relative_to(source_root):
        parser.error("The frozen G1 source is read-only; choose an output outside it")
    try:
        manifest = build_manifest(args.g1_source, input_roots=args.input_root,
                                  checkpoint=args.checkpoint, norm_stats=args.norm_stats)
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(destination)


if __name__ == "__main__":
    main()
