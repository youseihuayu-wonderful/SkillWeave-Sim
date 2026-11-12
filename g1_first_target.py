"""Replay two saved canonical analysis targets through offline kinematic gates.

No full raw32 archive is local. Each first action is held for two samples only
because EEFActionChunk requires two timestamps; the hold is not a recovered
second action or a neural-policy/chunk/task replay. No dynamics are advanced.

The saved JSON contains no complete MjData snapshot. Residual and cube-position
parity establishes only the recorded kinematic initialization, not bit-exact
physical-state replay. This candidate gate does not replace the original
submodule's legacy neural closed-loop entry point.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from g1_preflight import bounded_ik_config, create_preflight
from g1_source import DEFAULT_SOURCE, activate_source

TRACE_PATH = Path("results/lgg100_author32_online_ik_convergence_trace.json")
REPORT_TEMPLATE = "results/lgg100_author32_closed_loop_t0_offset008_baseline_{}.json"


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_evidence(source: Path) -> tuple[dict, dict, dict]:
    """Load committed analysis JSON, never a missing/raw neural archive."""
    trace_bytes = (source / TRACE_PATH).read_bytes()
    trace = json.loads(trace_bytes)
    reports = {}
    provenance = {
        str(TRACE_PATH): {"sha256": _hash(trace_bytes), "size_bytes": len(trace_bytes)}
    }
    for mode in ("realtime", "paused"):
        relative = REPORT_TEMPLATE.format(mode)
        payload = (source / relative).read_bytes()
        reports[mode] = json.loads(payload)
        provenance[relative] = {"sha256": _hash(payload), "size_bytes": len(payload)}
    return trace, reports, provenance


def initialize_scene(source: Path, translation=(0.08, 0.0, 0.0)):
    """Recreate the recorded nominal episode-0 scene without settling/stepping."""
    activate_source(source)
    import mujoco
    from stack_scene import build_model, reset_to_reference_pose, translate_cubes

    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    translate_cubes(model, data, np.asarray(translation, dtype=np.float64))
    return model, data


def first_target_hold(action):
    """Preserve a canonical action exactly in an explicitly synthetic hold."""
    from action_schema import EEFActionChunk

    action = np.asarray(action, dtype=np.float64)
    chunk = EEFActionChunk(np.array([0.0, 1.0 / 15.0]), np.stack((action, action)))
    if chunk.actions[0].tobytes() != action.tobytes():
        raise ValueError("saved analysis action is not canonical; refusing normalization")
    return chunk


def initial_residuals(model, data, action) -> tuple[np.ndarray, dict]:
    """Use the frozen diagnostic's FK/error calculation, without changing data."""
    import mujoco
    from action_schema import pelvis_vla_action_to_world_mujoco
    from g1_dual_arm_ik import G1DualArmIK
    from lgg100_ik_convergence_diagnostic import _errors

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    target = pelvis_vla_action_to_world_mujoco(
        action, data.xpos[pelvis], data.xquat[pelvis]
    )
    errors = _errors(G1DualArmIK(model, data), target, 0)
    result = {}
    for side in ("left", "right"):
        result[f"{side}_position_error_m"] = errors[f"{side}_position_error_m"]
        result[f"{side}_orientation_error_deg"] = float(np.rad2deg(
            errors[f"{side}_orientation_error_rad"]
        ))
    return target, result


def _parity(current: dict, saved: dict) -> dict:
    comparisons = {}
    for key, value in current.items():
        tolerance = 1e-7 if key.endswith("_deg") else 1e-9
        delta = abs(value - saved[key])
        comparisons[key] = {
            "current": value, "saved": saved[key], "absolute_difference": delta,
            "absolute_tolerance": tolerance, "matches": delta <= tolerance,
        }
    return {
        "matches": all(item["matches"] for item in comparisons.values()),
        "comparisons": comparisons,
    }


def diagnose_mode(source: Path, saved: dict, original_report: dict) -> dict:
    """Measure isolated solver reachability separately from full gate acceptance."""
    import mujoco
    from g1_fast_sequential_ik import solve_sequential_ik

    started = time.perf_counter_ns()
    model, data = initialize_scene(source, original_report["cube_translation_m"])
    action = np.asarray(saved["raw_first_action"], dtype="<f8")
    action_bytes = action.tobytes()
    canonical_json = json.dumps(action.tolist(), separators=(",", ":"), allow_nan=False)
    snapshot = {
        name: getattr(data, name).copy()
        for name in ("qpos", "qvel", "ctrl", "qacc", "act", "xpos", "site_xpos")
    }
    initial_time = data.time
    target, residuals = initial_residuals(model, data, action)
    parity = _parity(residuals, saved["trace"][0])
    world_matches = bool(
        np.allclose(target[:3], saved["target_world"]["left_position"], rtol=0, atol=1e-12)
        and np.allclose(target[7:10], saved["target_world"]["right_position"], rtol=0, atol=1e-12)
    )
    cubes = {}
    for color in ("red", "blue", "yellow"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{color}_cube")
        cubes[color] = data.xpos[body].tolist()
    cube_matches = all(np.allclose(
        position, original_report["initial_cubes"][color]["position_m"], rtol=0, atol=1e-12
    ) for color, position in cubes.items())
    chunk = first_target_hold(action)
    ik = solve_sequential_ik(model, data, action[None, :], bounded_ik_config(source))
    gate = create_preflight(model, source).check(data, chunk, phase="free_space")
    unchanged = data.time == initial_time and all(
        np.array_equal(getattr(data, name), value) for name, value in snapshot.items()
    )
    if not unchanged:
        raise RuntimeError("offline diagnostic unexpectedly mutated source MjData")
    initial_matches = parity["matches"] and world_matches and cube_matches
    return {
        "action": {
            "origin": "saved JSON canonical analysis values; not original NPZ bytes",
            "values": action.tolist(), "dtype": "<f8", "shape": [16],
            "bytes_base64": base64.b64encode(action_bytes).decode("ascii"),
            "bytes_sha256": _hash(action_bytes),
            "canonical_json": canonical_json,
            "canonical_json_sha256": _hash(canonical_json.encode("utf-8")),
            "preserved_by_chunk_constructor": chunk.actions[0].tobytes() == action_bytes,
            "original_archive_path_unavailable_locally": saved["artifact"],
            "original_archive_reported_sha256": original_report["chunks_output_sha256"],
            "original_full_chunk_reported_sha256": original_report["records"][0]["raw_sha256"],
        },
        "scene": {
            "initialization": "build_model; reset_to_reference_pose; translate_cubes; no mj_step",
            "cube_translation_m": original_report["cube_translation_m"],
            "cube_positions_m": cubes, "cube_positions_match_saved_report": cube_matches,
            "physics_timestep_s": float(model.opt.timestep), "simulation_time_s": data.time,
            "qpos_sha256": _hash(np.asarray(data.qpos, dtype="<f8").tobytes()),
        },
        "initial_residual_parity": parity,
        "world_target_positions_match_saved": world_matches,
        "world_target_wxyz": target.tolist(),
        "initial_scene_matches_saved_evidence": initial_matches,
        "legacy_evidence_not_rerun": {
            "online_ik_iterations": original_report["online_ik_iterations"],
            "swept_path_preflight": original_report["records"][0]["swept_path_preflight"],
            "errors_at_30": next(item for item in saved["trace"] if item["iteration"] == 30),
            "passes": saved["passes"],
        },
        "solver": {
            "accepted": ik.accepted, "reason": ik.reason,
            "total_iterations": ik.total_iterations, "elapsed_ms": ik.elapsed_ms,
            "targets": [dict(asdict(item), joint_target_rad=item.joint_target_rad.tolist())
                        for item in ik.targets],
        },
        "swept_collision_gate": asdict(gate),
        "synthetic_hold": {"samples": 2, "timestamps_s": chunk.timestamps.tolist(),
                           "distinct_saved_targets": 1, "phase": "free_space"},
        "source_data_unchanged": unchanged,
        "false_unreachable_resolved": initial_matches and ik.accepted,
        "saved_first_target_full_gate_accepted": initial_matches and gate.accepted,
        "total_diagnostic_elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000,
    }


def build_diagnostic(source: Path | None = None) -> dict:
    source = activate_source(source)
    import mujoco

    trace, reports, provenance = load_evidence(source)
    modes = {mode: diagnose_mode(source, trace["modes"][mode], reports[mode])
             for mode in ("realtime", "paused")}
    return {
        "schema": "skillweave.g1.saved-first-target.v1",
        "scope": "Offline canonical saved-first-target kinematic analysis only",
        "source": str(source), "source_evidence": provenance,
        "runtime": {"mujoco": mujoco.__version__, "numpy": np.__version__},
        "cause_from_frozen_code_and_records": (
            "lgg100_quarantined_closed_loop.py passes online_ik_iterations=30 into "
            "G1SweptPathPreflight. Its legacy damping=0.06, step=0.02 solver stops "
            "before right orientation meets 3 degrees (6.89/6.29 degrees at 30 in "
            "realtime/paused). Stored 250-iteration convergence without joint-limit "
            "hits refutes geometric unreachability. This diagnostic uses the existing "
            "bounded tolerance-aware solver, not a 250-iteration legacy rerun."
        ),
        "configuration": {
            "ik": asdict(bounded_ik_config(source)),
            "registered_position_tolerance_m": 0.005,
            "registered_orientation_tolerance_deg": 3.0,
            "maximum_arm_interpolation_step_rad": 0.01,
            "maximum_finger_interpolation_step_m": 0.001,
            "collision_policy": "unchanged frozen manipulator_contact_violations",
        },
        "modes": modes,
        "all_initial_scenes_match": all(m["initial_scene_matches_saved_evidence"] for m in modes.values()),
        "all_solver_targets_reachable": all(m["solver"]["accepted"] for m in modes.values()),
        "all_full_gates_accepted": all(m["saved_first_target_full_gate_accepted"] for m in modes.values()),
        "full_raw32_replayed": False, "neural_inference_performed": False,
        "physical_simulation_performed": False, "dynamics_steps": 0,
        "reconstruction_limits": {
            "complete_saved_mjdata_available": False,
            "bit_exact_physical_state_replay_claimed": False,
            "parity_scope": "recorded initial FK residuals, world targets and cube positions only",
        },
        "integration_scope": {
            "candidate_project_preflight_only": True,
            "original_neural_closed_loop_modified": False,
            "original_neural_closed_loop_still_uses_legacy_gate": True,
        },
        "observation_freshness": {
            "assessed": False, "maximum_observation_age_ms": 100.0,
            "reason": "Offline saved targets have no live observation timestamp; timing is not freshness.",
        },
        "task_success_demonstrated": False, "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }


def write_diagnostic(output: Path, source: Path | None = None) -> dict:
    """Refuse existing paths before work and atomically refuse races at creation."""
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite diagnostic: {output}")
    source_root = (DEFAULT_SOURCE if source is None else source).expanduser().resolve()
    if output.resolve().is_relative_to(source_root):
        raise ValueError("frozen G1 source is read-only; choose an output outside it")
    report = build_diagnostic(source)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--g1-source", type=Path)
    args = parser.parse_args()
    report = write_diagnostic(args.output, args.g1_source)
    print(json.dumps({
        "output": str(args.output),
        "all_initial_scenes_match": report["all_initial_scenes_match"],
        "all_solver_targets_reachable": report["all_solver_targets_reachable"],
        "all_full_gates_accepted": report["all_full_gates_accepted"],
        "gates": {name: mode["swept_collision_gate"] for name, mode in report["modes"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
