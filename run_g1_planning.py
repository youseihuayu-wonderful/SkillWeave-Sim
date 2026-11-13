"""Run real subscription planning and reviewed, simulation-only G1 execution."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import mujoco

from g1_arm_skill import ArmRaiseSkill, CONFIG
from g1_planning import G1PlanningSession, make_observation
from g1_source import activate_source
from planner import PlannerError

ROOT = Path(__file__).resolve().parent
DEFAULT_TASK = "请抬起右臂，保持一秒，然后回到原来的位置。"


async def run_g1_planning(
    task: str,
    output: Path,
    *,
    execute: bool = False,
    stop_after_s: float | None = None,
    source: Path | None = None,
    root: Path = ROOT,
) -> dict:
    """Plan only by default; --execute explicitly authorizes this one simulated skill."""
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite planner evidence: {output}")
    if not isinstance(task, str) or not 1 <= len(task.strip()) <= 1000:
        raise ValueError("Task must contain 1–1000 nonblank characters")
    maximum_nominal_s = sum(CONFIG[key] for key in (
        "initial_hold_s", "maximum_initial_settle_s", "raise_s", "hold_s", "return_s",
    ))
    if stop_after_s is not None:
        if not execute or not math.isfinite(stop_after_s) or not 0 <= stop_after_s < maximum_nominal_s:
            raise ValueError(f"stop-after-s requires --execute and a finite value in [0, {maximum_nominal_s})")
    source = activate_source(source)
    if output.resolve().is_relative_to(source):
        raise ValueError("Evidence output must be outside the frozen G1 source")

    from stack_scene import build_model, reset_to_reference_pose

    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    skill = ArmRaiseSkill(model, data, source=source)

    def observe():
        return make_observation(model, data, skill, scene_revision=0)

    initial = observe()
    session = G1PlanningSession(root, skill, observe)
    approved = False
    stop_requested_at_s = None
    try:
        session.start(task.strip())
        while session.locks_scene:
            state = session.state()
            if state["status"] == "ready":
                if not execute:
                    break
                if approved:
                    raise RuntimeError("The single-capability runner cannot approve a second invocation")
                session.execute(state["decision_id"])
                approved = True
            elif state["status"] in {"executing", "stopping"}:
                for _ in range(10):
                    if not skill.active:
                        session.advance()
                        break
                    if (stop_after_s is not None and stop_requested_at_s is None
                            and skill.state()["elapsed_s"] >= stop_after_s):
                        session.cancel()
                        stop_requested_at_s = skill.state()["elapsed_s"]
                    try:
                        skill.step()
                    except Exception as exc:
                        # Preserve the raw engine state; this is not proof of a safe stop.
                        session.fail_execution(f"{type(exc).__name__}: {exc}")
                        break
                    session.advance()
                    if not skill.active:
                        break
                await asyncio.sleep(0)
            else:
                # Deliberately paused simulation while the real provider is running.
                await asyncio.sleep(0.02)
        final = observe()
        terminal_status = session.state()["status"]
        report = session.report()
        physics_steps = skill.result()["physics_steps"]
        if stop_after_s is not None:
            scenario_passed = terminal_status == "stopped" and stop_requested_at_s is not None
        elif execute:
            scenario_passed = bool(report["success"])
        else:
            scenario_passed = terminal_status in {"ready", "unsupported"}
        report["runner"] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "authorization": "explicit_execute_flag" if execute else "proposal_only_no_execution",
            "terminal_status": terminal_status,
            "proposal_approved": approved,
            "stop_after_s": stop_after_s,
            "stop_requested_at_s": stop_requested_at_s,
            "simulation_timestep_s": float(model.opt.timestep),
            "initial_time_s": initial["simulation_time_s"],
            "final_time_s": final["simulation_time_s"],
            "initial_state_fingerprint": initial["state_fingerprint"],
            "final_state_fingerprint": final["state_fingerprint"],
            "physics_steps": physics_steps,
            "execution_performed": physics_steps > 0,
            "provider_wait_physics": "paused_simulation_only_not_hardware_wait_validation",
            "scenario_passed": scenario_passed,
        }
    finally:
        await session.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Approve the actual current plan for one simulated skill")
    parser.add_argument("--stop-after-s", type=float, help="Explicit cancellation scenario; not a model motion parameter")
    parser.add_argument("--g1-source", type=Path)
    args = parser.parse_args()
    try:
        report = asyncio.run(run_g1_planning(
            args.task, args.output, execute=args.execute,
            stop_after_s=args.stop_after_s, source=args.g1_source,
        ))
    except (OSError, ValueError, RuntimeError, PlannerError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    driver = report["runner"]
    print(json.dumps({
        "output": str(args.output), "status": driver["terminal_status"],
        "success": report["success"], "execution_performed": driver["execution_performed"],
        "physics_steps": driver["physics_steps"], "scenario_passed": driver["scenario_passed"],
    }, ensure_ascii=False))
    raise SystemExit(0 if driver["scenario_passed"] else 1)


if __name__ == "__main__":
    main()
