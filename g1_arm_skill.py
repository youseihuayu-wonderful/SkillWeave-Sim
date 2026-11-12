"""Simulation-only, world-anchored G1 scripted raise/hold/return skill.

Only ctrl is written on the live data. FK and swept preflight use separate data;
all movement and stopping are measured after real MuJoCo integration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np

from g1_source import SOURCE_REVISION, activate_source


ACCEPTANCE = {
    "minimum_raise_m": 0.04,
    "minimum_hold_s": 1.0,
    "return_position_tolerance_m": 0.005,
    "return_orientation_tolerance_rad": math.radians(3),
    "hard_tracking_error_m": 0.05,
    "hard_orientation_tracking_error_rad": math.radians(3),
    "minimum_pelvis_height_m": 0.70,
    "maximum_base_xy_drift_m": 0.08,
    "maximum_base_tilt_rad": 0.15,
    "maximum_robot_abs_velocity": 0.05,
    "maximum_eef_speed_m_s": 0.01,
    "maximum_eef_angular_speed_rad_s": 0.05,
    "sustained_settle_s": 0.5,
}
CONFIG = {
    "initial_hold_s": 2.0, "raise_s": 3.0, "hold_s": 1.0,
    "return_s": 3.0, "maximum_settle_s": 8.0,
    "maximum_initial_settle_s": 8.0,
    "maximum_stop_s": 8.0, "shoulder_raise_delta_rad": -0.25,
    "gravity_position_offset_bound_rad": 0.12,
    "feedback_joint_delta_bound_rad": 0.20,
    "feedback_velocity_damping_s": 0.04,
    "feedback_damping": 0.015, "trace_interval_s": 0.05,
}
FLAGS = {
    "executor_kind": "scripted_skill", "planner_connected": False,
    "policy_connected": False, "hardware_execution_enabled": False,
}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class ArmRaiseSkill:
    """One physical scripted run; explicit scene reset is required before reuse."""

    def __init__(self, model, data, *, source: Path | None = None):
        # Source validation/import activation is deliberately confined to startup.
        self.source = activate_source(source)
        from dex1_gripper import Dex1Controller
        from g1_dual_arm_ik import G1DualArmIK, orientation_error
        from g1_preflight import create_preflight
        from safety_governor import (
            JerkLimitedActionFilter, JerkLimitedJointFilter,
            manipulator_contact_violations,
        )
        if abs(model.opt.timestep - 0.002) > 1e-12:
            raise ValueError("The frozen G1 timestep must be 0.002 s")
        self.model, self.data = model, data
        self.dt = float(model.opt.timestep)
        self.solver = G1DualArmIK(model, data)
        self.gripper = Dex1Controller(model)
        self._orientation_error = orientation_error
        self._contacts = manipulator_contact_violations
        self._preflight_gate = create_preflight(model, self.source)
        self._joint_filter = JerkLimitedJointFilter()
        self._action_filter = JerkLimitedActionFilter()
        self._infos = (self.solver.left, self.solver.right)
        self._sites = (self.solver.left_site, self.solver.right_site)
        self._arm_qpos = np.concatenate([info["qpos"] for info in self._infos])
        self._arm_act = np.concatenate([info["actuator"] for info in self._infos])
        self._arm_joint = np.concatenate([info["joint"] for info in self._infos])
        self._lower = model.jnt_range[self._arm_joint, 0].copy()
        self._upper = model.jnt_range[self._arm_joint, 1].copy()
        self._pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._robot_dofs = np.flatnonzero(
            model.body_rootid[model.dof_bodyid] == model.body_rootid[self._pelvis]
        )
        self._limited = np.flatnonzero(model.jnt_limited)
        self._limited_qpos = model.jnt_qposadr[self._limited]
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))
        self._regularizer = np.eye(6) * CONFIG["feedback_damping"] ** 2
        self._scratch = mujoco.MjData(model)
        self._scratch_solver = G1DualArmIK(model, self._scratch)
        self.reset()

    @property
    def active(self):
        return self._status in {"running", "stopping"}

    @property
    def requires_reset(self):
        return self._requires_reset

    def _world_command(self, solver=None):
        solver = self.solver if solver is None else solver
        left, right = solver.pose("left"), solver.pose("right")
        return np.r_[left[0], left[1], right[0], right[1], self._grippers]

    def reset(self):
        """Capture caller-reset scene anchors; never reset or write live qpos/qvel."""
        mujoco.mj_forward(self.model, self.data)
        self._grippers = self.gripper.motor_states(self.data).copy()
        self._reference_qpos = self.data.qpos.copy()
        self._base_ctrl = self.data.ctrl.copy()
        self._home = self._world_command()
        self._target = self._home.copy()
        self._stop_target = self._home.copy()
        self._base_position = self.data.xpos[self._pelvis].copy()
        self._base_rotation = self.data.xmat[self._pelvis].reshape(3, 3).copy()
        self._status, self._phase, self._reason = "idle", "idle", None
        self._requires_reset = False
        self._physics_steps = 0
        self._started_time = float(self.data.time)
        self._stop_started = None
        self._settled_for = 0.0
        self._hold_above = 0.0
        self._max_hold_above = 0.0
        self._max_elevation = 0.0
        self._stop_settled = False
        self._motion_started = None
        self._initial_settled_for = 0.0
        self._preflight = None
        self._trace = []
        self._history = []
        self._contact_reasons = set()
        self._maxima = {}
        self._joint_filter.reset(self._base_ctrl[self._arm_act])
        if np.isfinite(self._home).all():
            self._action_filter.reset(self._home)
        self._metrics = self._measure()

    def _elapsed(self):
        return float(self.data.time) - self._started_time

    def _stage(self, phase):
        if phase != self._phase:
            self._phase = phase
            self._history.append({"phase": phase, "elapsed_s": self._elapsed(),
                                  "physics_step": self._physics_steps})

    def _reference_command(self, fraction):
        # This is explicitly scratch FK, never a live-state assignment.
        self._scratch.qpos[:] = self._reference_qpos
        self._scratch.qpos[self.solver.right["qpos"][0]] += (
            CONFIG["shoulder_raise_delta_rad"] * fraction
        )
        mujoco.mj_fwdPosition(self.model, self._scratch)
        return self._world_command(self._scratch_solver)

    def _run_preflight(self):
        from action_schema import (
            EEFActionChunk, quaternion_multiply_wxyz, quaternion_to_matrix_wxyz,
        )
        world = [self._reference_command(float(x)) for x in np.r_[
            np.linspace(0, 1, 61), np.linspace(1, 0, 61)[1:]
        ]]
        pelvis_position = self.data.xpos[self._pelvis]
        pelvis_quat = self.data.xquat[self._pelvis]
        inverse_quat = pelvis_quat * np.array([1, -1, -1, -1])
        rotation = quaternion_to_matrix_wxyz(pelvis_quat)
        actions = []
        for command in world:
            action = command.copy()
            for offset in (0, 7):
                action[offset:offset + 3] = rotation.T @ (
                    command[offset:offset + 3] - pelvis_position
                )
                local_quat = quaternion_multiply_wxyz(
                    inverse_quat, command[offset + 3:offset + 7]
                )
                action[offset + 3:offset + 7] = local_quat[[1, 2, 3, 0]]
            actions.append(action)
        chunk = EEFActionChunk(np.linspace(0, 7, len(actions)), np.array(actions))
        return asdict(self._preflight_gate.check(self.data, chunk, phase="free_space"))

    def _invalid_state(self):
        if not all(np.isfinite(value).all() for value in (
            self.data.qpos, self.data.qvel, self.data.ctrl,
        )):
            return "non_finite_simulation_state"
        values = self.data.qpos[self._limited_qpos]
        ranges = self.model.jnt_range[self._limited]
        if np.any(values < ranges[:, 0] - 1e-5) or np.any(values > ranges[:, 1] + 1e-5):
            return "actual_joint_limit_violation"
        violations = self._contacts(self.model, self.data, "free_space")
        self._contact_reasons.update(violations)
        if violations:
            return "phase_invalid_contact:" + ",".join(violations)
        metrics = self._measure()
        if metrics["pelvis_height_m"] < ACCEPTANCE["minimum_pelvis_height_m"]:
            return "pelvis_height_below_hard_minimum"
        if metrics["base_xy_drift_m"] > ACCEPTANCE["maximum_base_xy_drift_m"]:
            return "base_xy_drift_above_hard_maximum"
        if metrics["base_tilt_rad"] > ACCEPTANCE["maximum_base_tilt_rad"]:
            return "base_tilt_above_hard_maximum"
        if metrics["tracking_error_m"] > ACCEPTANCE["hard_tracking_error_m"]:
            return "eef_tracking_error_above_hard_maximum"
        if metrics["orientation_tracking_error_rad"] > ACCEPTANCE["hard_orientation_tracking_error_rad"]:
            return "eef_orientation_tracking_error_above_hard_maximum"
        return None

    def start(self):
        if self.active:
            return {**self.state(), "reason": "skill_already_active"}
        if self._status != "idle" or self._requires_reset:
            return {**self.state(), "reason": "explicit_scene_reset_required"}
        reason = self._invalid_state()
        if reason is None:
            self._metrics = self._measure()
        if reason is None and self._metrics["robot_max_abs_velocity"] > ACCEPTANCE["maximum_robot_abs_velocity"]:
            reason = "start_state_not_stationary"
        if reason is None:
            self._preflight = self._run_preflight()
            if not self._preflight["accepted"]:
                reason = "preflight_rejected:" + self._preflight["reason"]
        if reason:
            self._status, self._reason = "rejected", reason
            self._requires_reset = True
            return self.state()
        self._started_time = float(self.data.time)
        self._status = "running"
        self._stage("initial_hold")
        return self.state()

    def _begin_stop(self, reason=None):
        self._stop_target = self._world_command()
        self._stop_started = self._elapsed()
        self._settled_for = 0.0
        self._status = "stopping"
        self._stage("stopping")
        if reason is not None:
            self._reason = reason
            self._requires_reset = True

    def request_stop(self):
        if self._status == "running":
            self._begin_stop()
        return self.state()

    @staticmethod
    def _smooth(fraction):
        u = float(np.clip(fraction, 0, 1))
        return u ** 3 * (10 - 15 * u + 6 * u * u)

    def _desired(self):
        if self._status == "stopping":
            return self._stop_target
        if self._status != "running":
            return self._target
        if self._motion_started is None:
            self._stage("initial_hold")
            return self._home
        elapsed = round((self._elapsed() - self._motion_started) / self.dt) * self.dt
        raised = CONFIG["raise_s"]
        held = raised + CONFIG["hold_s"]
        returned = held + CONFIG["return_s"]
        if elapsed < raised:
            self._stage("raise")
            fraction = self._smooth(elapsed / CONFIG["raise_s"])
        elif elapsed < held:
            self._stage("hold")
            fraction = 1.0
        elif elapsed < returned:
            self._stage("return")
            fraction = 1 - self._smooth((elapsed - held) / CONFIG["return_s"])
        else:
            self._stage("settle")
            fraction = 0.0
        return self._reference_command(fraction)

    def _apply_feedback(self, desired):
        filtered, eef_telemetry = self._action_filter.step(desired, self.dt)
        desired_joints = np.empty(14)
        for index, (info, site) in enumerate(zip(self._infos, self._sites, strict=True)):
            offset = index * 7
            position, quaternion = self.solver.pose(("left", "right")[index])
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, site)
            jacobian = np.vstack((self._jp[:, info["dof"]], self._jr[:, info["dof"]]))
            error = np.r_[filtered[offset:offset + 3] - position,
                          self._orientation_error(filtered[offset + 3:offset + 7], quaternion)]
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + self._regularizer, error
            )
            delta = np.clip(delta, -CONFIG["feedback_joint_delta_bound_rad"],
                            CONFIG["feedback_joint_delta_bound_rad"])
            # Invert the unchanged position servo's static gravity error. This
            # is a bounded ctrl offset, NOT applied-force injection or gain edits.
            gravity_offset = np.clip(
                self.data.qfrc_bias[info["dof"]]
                / self.model.actuator_gainprm[info["actuator"], 0],
                -CONFIG["gravity_position_offset_bound_rad"],
                CONFIG["gravity_position_offset_bound_rad"],
            )
            desired_joints[offset:offset + 7] = (
                self.data.qpos[info["qpos"]] + delta + gravity_offset
                - CONFIG["feedback_velocity_damping_s"] * self.data.qvel[info["dof"]]
            )
        command, joint_telemetry = self._joint_filter.step(
            desired_joints, self.dt, self._lower, self._upper
        )
        self.data.ctrl[:] = self._base_ctrl
        self.data.ctrl[self._arm_act] = command
        self._target = filtered
        telemetry = asdict(joint_telemetry)
        telemetry.update(asdict(eef_telemetry))
        for key, value in telemetry.items():
            self._maxima[key] = max(self._maxima.get(key, 0.0), value)
        if any(telemetry[key] > 1 + 1e-6 for key in (
            "max_speed_ratio", "max_acceleration_ratio", "max_jerk_ratio",
        )):
            return "filtered_joint_command_limit_violation"
        envelope = self._action_filter.envelope
        if (eef_telemetry.max_translation_speed_m_s > envelope.max_eef_speed_m_s + 1e-6
                or eef_telemetry.max_translation_acceleration_m_s2 > envelope.max_eef_acceleration_m_s2 + 1e-5
                or eef_telemetry.max_translation_jerk_m_s3 > envelope.max_eef_jerk_m_s3 + 1e-3
                or eef_telemetry.max_angular_speed_rad_s > envelope.max_angular_speed_rad_s + 1e-6
                or eef_telemetry.max_gripper_speed_rad_s > envelope.max_gripper_speed_rad_s + 1e-5):
            return "filtered_eef_command_limit_violation"
        return None

    def _measure(self):
        actual = self._world_command()
        speed, angular = 0.0, 0.0
        for site in self._sites:
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, site)
            speed = max(speed, float(np.linalg.norm(self._jp @ self.data.qvel)))
            angular = max(angular, float(np.linalg.norm(self._jr @ self.data.qvel)))
        rotation = self.data.xmat[self._pelvis].reshape(3, 3)
        tilt = math.acos(float(np.clip(np.dot(rotation[:, 2], self._base_rotation[:, 2]), -1, 1)))
        return {
            "right_elevation_m": float(actual[9] - self._home[9]),
            "max_right_elevation_m": self._max_elevation,
            "tracking_error_m": float(max(np.linalg.norm(actual[:3] - self._target[:3]),
                                             np.linalg.norm(actual[7:10] - self._target[7:10]))),
            "orientation_tracking_error_rad": float(max(
                np.linalg.norm(self._orientation_error(self._target[3:7], actual[3:7])),
                np.linalg.norm(self._orientation_error(self._target[10:14], actual[10:14])),
            )),
            "left_return_error_m": float(np.linalg.norm(actual[:3] - self._home[:3])),
            "right_return_error_m": float(np.linalg.norm(actual[7:10] - self._home[7:10])),
            "return_orientation_error_rad": float(max(
                np.linalg.norm(self._orientation_error(self._home[3:7], actual[3:7])),
                np.linalg.norm(self._orientation_error(self._home[10:14], actual[10:14])),
            )),
            "robot_max_abs_velocity": float(np.max(np.abs(self.data.qvel[self._robot_dofs]))),
            "eef_speed_m_s": speed, "eef_angular_speed_rad_s": angular,
            "base_xy_drift_m": float(np.linalg.norm(self.data.xpos[self._pelvis, :2] - self._base_position[:2])),
            "pelvis_height_m": float(self.data.xpos[self._pelvis, 2]),
            "base_tilt_rad": tilt, "hold_above_minimum_s": self._max_hold_above,
            "settled_for_s": self._settled_for, "stop_settled": self._stop_settled,
            "initial_settled_for_s": self._initial_settled_for,
            "left_eef_world_m": actual[:3].tolist(), "right_eef_world_m": actual[7:10].tolist(),
        }

    def _low_velocity(self):
        m = self._metrics
        return (m["robot_max_abs_velocity"] <= ACCEPTANCE["maximum_robot_abs_velocity"]
                and m["eef_speed_m_s"] <= ACCEPTANCE["maximum_eef_speed_m_s"]
                and m["eef_angular_speed_rad_s"] <= ACCEPTANCE["maximum_eef_angular_speed_rad_s"])

    def _at_home(self):
        m = self._metrics
        return (max(m["left_return_error_m"], m["right_return_error_m"])
                <= ACCEPTANCE["return_position_tolerance_m"]
                and m["return_orientation_error_rad"] <= ACCEPTANCE["return_orientation_tolerance_rad"])

    def _finish(self, status, reason=None):
        self._status = status
        if reason is not None:
            self._reason = reason
        self._requires_reset |= status == "failed"
        self._stage("terminal")

    def step(self):
        """Advance one real step, except latched rejected/failed or nonfinite data.

        Fault/rejection never resumes integration until caller scene-reset and
        reset(). A running stop still integrates until measured rest or timeout.
        """
        # Invalid numerical input cannot safely be integrated: MuJoCo may reset
        # divergent state internally. Fail visibly instead of manufacturing motion.
        if not all(np.isfinite(value).all() for value in (self.data.qpos, self.data.qvel, self.data.ctrl)):
            if self._status not in {"failed", "rejected"}:
                self._finish("failed", "non_finite_simulation_state")
            raise RuntimeError("Non-finite simulation state; explicit scene reset required")
        if self._status in {"failed", "rejected"}:
            return
        violation = self._invalid_state()
        if violation and self._status != "stopping":
            self._begin_stop(violation)
        command_failure = self._apply_feedback(self._desired())
        if command_failure:
            if self._status != "stopping":
                self._begin_stop(command_failure)
            else:
                self._requires_reset = True
                self._reason = self._reason or command_failure
        mujoco.mj_step(self.model, self.data)
        self._physics_steps += 1
        mujoco.mj_forward(self.model, self.data)
        self._metrics = self._measure()
        for key in ("tracking_error_m", "orientation_tracking_error_rad",
                    "robot_max_abs_velocity", "eef_speed_m_s",
                    "eef_angular_speed_rad_s", "base_xy_drift_m", "base_tilt_rad"):
            self._maxima[key] = max(self._maxima.get(key, 0.0), self._metrics[key])
        self._max_elevation = max(self._max_elevation, self._metrics["right_elevation_m"])
        violation = self._invalid_state()
        if violation:
            if self._status != "stopping":
                self._begin_stop(violation)
            elif not self._requires_reset:
                self._requires_reset, self._reason = True, violation
        if self._status == "running":
            if self._phase == "initial_hold":
                self._initial_settled_for = (self._initial_settled_for + self.dt
                                             if self._low_velocity() and self._at_home() else 0.0)
                if (self._elapsed() >= CONFIG["initial_hold_s"]
                        and self._initial_settled_for + 1e-9 >= ACCEPTANCE["sustained_settle_s"]):
                    self._motion_started = self._elapsed()
                elif self._elapsed() > CONFIG["initial_hold_s"] + CONFIG["maximum_initial_settle_s"]:
                    self._begin_stop("initial_world_home_did_not_settle")
            if self._phase == "hold" and self._metrics["right_elevation_m"] >= ACCEPTANCE["minimum_raise_m"]:
                self._hold_above += self.dt
                self._max_hold_above = max(self._max_hold_above, self._hold_above)
            elif self._phase == "hold":
                self._hold_above = 0.0
            if self._phase == "settle":
                self._settled_for = self._settled_for + self.dt if self._low_velocity() and self._at_home() else 0.0
                if self._settled_for + 1e-9 >= ACCEPTANCE["sustained_settle_s"]:
                    if self._max_hold_above + 1e-9 >= ACCEPTANCE["minimum_hold_s"]:
                        self._finish("succeeded")
                    else:
                        self._finish("failed", "actual_raise_hold_acceptance_not_met")
                elif self._elapsed() - self._motion_started > sum(CONFIG[key] for key in (
                    "raise_s", "hold_s", "return_s",
                )) + CONFIG["maximum_settle_s"]:
                    self._begin_stop("final_return_did_not_settle")
        elif self._status == "stopping":
            self._settled_for = self._settled_for + self.dt if self._low_velocity() else 0.0
            if self._settled_for + 1e-9 >= ACCEPTANCE["sustained_settle_s"]:
                self._stop_settled = True
                self._finish("failed" if self._requires_reset else "stopped")
            elif self._elapsed() - self._stop_started > CONFIG["maximum_stop_s"]:
                self._finish("failed", self._reason or "physical_stop_did_not_settle")
        self._metrics.update(max_right_elevation_m=self._max_elevation,
                             hold_above_minimum_s=self._max_hold_above,
                             initial_settled_for_s=self._initial_settled_for,
                             settled_for_s=self._settled_for, stop_settled=self._stop_settled)
        if self._physics_steps % round(CONFIG["trace_interval_s"] / self.dt) == 0 or not self.active:
            self._trace.append({"elapsed_s": self._elapsed(), "phase": self._phase,
                                "status": self._status, **self._metrics})

    def state(self):
        return _json_safe({**FLAGS, "status": self._status, "phase": self._phase,
                           "active": self.active, "requires_reset": self.requires_reset,
                           "elapsed_s": self._elapsed(), "reason": self._reason,
                           "metrics": self._metrics})

    def result(self):
        return _json_safe({**self.state(), "acceptance": dict(ACCEPTANCE), "config": dict(CONFIG),
                           "source": {"path": str(self.source), "revision": SOURCE_REVISION},
                           "physics_steps": self._physics_steps, "preflight": self._preflight,
                           "motion_started_at_s": self._motion_started,
                           "terminal_step_contract": "failed/rejected latch without integration; nonfinite raises without integration; succeeded/stopped keep bounded hold",
                           "home_world_command": self._home, "maxima": self._maxima,
                           "contact_reasons": sorted(self._contact_reasons),
                           "stop_result": {
                               "began_at_s": self._stop_started,
                               "settled": self._stop_settled,
                               "timed_out": self._stop_started is not None
                               and not self._stop_settled and self._status == "failed"
                               and self._elapsed() - self._stop_started > CONFIG["maximum_stop_s"],
                           },
                           "stage_history": self._history, "trace": self._trace})


def run_headless(output: Path, source: Path | None = None, stop_after_s: float | None = None):
    """Create immutable evidence with O_EXCL; no overwrite or symlink following."""
    output = Path(output).absolute()
    source = activate_source(source)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Evidence output must be outside the frozen source")
    maximum_nominal_s = sum(CONFIG[key] for key in (
        "initial_hold_s", "maximum_initial_settle_s", "raise_s", "hold_s", "return_s",
    ))
    if stop_after_s is not None and (not math.isfinite(stop_after_s) or not 0 <= stop_after_s < maximum_nominal_s):
        raise ValueError(f"stop-after-s must be finite and in [0, {maximum_nominal_s})")
    from stack_scene import build_model, reset_to_reference_pose
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    engine = ArmRaiseSkill(model, data, source=source)
    engine.start()
    requested = False
    requested_at_s = None
    while engine.active:
        if stop_after_s is not None and not requested and engine.state()["elapsed_s"] >= stop_after_s:
            engine.request_stop()
            requested_at_s = engine.state()["elapsed_s"]
            requested = True
        try:
            engine.step()
        except RuntimeError:
            if engine.state()["status"] != "failed":
                raise
            break
    report = engine.result()
    report["explicit_stop_scenario"] = stop_after_s is not None
    report["stop_after_s"] = stop_after_s
    report["stop_requested_at_s"] = requested_at_s
    report["accepted"] = report["status"] == ("stopped" if stop_after_s is not None else "succeeded")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--g1-source", type=Path)
    parser.add_argument("--stop-after-s", type=float)
    args = parser.parse_args()
    try:
        report = run_headless(args.output, args.g1_source, args.stop_after_s)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps({key: report[key] for key in ("status", "accepted", "reason", "physics_steps")}))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
