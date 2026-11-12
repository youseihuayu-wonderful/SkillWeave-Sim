"""Real frozen-model dynamics regressions (no policy, fake physics, or scene fixes)."""
from pathlib import Path
import json
import math
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from g1_arm_skill import ACCEPTANCE, ArmRaiseSkill, run_headless
from g1_source import activate_source


class G1ArmSkillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = activate_source()
        from stack_scene import build_model, reset_to_reference_pose
        cls.model = build_model()
        cls.reset_scene = staticmethod(reset_to_reference_pose)
        cls.normal_data, cls.normal_engine = cls.make_engine()
        cls.original_model = {
            name: getattr(cls.model, name).copy()
            for name in ("body_mass", "geom_friction", "geom_contype", "geom_conaffinity",
                         "actuator_gainprm", "actuator_biasprm", "jnt_range")
        }
        assert cls.normal_engine.start()["status"] == "running"
        cls.normal_controls, cls.normal_samples = cls.drive(cls.normal_engine)
        cls.normal = cls.normal_engine.result()

    @classmethod
    def make_engine(cls):
        data = mujoco.MjData(cls.model)
        cls.reset_scene(cls.model, data)
        return data, ArmRaiseSkill(cls.model, data, source=cls.source)

    @classmethod
    def drive(cls, engine, maximum_steps=14000):
        """Observe integration boundaries to catch live teleport/reset shortcuts."""
        from g1_dual_arm_ik import G1DualArmIK
        solver = G1DualArmIK(engine.model, engine.data)
        actuators = np.r_[solver.left["actuator"], solver.right["actuator"]]
        controls = [engine.data.ctrl[actuators].copy()]
        samples = []
        real_step = mujoco.mj_step
        expected = None
        calls = 0

        def measured_step(model, data):
            nonlocal calls
            np.testing.assert_array_equal(data.qpos, expected[0])
            np.testing.assert_array_equal(data.qvel, expected[1])
            controls.append(data.ctrl[actuators].copy())
            calls += 1
            return real_step(model, data)

        with patch.object(mujoco, "mj_step", side_effect=measured_step):
            for _ in range(maximum_steps):
                if not engine.active:
                    break
                expected = engine.data.qpos.copy(), engine.data.qvel.copy()
                before_time, before_calls = engine.data.time, calls
                engine.step()
                assert calls == before_calls + 1, "step must perform exactly one real integration"
                assert abs(engine.data.time - before_time - .002) < 1e-11
                sample = engine.state()
                sample["measured_qvel"] = engine.data.qvel.copy()
                samples.append(sample)
            else:
                raise AssertionError("Skill did not reach a terminal result within its bounded windows")
        return np.asarray(controls), samples

    def test_actual_raise_hold_original_world_return_and_stable_settle(self):
        result = self.normal
        self.assertEqual(result["status"], "succeeded", result["reason"])
        self.assertFalse(result["requires_reset"])
        first_raise = next(i for i, sample in enumerate(self.normal_samples) if sample["phase"] == "raise")
        initial_stable = self.normal_samples[first_raise - 250:first_raise]
        self.assertEqual(len(initial_stable), 250)
        for sample in initial_stable:
            metric = sample["metrics"]
            self.assertLessEqual(metric["robot_max_abs_velocity"], .05)
            self.assertLessEqual(metric["eef_speed_m_s"], .01)
            self.assertLessEqual(metric["eef_angular_speed_rad_s"], .05)
            self.assertLessEqual(max(metric["left_return_error_m"], metric["right_return_error_m"]), .005)
            self.assertLessEqual(metric["return_orientation_error_rad"], math.radians(3))
        hold = [sample for sample in self.normal_samples if sample["phase"] == "hold"]
        self.assertGreaterEqual(len(hold) * .002, 1 - 1e-9)
        self.assertTrue(all(sample["metrics"]["right_elevation_m"] >= .04 for sample in hold))
        self.assertGreater(result["metrics"]["max_right_elevation_m"], .04)
        # Original WORLD EEF anchors, not a replacement home after base drift.
        home = result["home_world_command"]
        actual_left = self.normal_data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_eef")]
        actual_right = self.normal_data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_eef")]
        self.assertLessEqual(np.linalg.norm(actual_left - home[:3]), .005)
        self.assertLessEqual(np.linalg.norm(actual_right - home[7:10]), .005)
        self.assertLessEqual(result["metrics"]["return_orientation_error_rad"], math.radians(3))
        self.assertGreaterEqual(result["metrics"]["settled_for_s"], .5 - 1e-9)
        for sample in self.normal_samples[-250:]:
            self.assertLessEqual(sample["metrics"]["robot_max_abs_velocity"], .05)
            self.assertLessEqual(sample["metrics"]["eef_speed_m_s"], .01)
            self.assertLessEqual(sample["metrics"]["eef_angular_speed_rad_s"], .05)
        self.assertEqual(result["contact_reasons"], [])
        self.assertLessEqual(max(s["metrics"]["tracking_error_m"] for s in self.normal_samples), .05)
        self.assertLessEqual(max(s["metrics"]["orientation_tracking_error_rad"]
                                 for s in self.normal_samples), math.radians(3))
        self.assertGreaterEqual(min(s["metrics"]["pelvis_height_m"] for s in self.normal_samples), .70)

    def test_real_commands_obey_original_joint_envelope_and_model_is_unchanged(self):
        from safety_governor import JointMotionEnvelope
        speeds, accelerations, jerks = JointMotionEnvelope().arrays()
        velocity = np.diff(self.normal_controls, axis=0) / .002
        acceleration = np.diff(np.vstack((np.zeros(14), velocity)), axis=0) / .002
        jerk = np.diff(np.vstack((np.zeros(14), acceleration)), axis=0) / .002
        self.assertLessEqual(float(np.max(np.abs(velocity) / speeds)), 1 + 1e-8)
        self.assertLessEqual(float(np.max(np.abs(acceleration) / accelerations)), 1 + 1e-7)
        self.assertLessEqual(float(np.max(np.abs(jerk) / jerks)), 1 + 1e-5)
        for name, original in self.original_model.items():
            np.testing.assert_array_equal(getattr(self.model, name), original)
        self.assertTrue(self.normal["preflight"]["accepted"])

    def test_mid_motion_stop_cancels_return_but_keeps_integrating_to_measured_rest(self):
        data, engine = self.make_engine()
        engine.start()
        for _ in range(8500):
            state = engine.state()
            if state["phase"] == "raise" and state["metrics"]["right_elevation_m"] > .02:
                break
            self.assertTrue(engine.active, state["reason"])
            engine.step()
        else:
            self.fail("No actual mid-raise motion reached within the bounded initial window")
        before = data.qpos.copy(), data.qvel.copy(), data.time
        self.assertGreater(engine.state()["metrics"]["eef_speed_m_s"], .005)
        state = engine.request_stop()
        self.assertEqual(state["status"], "stopping")
        self.assertTrue(state["active"])
        np.testing.assert_array_equal(data.qpos, before[0])
        np.testing.assert_array_equal(data.qvel, before[1])
        self.assertEqual(data.time, before[2])
        controls, samples = self.drive(engine, maximum_steps=4100)
        report = engine.result()
        self.assertEqual(report["status"], "stopped", report["reason"])
        self.assertTrue(report["metrics"]["stop_settled"])
        self.assertGreater(data.time - before[2], .5)
        self.assertGreater(np.max(np.abs(data.qpos - before[0])), 1e-5)
        self.assertTrue(all(s["phase"] in {"stopping", "terminal"} for s in samples))
        self.assertLessEqual(max(s["metrics"]["right_elevation_m"] for s in samples), .04)
        for sample in samples[-250:]:
            self.assertLessEqual(sample["metrics"]["robot_max_abs_velocity"], .05)
            self.assertLessEqual(sample["metrics"]["eef_speed_m_s"], .01)
        self.assertFalse(report["requires_reset"])
        # A terminal hold must still notice a new real collision if physics continues.
        self._put_cube_in_finger(data)
        engine.step()
        self.assertEqual(engine.state()["status"], "stopping")
        self.assertTrue(engine.requires_reset)
        self.assertIn("phase_invalid_contact", engine.state()["reason"])

    def test_start_and_reset_never_step_or_mutate_live_state(self):
        data, engine = self.make_engine()
        before = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("unexpected integration")):
            state = engine.start()
        self.assertEqual(state["status"], "running")
        for actual, expected in zip((data.qpos, data.qvel, data.ctrl), before[:3], strict=True):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(data.time, before[3])
        # Callers own scene reset; engine.reset merely captures new anchors.
        self.reset_scene(self.model, data)
        qpos, qvel = data.qpos.copy(), data.qvel.copy()
        engine.reset()
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        self.assertEqual(engine.state()["status"], "idle")

    def _put_cube_in_finger(self, data):
        from safety_governor import manipulator_contact_violations
        geoms = [i for i in range(self.model.ngeom)
                 if "left_dex1" in (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                       int(self.model.geom_bodyid[i])) or "")
                 and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])]
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_free")
        address = self.model.jnt_qposadr[joint]
        data.qpos[address:address + 3] = data.geom_xpos[geoms[0]]
        mujoco.mj_forward(self.model, data)
        self.assertIn("dex_cube_outside_manipulation_phase",
                      manipulator_contact_violations(self.model, data, "free_space"))

    def test_unsafe_collision_and_joint_limit_starts_are_rejected_without_motion(self):
        for scenario in ("collision", "joint_limit", "moving"):
            with self.subTest(scenario=scenario):
                data, engine = self.make_engine()
                if scenario == "collision":
                    self._put_cube_in_finger(data)
                elif scenario == "joint_limit":
                    joint = engine.solver.right["joint"][0]
                    data.qpos[self.model.jnt_qposadr[joint]] = self.model.jnt_range[joint, 1] + .01
                    mujoco.mj_forward(self.model, data)
                else:
                    data.qvel[engine.solver.right["dof"][0]] = .2
                before = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time
                self.assertEqual(engine.start()["status"], "rejected")
                self.assertFalse(engine.active)
                self.assertTrue(engine.requires_reset)
                for actual, expected in zip((data.qpos, data.qvel, data.ctrl), before[:3], strict=True):
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual(data.time, before[3])
                self.assertEqual(engine.result()["physics_steps"], 0)
                with patch.object(mujoco, "mj_step", side_effect=AssertionError("rejection must latch")):
                    engine.step()
                self.assertEqual(engine.state()["status"], "rejected")
                self.assertEqual(data.time, before[3])

    def test_real_runtime_collision_cancels_nominal_skill_and_never_reports_stop_success(self):
        data, engine = self.make_engine()
        engine.start()
        engine.step()
        self._put_cube_in_finger(data)
        engine.step()
        state = engine.state()
        self.assertTrue(state["requires_reset"])
        self.assertEqual(state["status"], "stopping")
        self.assertIn("phase_invalid_contact", state["reason"])
        self.drive(engine, maximum_steps=4100)
        result = engine.result()
        self.assertEqual(result["status"], "failed")
        self.assertIn("dex_cube_outside_manipulation_phase", result["contact_reasons"])
        self.assertNotIn("raise", [stage["phase"] for stage in result["stage_history"]])
        self.assertNotEqual(engine.start()["status"], "running")
        before_time, before_qpos = data.time, data.qpos.copy()
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("failure must latch")):
            engine.step()
            engine.step()
        self.assertEqual(engine.state()["status"], "failed")
        self.assertEqual(data.time, before_time)
        np.testing.assert_array_equal(data.qpos, before_qpos)

    def test_actual_orientation_tracking_fault_stops_even_with_small_position_error(self):
        data, engine = self.make_engine()
        engine.start()
        # A real externally disturbed wrist rotation, not a fabricated safety flag.
        data.qpos[engine.solver.right["qpos"][6]] += .08
        mujoco.mj_forward(self.model, data)
        engine.step()
        state = engine.state()
        self.assertLess(state["metrics"]["tracking_error_m"], .05)
        self.assertGreater(state["metrics"]["orientation_tracking_error_rad"], math.radians(3))
        self.assertEqual(state["status"], "stopping")
        self.assertEqual(state["reason"], "eef_orientation_tracking_error_above_hard_maximum")
        self.assertTrue(engine.requires_reset)
        self.drive(engine, maximum_steps=4100)
        self.assertEqual(engine.state()["status"], "failed")

    def test_nonfinite_fault_is_visible_json_safe_and_does_not_trigger_mujoco_reset(self):
        data, engine = self.make_engine()
        engine.start()
        data.qvel[0] = float("nan")
        before_qpos, before_time = data.qpos.copy(), data.time
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            engine.step()
        self.assertEqual(engine.state()["status"], "failed")
        self.assertTrue(engine.requires_reset)
        np.testing.assert_array_equal(data.qpos, before_qpos)
        self.assertEqual(data.time, before_time)
        json.dumps(engine.result(), allow_nan=False)


    def test_headless_cli_writes_actual_stop_evidence_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new" / "nested" / "stop.json"
            process = subprocess.run(
                [sys.executable, "-m", "g1_arm_skill", "--output", str(output), "--stop-after-s", "10.5"],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "stopped")
            self.assertTrue(report["accepted"])
            self.assertTrue(report["metrics"]["stop_settled"])
            self.assertLess(report["motion_started_at_s"], report["stop_requested_at_s"])
            self.assertEqual(report["trace"][-1]["status"], "stopped")
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                run_headless(output, self.source)
            self.assertEqual(output.read_bytes(), original)
            link = Path(directory) / "dangling.json"
            link.symlink_to(Path(directory) / "missing.json")
            with self.assertRaises(FileExistsError):
                run_headless(link, self.source)
            self.assertFalse((Path(directory) / "missing.json").exists())
            with self.assertRaises(ValueError):
                run_headless(self.source / "forbidden_skill_evidence.json", self.source)


if __name__ == "__main__":
    unittest.main()
