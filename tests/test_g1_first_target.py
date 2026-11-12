"""Actual frozen-model kinematic regressions; no neural inference or dynamics."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from g1_first_target import (
    build_diagnostic,
    first_target_hold,
    initialize_scene,
    load_evidence,
    write_diagnostic,
)
from g1_preflight import bounded_ik_config, create_preflight
from g1_source import activate_source


class G1FirstTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = activate_source()
        cls.report = build_diagnostic(cls.source)
        cls.trace, cls.original_reports, _ = load_evidence(cls.source)

    def test_saved_targets_match_initial_fk_and_bounded_solver_reaches(self):
        self.assertTrue(self.report["all_initial_scenes_match"])
        self.assertTrue(self.report["all_solver_targets_reachable"])
        for name, mode in self.report["modes"].items():
            with self.subTest(mode=name):
                self.assertTrue(mode["false_unreachable_resolved"])
                self.assertTrue(mode["source_data_unchanged"])
                self.assertEqual(mode["scene"]["simulation_time_s"], 0.0)
                self.assertAlmostEqual(mode["scene"]["physics_timestep_s"], 0.002)
                solver = mode["solver"]
                self.assertEqual(solver["reason"], "accepted")
                self.assertGreater(solver["total_iterations"], 0)
                self.assertLessEqual(solver["total_iterations"], 60)
                for target in solver["targets"]:
                    self.assertTrue(target["joint_limits_ok"])
                    self.assertTrue(target["reachable"])
                    self.assertLessEqual(target["maximum_position_error_m"], 0.004)
                    self.assertLessEqual(target["maximum_orientation_error_rad"], np.deg2rad(2.5))
                # A distinct collision block must remain a rejection, not be
                # relabelled as solver failure or silently treated as a pass.
                gate = mode["swept_collision_gate"]
                self.assertGreaterEqual(gate["checked_interpolated_configurations"], 1)
                if gate["accepted"]:
                    self.assertEqual(gate["reason"], "accepted")
                    self.assertEqual(gate["checked_targets"], 2)
                    self.assertGreater(gate["checked_interpolated_configurations"], 2)
                    self.assertFalse(gate["collision_reasons"])
                    self.assertTrue(mode["saved_first_target_full_gate_accepted"])
                else:
                    self.assertIn(gate["reason"], ("initial_configuration_collision", "swept_collision"))
                    self.assertTrue(gate["collision_reasons"])
                    self.assertIsNotNone(gate["rejection_substep"])
                    self.assertFalse(mode["saved_first_target_full_gate_accepted"])

    def test_output_inside_frozen_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            output = source / "new.json"
            with self.assertRaisesRegex(ValueError, "read-only"):
                write_diagnostic(output, source)
            self.assertFalse(output.exists())


    def test_unreachable_target_remains_rejected_without_source_mutation(self):
        from g1_fast_sequential_ik import solve_sequential_ik

        model, data = initialize_scene(self.source)
        action = np.asarray(self.trace["modes"]["realtime"]["raw_first_action"]).copy()
        action[:3] = [10.0, 10.0, 10.0]
        action[7:10] = [10.0, -10.0, 10.0]
        before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time)
        result = solve_sequential_ik(model, data, action[None, :], bounded_ik_config(self.source))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "ik_tolerance_not_reached")
        self.assertEqual(result.total_iterations, 60)
        self.assertFalse(result.targets[0].reachable)
        self.assertGreater(result.targets[0].maximum_position_error_m, 1.0)
        gate = create_preflight(model, self.source).check(data, first_target_hold(action), phase="free_space")
        self.assertFalse(gate.accepted)
        self.assertIn(gate.reason, ("ik_tolerance_not_reached", "initial_configuration_collision"))
        for actual, expected in zip((data.qpos, data.qvel, data.ctrl), before[:3], strict=True):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(data.time, before[3])

    def test_real_cube_finger_collision_still_blocks_preflight(self):
        import mujoco
        from safety_governor import manipulator_contact_violations

        model, data = initialize_scene(self.source)
        # Put the real red cube inside a collision-enabled left finger geom.
        # The gate must reject actual MuJoCo contacts, not a manufactured flag.
        candidates = [index for index in range(model.ngeom)
                      if "left_dex1" in (mujoco.mj_id2name(
                          model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index])
                      ) or "") and (model.geom_contype[index] or model.geom_conaffinity[index])]
        self.assertTrue(candidates, "frozen model must expose Dex1 collision geometry")
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_free")
        qpos = int(model.jnt_qposadr[joint])
        data.qpos[qpos:qpos + 3] = data.geom_xpos[candidates[0]]
        mujoco.mj_forward(model, data)
        violations = manipulator_contact_violations(model, data, "free_space")
        self.assertIn("dex_cube_outside_manipulation_phase", violations)
        before = data.qpos.copy()
        action = self.trace["modes"]["realtime"]["raw_first_action"]
        gate = create_preflight(model, self.source).check(data, first_target_hold(action), phase="free_space")
        self.assertFalse(gate.accepted)
        self.assertEqual(gate.reason, "initial_configuration_collision")
        self.assertIn("dex_cube_outside_manipulation_phase", gate.collision_reasons)
        self.assertEqual(gate.total_ik_iterations, 0)
        np.testing.assert_array_equal(data.qpos, before)
        self.assertEqual(data.time, 0.0)

    def test_existing_record_and_dangling_symlink_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_bytes(b"original evidence\n")
            with self.assertRaises(FileExistsError):
                write_diagnostic(output, self.source)
            self.assertEqual(output.read_bytes(), b"original evidence\n")
            link = Path(directory) / "dangling.json"
            link.symlink_to(Path(directory) / "absent.json")
            with self.assertRaises(FileExistsError):
                write_diagnostic(link, self.source)
            self.assertFalse(link.resolve().exists())


if __name__ == "__main__":
    unittest.main()
