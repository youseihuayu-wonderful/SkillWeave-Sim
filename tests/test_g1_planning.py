"""Consumer safety/race contracts using declared provider and executor doubles.

These deterministic doubles are not subscription, physical-execution or wiring
evidence. The observation tests alone read the real frozen MuJoCo scene.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from g1_arm_skill import ACCEPTANCE, ArmRaiseSkill
from g1_planning import G1PlanningSession, SKILL_ID, make_observation
from g1_source import SOURCE_REVISION, activate_source
from planner import PlannerConfig, PlannerError, StructuredResult


ROOT = Path(__file__).resolve().parents[1]


def execution_decision():
    return {
        "decision": "execute", "skill": SKILL_ID, "parameters": {},
        "summary": "执行唯一的右臂复合脚本", "reason": "整个请求在固定技能范围内",
        "unsupported_requirements": [],
    }


def unsupported_decision():
    return {
        "decision": "unsupported", "skill": None, "parameters": {},
        "summary": "不能完整执行请求", "reason": "没有拿手机拍照能力",
        "unsupported_requirements": ["拿手机拍照"],
    }


def assessment(outcome):
    return {"outcome": outcome, "summary": "记录真实终态", "reason": "根据本次真实执行结果进行评估"}


class ProviderDouble:
    """Test-only explicit responses, with controllable cancellation/reaping races."""
    def __init__(self):
        self.calls = []
        self.entered = asyncio.Queue()
        self.ignore_cancellation = False
        self.cleanup_gate = None
        self.cleanup_started = asyncio.Event()
        self.reaped = asyncio.Event()

    async def __call__(self, config, *, instructions, schema, payload):
        future = asyncio.get_running_loop().create_future()
        record = {"payload": deepcopy(payload), "future": future}
        self.calls.append(record)
        await self.entered.put(record)
        try:
            try:
                response = await asyncio.shield(future)
            except asyncio.CancelledError:
                if not self.ignore_cancellation:
                    raise
                response = await future
            if isinstance(response, Exception):
                raise response
            return StructuredResult(response, "declared-provider-double", f"double-thread-{len(self.calls)}", .01, 0)
        finally:
            self.cleanup_started.set()
            if self.cleanup_gate is not None:
                await self.cleanup_gate.wait()
            self.reaped.set()

    async def next_call(self):
        return await asyncio.wait_for(self.entered.get(), 1)

    async def reply(self, record, payload):
        record["future"].set_result(payload)
        for _ in range(5):
            await asyncio.sleep(0)


class SkillDouble:
    """Explicit executor state-machine double; it never simulates real physics."""
    def __init__(self):
        self.starts = 0
        self.stops = 0
        self.start_outcome = "running"
        self.start_error = None
        self._state = {
            "status": "idle", "phase": "idle", "active": False, "requires_reset": False,
            "elapsed_s": 0.0, "reason": None,
            "metrics": {"robot_max_abs_velocity": 0.0, "eef_speed_m_s": 0.0,
                        "eef_angular_speed_rad_s": 0.0, "settled_for_s": 0.0},
        }
        self.trace = [{"test_only_trace": index} for index in range(1000)]

    def state(self):
        return deepcopy(self._state)

    def start(self):
        self.starts += 1
        if self.start_error is not None:
            self._state.update(status="running", active=True)
            raise self.start_error
        if self.start_outcome == "running":
            self._state.update(status="running", phase="initial_hold", active=True)
        else:
            self.finish(self.start_outcome)
        return self.state()

    def request_stop(self):
        self.stops += 1
        self._state.update(status="stopping", phase="stopping", active=True)
        return self.state()

    def finish(self, outcome):
        self._state.update(status=outcome, phase="terminal", active=False,
                           requires_reset=outcome in ("failed", "rejected"),
                           reason=None if outcome == "succeeded" else f"double-{outcome}", elapsed_s=9.5)
        self._state["metrics"].update(settled_for_s=.5, stop_settled=outcome == "stopped")

    def result(self):
        return {
            **self.state(), "acceptance": dict(ACCEPTANCE), "physics_steps": 4750,
            "source": {"path": "/private/test-only-source", "revision": SOURCE_REVISION},
            "trace": self.trace, "home_world_command": [0] * 16,
            "contact_reasons": [], "maxima": {},
            "preflight": {"accepted": self._state["status"] != "rejected", "reason": "test-only"},
            "stop_result": {"settled": self._state["status"] == "stopped", "timed_out": False},
            "stage_history": [{"phase": self._state["phase"], "elapsed_s": 9.5, "physics_step": 4750}],
        }


class SceneDouble:
    """Declared state binding, not a sensor or physics implementation."""
    def __init__(self, skill):
        self.skill = skill
        self.revision = 0
        self.physical_state = "test-only-unchanged-state"
        self.error = None
        self.contacts = []

    def observe(self):
        return {
            "scene_revision": self.revision,
            "state_fingerprint": hashlib.sha256(self.physical_state.encode()).hexdigest(),
            "simulation_time_s": 0.0, "skill": self.skill.state(),
            "safety": {"phase": "free_space", "numerically_finite": True,
                       "numerical_error": None, "joint_limits_valid": True,
                       "contact_violations": list(self.contacts)},
            "execution_error": self.error, "source": {"revision": SOURCE_REVISION},
            "scope": {"observation": "structured_state_only", "camera_input": False,
                      "vla_connected": False, "policy_connected": False, "hardware_execution_enabled": False},
        }


class G1PlanningContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.provider = ProviderDouble()
        self.skill = SkillDouble()
        self.scene = SceneDouble(self.skill)
        config_patch = patch("g1_planning.PlannerConfig.load", return_value=PlannerConfig("declared-test-provider"))
        provider_patch = patch("g1_planning.request_structured", self.provider)
        config_patch.start()
        provider_patch.start()
        self.addCleanup(config_patch.stop)
        self.addCleanup(provider_patch.stop)
        self.session = G1PlanningSession(ROOT, self.skill, self.scene.observe)
        self.addAsyncCleanup(self.session.close)

    async def ready(self):
        started = self.session.start("请右臂抬起保持一秒再返回")
        self.assertEqual(started["status"], "planning")
        self.assertTrue(started["locks_scene"])
        call = await self.provider.next_call()
        self.assertEqual(self.skill.starts, 0, "Model planning alone must not dispatch")
        await self.provider.reply(call, execution_decision())
        self.assertEqual(self.session.state()["status"], "ready")
        return self.session.state()["decision_id"]

    async def finish_assessment(self, outcome):
        call = await self.provider.next_call()
        self.assertEqual(call["payload"]["local_outcome"], outcome)
        await self.provider.reply(call, assessment(outcome))
        return call

    async def test_only_reviewed_dispatch_and_terminal_agreement_allow_success(self):
        decision_id = await self.ready()
        self.session.execute(decision_id)
        self.assertEqual(self.skill.starts, 1)
        self.assertEqual(self.session.state()["status"], "executing")
        self.assertFalse(self.session.report()["success"])
        for _ in range(3):
            self.session.advance()
        self.assertEqual(len(self.provider.calls), 1)
        self.skill.finish("succeeded")
        self.session.advance()
        self.assertEqual(self.session.state()["status"], "assessing")
        self.assertEqual(self.session.state()["local_outcome"], "succeeded")
        self.assertFalse(self.session.report()["success"])
        await self.finish_assessment("succeeded")
        report = self.session.report()
        self.assertTrue(report["success"])
        self.assertEqual(len(report["history"]), 2)
        self.assertEqual(self.skill.starts, 1)
        self.assertFalse(report["vla_connected"])
        self.assertFalse(report["hardware_execution_enabled"])
        self.assertNotIn("execution_result", self.session.state())

    async def test_unsupported_entire_request_never_dispatches_partial_skill(self):
        self.session.start("抬右臂后拿手机拍照")
        call = await self.provider.next_call()
        await self.provider.reply(call, unsupported_decision())
        state = self.session.state()
        self.assertEqual(state["status"], "unsupported")
        with self.assertRaises(ValueError):
            self.session.execute(state["decision_id"])
        self.session.advance()
        self.assertEqual(self.skill.starts, 0)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertIsNone(self.session.report()["execution_result"])

    async def test_out_of_catalog_or_parameterized_or_malformed_response_cannot_move(self):
        mutations = [
            {"skill": "phone_photo"}, {"parameters": {"hold_s": 5}},
            {"decision": "complete"}, {"unsupported_requirements": ["take photo"]},
            {"summary": " "}, {"reason": ""}, {"extra": "motor command"},
            {"skill": None}, {"parameters": None}, {"unsupported_requirements": [3]},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.session.reset()
                self.session.start("请求执行")
                call = await self.provider.next_call()
                await self.provider.reply(call, execution_decision() | mutation)
                self.assertEqual(self.session.state()["status"], "failed")
                self.assertIsNotNone(self.session.state()["error"])
                with self.assertRaises(ValueError):
                    self.session.execute(self.session.state()["decision_id"])
                self.assertEqual(self.skill.starts, 0)

    async def test_invalid_unsupported_protocol_does_not_become_ready(self):
        self.session.start("请拍照")
        call = await self.provider.next_call()
        await self.provider.reply(call, unsupported_decision() | {"unsupported_requirements": []})
        self.assertEqual(self.session.state()["status"], "failed")
        self.assertEqual(self.skill.starts, 0)

    async def test_bool_stale_and_duplicate_approval_cannot_dispatch(self):
        decision_id = await self.ready()
        for invalid in (True, False, str(decision_id), decision_id - 1, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.session.execute(invalid)
        self.assertEqual(self.skill.starts, 0)
        self.session.execute(decision_id)
        with self.assertRaises(ValueError):
            self.session.execute(decision_id)
        self.assertEqual(self.skill.starts, 1)

    async def test_changed_revision_fingerprint_or_observation_invalidates_approval(self):
        for field in ("revision", "physical_state", "contacts", "error"):
            with self.subTest(field=field):
                self.session.reset()
                self.scene.revision = 0
                self.scene.physical_state = "test-only-unchanged-state"
                self.scene.contacts = []
                self.scene.error = None
                decision_id = await self.ready()
                if field == "revision":
                    self.scene.revision += 1
                elif field == "physical_state":
                    self.scene.physical_state = "changed-actual-state"
                elif field == "contacts":
                    self.scene.contacts = ["workspace_contact"]
                else:
                    self.scene.error = "host fault"
                with self.assertRaises(PlannerError):
                    self.session.execute(decision_id)
                self.assertEqual(self.session.state()["status"], "failed")
                self.assertEqual(self.skill.starts, 0)
                with self.assertRaises(ValueError):
                    self.session.execute(decision_id)

    async def test_faulted_or_nonidle_skill_and_invalid_tasks_reject_before_model(self):
        for invalid in (None, True, 1, "", " " * 5, "x" * 1001):
            with self.subTest(task=invalid), self.assertRaises(ValueError):
                self.session.start(invalid)
        self.skill.finish("failed")
        with self.assertRaises(PlannerError):
            self.session.start("执行")
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.skill.starts, 0)

    async def test_concurrent_request_rejected_without_replacing_current_proposal(self):
        self.session.start("第一次请求")
        call = await self.provider.next_call()
        with self.assertRaises(ValueError):
            self.session.start("第二次请求")
        await self.provider.reply(call, execution_decision())
        self.assertEqual(self.session.state()["task"], "第一次请求")
        self.assertEqual(self.skill.starts, 0)

    async def test_stop_waits_for_true_terminal_and_never_restarts_stop_timer(self):
        self.session.execute(await self.ready())
        self.session.cancel()
        self.session.cancel()
        self.assertEqual(self.skill.stops, 1)
        self.assertEqual(self.session.state()["status"], "stopping")
        self.session.advance()
        self.assertEqual(len(self.provider.calls), 1)
        self.assertIsNone(self.session.state()["local_outcome"])
        self.skill.finish("stopped")
        self.session.advance()
        self.assertEqual(self.session.state()["status"], "assessing")
        call = await self.finish_assessment("stopped")
        self.assertTrue(call["payload"]["terminal_observed"])
        self.assertTrue(call["payload"]["physical_stop_proven"])
        self.assertEqual(self.session.state()["status"], "stopped")
        self.assertFalse(self.session.report()["success"])
        self.assertEqual(self.skill.starts, 1)

    async def test_real_stop_failure_remains_failed_not_user_cancelled(self):
        self.session.execute(await self.ready())
        self.session.cancel()
        self.skill.finish("failed")
        self.session.advance()
        await self.finish_assessment("failed")
        self.assertEqual(self.session.state()["status"], "failed")
        self.assertEqual(self.session.report()["execution_result"]["status"], "failed")
        self.assertFalse(self.session.report()["success"])

    async def test_immediate_rejection_gets_feedback_from_actual_result(self):
        decision_id = await self.ready()
        self.skill.start_outcome = "rejected"
        self.session.execute(decision_id)
        self.assertEqual(self.session.state()["status"], "assessing")
        call = await self.finish_assessment("rejected")
        self.assertEqual(call["payload"]["execution_result"]["status"], "rejected")
        self.assertEqual(self.session.state()["status"], "rejected")
        self.assertEqual(self.skill.starts, 1)

    async def test_feedback_disagreement_preserves_actual_success_but_fails_workflow(self):
        self.session.execute(await self.ready())
        self.skill.finish("succeeded")
        self.session.advance()
        call = await self.provider.next_call()
        await self.provider.reply(call, assessment("stopped"))
        report = self.session.report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["local_outcome"], "succeeded")
        self.assertEqual(report["execution_result"]["status"], "succeeded")
        self.assertEqual(report["assessment"]["outcome"], "stopped")
        self.assertFalse(report["assessment_valid"])
        self.assertFalse(report["success"])
        self.assertIsNotNone(report["error"])
        self.session.advance()
        self.assertEqual(self.skill.starts, 1)
        self.assertEqual(len(self.provider.calls), 2)

    async def test_failed_engine_cannot_be_upgraded_by_model_success(self):
        self.session.execute(await self.ready())
        self.skill.finish("failed")
        self.session.advance()
        call = await self.provider.next_call()
        await self.provider.reply(call, assessment("succeeded"))
        report = self.session.report()
        self.assertFalse(report["success"])
        self.assertEqual(report["local_outcome"], "failed")
        self.assertEqual(report["execution_result"]["status"], "failed")

    async def test_provider_failure_never_dispatches_or_fakes_feedback(self):
        self.session.start("执行")
        call = await self.provider.next_call()
        await self.provider.reply(call, PlannerError("declared provider failure"))
        self.assertEqual(self.session.state()["status"], "failed")
        self.assertEqual(self.skill.starts, 0)
        self.session.execute(await self.ready())
        self.skill.finish("succeeded")
        self.session.advance()
        call = await self.provider.next_call()
        await self.provider.reply(call, PlannerError("declared assessment outage"))
        report = self.session.report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["local_outcome"], "succeeded")
        self.assertIsNone(report["assessment"])
        self.assertFalse(report["success"])
        self.assertEqual(self.skill.starts, 1)

    async def test_malformed_feedback_cannot_erase_actual_outcome(self):
        self.session.execute(await self.ready())
        self.skill.finish("stopped")
        self.session.advance()
        call = await self.provider.next_call()
        await self.provider.reply(call, assessment("stopped") | {"next_action": "execute_again"})
        report = self.session.report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["local_outcome"], "stopped")
        self.assertIsNone(report["assessment"])
        self.assertFalse(report["success"])
        self.assertEqual(self.skill.starts, 1)

    async def test_host_failure_preserves_running_result_and_unproven_stop(self):
        self.session.execute(await self.ready())
        self.session.fail_execution("declared host integration failure")
        call = await self.provider.next_call()
        payload = call["payload"]
        self.assertEqual(payload["execution_result"]["status"], "running")
        self.assertEqual(payload["local_outcome"], "failed")
        self.assertFalse(payload["terminal_observed"])
        self.assertFalse(payload["physical_stop_proven"])
        await self.provider.reply(call, assessment("failed"))
        report = self.session.report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["execution_result"]["status"], "running")
        self.assertFalse(report["success"])
        self.assertEqual(self.skill.stops, 0)

    async def test_start_exception_consumes_only_dispatch_and_reports_actual_engine(self):
        self.skill.start_error = RuntimeError("declared startup failure")
        self.session.execute(await self.ready())
        call = await self.provider.next_call()
        self.assertEqual(call["payload"]["local_outcome"], "failed")
        self.assertEqual(call["payload"]["execution_result"]["status"], "running")
        await self.provider.reply(call, assessment("failed"))
        self.assertEqual(self.skill.starts, 1)
        self.assertFalse(self.session.report()["success"])

    async def test_feedback_excludes_local_trace_and_private_paths(self):
        self.session.execute(await self.ready())
        self.skill.finish("succeeded")
        self.session.advance()
        call = await self.provider.next_call()
        serialized = json.dumps(call["payload"])
        self.assertNotIn("test_only_trace", serialized)
        self.assertNotIn("/private/test-only-source", serialized)
        self.assertNotIn("home_world_command", serialized)
        await self.provider.reply(call, assessment("succeeded"))

    async def test_cancel_planning_ignores_late_provider_reply(self):
        self.provider.ignore_cancellation = True
        self.session.start("执行")
        call = await self.provider.next_call()
        self.session.cancel()
        await self.provider.reply(call, execution_decision())
        self.assertEqual(self.session.state()["status"], "cancelled")
        self.assertIsNone(self.session.state()["decision_id"])
        self.assertIsNone(self.session.state()["decision"])
        self.assertEqual(self.skill.starts, 0)

    async def test_reset_ignores_late_reply_and_old_nonce_cannot_approve_new_run(self):
        self.provider.ignore_cancellation = True
        old_id = self.session.start("旧请求")["decision_id"]
        old_call = await self.provider.next_call()
        self.session.reset()
        await self.provider.reply(old_call, execution_decision())
        self.assertEqual(self.session.state()["status"], "idle")
        new_id = await self.ready()
        self.assertNotEqual(old_id, new_id)
        with self.assertRaises(ValueError):
            self.session.execute(old_id)
        self.assertEqual(self.skill.starts, 0)

    async def test_cancel_assessment_preserves_execution_but_not_overall_success(self):
        self.provider.ignore_cancellation = True
        self.session.execute(await self.ready())
        self.skill.finish("succeeded")
        self.session.advance()
        call = await self.provider.next_call()
        self.session.cancel()
        await self.provider.reply(call, assessment("succeeded"))
        report = self.session.report()
        self.assertEqual(report["status"], "cancelled")
        self.assertTrue(report["assessment_cancelled"])
        self.assertEqual(report["execution_result"]["status"], "succeeded")
        self.assertEqual(report["local_outcome"], "succeeded")
        self.assertIsNone(report["assessment"])
        self.assertFalse(report["success"])

    async def test_close_waits_for_provider_cleanup_and_issues_no_robot_commands(self):
        self.provider.cleanup_gate = asyncio.Event()
        self.session.start("执行")
        await self.provider.next_call()
        closing = asyncio.create_task(self.session.close())
        await asyncio.wait_for(self.provider.cleanup_started.wait(), 1)
        self.assertFalse(closing.done())
        self.provider.cleanup_gate.set()
        await asyncio.wait_for(closing, 1)
        self.assertTrue(self.provider.reaped.is_set())
        self.assertEqual(self.skill.starts, 0)
        self.assertEqual(self.skill.stops, 0)
        with self.assertRaises(ValueError):
            self.session.start("不能复活关闭的会话")

    async def test_close_during_execution_does_not_claim_physical_stop(self):
        self.session.execute(await self.ready())
        await self.session.close()
        report = self.session.report()
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["execution_result"]["status"], "running")
        self.assertIsNotNone(report["execution_error"])
        self.assertFalse(report["physical_stop_proven"])
        self.assertFalse(report["success"])
        self.assertEqual(self.skill.stops, 0)

    async def test_close_after_engine_terminal_preserves_fact_without_fabricating_assessment(self):
        self.session.execute(await self.ready())
        self.skill.finish("succeeded")
        await self.session.close()
        report = self.session.report()
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["local_outcome"], "succeeded")
        self.assertTrue(report["terminal_observed"])
        self.assertIsNone(report["execution_error"])
        self.assertTrue(report["assessment_cancelled"])
        self.assertIsNone(report["assessment"])
        self.assertFalse(report["success"])
        self.assertEqual(self.skill.stops, 0)


class G1ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = activate_source()
        from stack_scene import build_model, reset_to_reference_pose
        cls.model = build_model()
        cls.reset_scene = staticmethod(reset_to_reference_pose)

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        self.reset_scene(self.model, self.data)
        self.skill = ArmRaiseSkill(self.model, self.data, source=self.source)

    def test_observation_reads_actual_state_without_physics_or_skill_mutation(self):
        before = {name: getattr(self.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
        before_time = self.data.time
        before_result = deepcopy(self.skill.result())
        observation = make_observation(self.model, self.data, self.skill, scene_revision=7)
        json.dumps(observation, allow_nan=False)
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(self.data, name), values)
        self.assertEqual(self.data.time, before_time)
        self.assertEqual(self.skill.result(), before_result)
        self.assertNotIn("path", observation["source"])

    def test_fingerprint_changes_for_every_bound_actual_state_component(self):
        original = make_observation(self.model, self.data, self.skill, scene_revision=0)
        for name in ("qpos", "qvel", "ctrl"):
            with self.subTest(component=name):
                values = getattr(self.data, name)
                saved = values[0]
                values[0] += 1e-7
                changed = make_observation(self.model, self.data, self.skill, scene_revision=0)
                self.assertNotEqual(changed["state_fingerprint"], original["state_fingerprint"])
                values[0] = saved
        self.data.time += .002
        changed = make_observation(self.model, self.data, self.skill, scene_revision=0)
        self.assertNotEqual(changed["state_fingerprint"], original["state_fingerprint"])

    def test_nonfinite_current_state_is_explicit_and_json_safe(self):
        self.data.qvel[0] = np.nan
        observation = make_observation(self.model, self.data, self.skill, scene_revision=0,
                                       execution_error="declared observation fault")
        self.assertFalse(observation["safety"]["numerically_finite"])
        self.assertEqual(observation["safety"]["numerical_error"], "non_finite_simulation_state")
        json.dumps(observation, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
