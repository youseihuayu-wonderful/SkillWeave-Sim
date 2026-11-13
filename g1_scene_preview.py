"""Local G1 subscription-planner review and scripted-skill physics observer."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import io
import json
from pathlib import Path

from aiohttp import web
import mujoco
import numpy as np
from PIL import Image

from g1_arm_skill import ArmRaiseSkill
from g1_planning import G1PlanningSession, make_observation
from g1_source import activate_source
from planner import PlannerError


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SkillWeave Sim · G1 订阅规划与真实物理</title>
<style>
:root{color-scheme:dark;font:15px/1.55 system-ui,sans-serif;background:#10151d;color:#e5edf7}
body{margin:0;padding:12px;max-width:1280px;margin-inline:auto}h1{margin:0;font-size:23px}h2{font-size:17px;margin:10px 0}p{margin:7px 0}
.muted,dt{color:#a9b8cc}.notice{padding:6px 10px;border:1px solid #b58a38;border-radius:8px;background:#302819;margin:8px 0}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}button,select,textarea{font:inherit;padding:7px 10px;border-radius:7px;background:#213047;border:1px solid #465b78;color:inherit}
button{cursor:pointer}button:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid #78baff}button:disabled{opacity:.45;cursor:default}
textarea{display:block;box-sizing:border-box;width:100%;resize:vertical;min-height:46px}.workspace{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:16px}
img{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;background:#080c12;border-radius:10px}
.stats{display:flex;gap:14px;flex-wrap:wrap;font-variant-numeric:tabular-nums}.error{color:#ffb2b2;white-space:pre-wrap}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,monospace;max-height:260px;overflow:auto;margin:6px 0}
.panel{padding:10px;background:#172130;border-radius:8px;margin-bottom:10px}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:8px 16px}
.metrics div{padding:8px;background:#172130;border-radius:7px}.metrics dd{margin:0;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}a{color:#99caff}code{overflow-wrap:anywhere}summary{cursor:pointer}
@media(max-width:820px){.workspace{grid-template-columns:1fr}h1{font-size:20px}body{padding:8px}}
</style></head><body>
<h1>G1 · GPT 订阅规划 → 审批 → 人工编码技能</h1>
<div class="notice"><strong>MuJoCo 真实物理 · 非 VLA · 真机禁用</strong>。唯一能力：固定受限右臂抬升 3 秒、保持 1 秒、返回 3 秒，并实测稳定；不支持手机、拍照或其他动作。</div>
<section aria-label="自然语言规划">
<details id="planner-panel" open><summary>任务输入与当前提案（执行时折叠以展示机器人）</summary>
<label for="planner-task">用中文描述完整任务（规划本身不会运动，必须审批当前提案）</label>
<textarea id="planner-task" rows="1" maxlength="1000" placeholder="输入你的完整任务；未知或超出能力的要求应被拒绝，不会部分替代。"></textarea>
<section class="panel" id="proposal-panel" hidden><h2>当前模型提案（实际回复）</h2><p id="proposal-task"></p><p id="planner-decision" role="status">尚未请求模型，无预填提案。</p><p id="proposal-id"></p><pre id="decision-details"></pre></section>
<div class="controls"><button id="planner-start">请求真实 GPT 规划</button><button id="planner-execute" disabled>批准当前提案并执行</button><a href="/planner/report" download="g1-planning-report.json">导出规划与实际结果</a></div></details>
<div class="controls"><button id="planner-cancel" disabled>取消规划 / 物理停止</button><strong id="planner-status" role="status">等待规划器状态</strong></div>
<p id="planner-error" class="error" role="alert"></p></section>
<div class="workspace"><main>
<img id="frame" alt="当前 G1 MuJoCo 场景实际渲染">
<p><strong id="status">等待实际技能状态</strong> · <span id="phase">—</span> · 技能仿真时长 <output id="elapsed">0.000</output> s</p>
<p id="error" class="error" role="alert"></p>
<div class="stats"><span id="mode">初始化中</span><span>仿真时钟 <output id="time">0.000</output> s</span><span>接触对 <output id="contacts">0</output></span></div>
<div class="controls"><button id="stop">物理停止并验证稳定</button><button id="reset">显式复位并暂停</button><label>视角 <select id="camera"><option value="overview">外部观察</option><option value="cam_left_high">头部相机</option><option value="cam_left_wrist">左腕相机</option><option value="cam_right_wrist">右腕相机</option></select></label></div>
<details><summary>独立手动基线 / 空闲预览（不是规划失败回退）</summary>
<div class="controls"><button id="start">手动开始固定抬臂技能</button><button id="resume">播放空闲物理</button><button id="pause">暂停空闲预览</button><button id="step">单步 2 ms</button></div>
<p class="muted">手动基线绕过规划审批，仅供独立验证；规划会话锁定时不可用。终态后须显式复位。</p></details>
<details><summary>实际物理测量与原始技能结果</summary><p id="reason" role="status"></p><dl class="metrics" id="metrics"></dl>
<a href="/skill/result" download="g1-scripted-skill-result.json">下载引擎原始结果（含实测轨迹）</a></details>
<p class="muted">物理停止会继续推进真实仿真直至实测稳定；暂停不等于安全停止。规划、待审批和结果评估期间物理时钟暂停。</p>
</main><aside>
<p id="connection" class="muted">尚无连接证据</p>
<section class="panel"><h2>真实执行后的模型反馈</h2><p class="muted">以下为模型原文，不覆盖本地终态；低速停稳不代表碰撞已解除或真机安全。</p><p id="planner-assessment" role="status">尚无执行反馈。</p><pre id="assessment-details"></pre><p id="local-outcome"></p></section>
<details class="panel"><summary>真实调用元数据 / 历史</summary><pre id="planner-history">尚无调用记录。</pre></details>
<p class="muted">模型只选择固定人工技能；不生成电机指令、不理解相机画面、不连接 VLA 或真机。只有本地实测成功且模型反馈一致才构成规划运行成功；模型不能覆盖物理失败。</p>
<details><summary>源工程任务说明（未执行）</summary><code id="prompt"></code><p class="muted">实际执行器仅为受限抬臂宏动作，不是叠块策略。</p></details>
</aside></div>
<script>
const $=id=>document.getElementById(id);
const statuses={idle:'空闲',running:'执行中',stopping:'物理停止验证中',succeeded:'成功（引擎实测通过；规划反馈另列）',stopped:'已物理停止（非成功）',failed:'失败（需要复位）',rejected:'启动被拒绝'};
const plannerStatuses={idle:'尚未规划',unconfigured:'订阅规划器未配置',planning:'正在请求真实模型 · 物理暂停',ready:'待审批当前提案 · 物理暂停',executing:'已审批 · 正在执行真实技能',stopping:'正在物理停止并验证稳定',assessing:'正在请求真实结果反馈 · 物理暂停',unsupported:'完整请求不受支持 · 未执行',cancelled:'规划 / 评估已取消',succeeded:'规划运行成功（实测成功且反馈一致）',stopped:'已实测停止（非成功）',failed:'规划或结果评估失败',rejected:'实际启动被拒绝'};
const phases={idle:'空闲',initial_hold:'初始保持与稳定验证',raise:'抬升',hold:'保持',return:'返回原始位姿',settle:'返回后稳定验证',stopping:'物理停止验证',terminal:'终态'};
const metricNames={right_elevation_m:'右末端当前抬升（m）',max_right_elevation_m:'右末端最大抬升（m）',tracking_error_m:'实际末端跟踪误差（m）',orientation_tracking_error_rad:'实际末端姿态跟踪误差（rad）',left_return_error_m:'左末端返回误差（m）',right_return_error_m:'右末端返回误差（m）',return_orientation_error_rad:'末端返回姿态误差（rad）',robot_max_abs_velocity:'机器人最大绝对广义速度（平移 m/s；转动 rad/s）',eef_speed_m_s:'实际末端速度（m/s）',base_xy_drift_m:'基座水平漂移（m）',pelvis_height_m:'骨盆高度（m）',base_tilt_rad:'基座倾角（rad）',hold_above_minimum_s:'达到最低抬升的实际保持时长（s）',settled_for_s:'持续满足低速稳定条件（s）',stop_settled:'物理停止稳定验证通过',initial_settled_for_s:'初始参考位姿持续满足稳定条件（s）'};
const terminal=status=>['succeeded','stopped','failed','rejected'].includes(status);
const number=(value,digits=4)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(digits):value==null?'—':String(value);
let currentState=null,currentDecisionId=null,sending=false,actionError='';
async function json(url,options){const r=await fetch(url,{cache:'no-store',...options});const data=await r.json();if(!r.ok)throw Error(data.error||data.skill?.reason||r.statusText);return data;}
function showMetrics(metrics){const rows=[];function visit(value,path){if(value&&typeof value==='object'&&!Array.isArray(value)){for(const [key,item] of Object.entries(value))visit(item,path?path+'.'+key:key);return;}const row=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=metricNames[path]||path;dd.textContent=Array.isArray(value)?value.map(item=>number(item)).join(', '):typeof value==='boolean'?(value?'是':'否'):number(value);row.append(dt,dd);rows.push(row);}visit(metrics||{},'');$('metrics').replaceChildren(...rows);}
function show(s){
  const previous=currentState?.planner?.status;
  currentState=s;const skill=s.skill,p=s.planner,active=skill.active,blocked=!!s.error||skill.requires_reset,done=terminal(skill.status),locked=p.locks_scene;
  $('time').textContent=number(s.time_s,3);$('contacts').textContent=s.contacts;$('mode').textContent=active?'技能物理推进中':s.paused?'已暂停':'空闲物理预览中';$('prompt').textContent=s.task_prompt;
  $('error').textContent=actionError||s.error||'';$('status').textContent=statuses[skill.status]||skill.status;$('phase').textContent=phases[skill.phase]||skill.phase;$('elapsed').textContent=number(skill.elapsed_s,3);$('reason').textContent=skill.reason?'实际原因 / 结果：'+skill.reason:'尚无引擎终态结论';showMetrics(skill.metrics);
  $('start').disabled=sending||locked||active||blocked||done;$('stop').disabled=sending||!active||!!s.error||skill.status==='stopping';$('resume').disabled=sending||locked||active||!s.paused||blocked||done;$('pause').disabled=sending||locked||active||s.paused;$('step').disabled=sending||locked||active||!s.paused||blocked||done;$('reset').disabled=sending||!s.reset_allowed;
  $('connection').textContent=(s.planner_connected?'已有真实响应线程证据':s.planner_configured?'已找到订阅 CLI，尚无真实响应证据':'未找到可用订阅 CLI')+' · '+(p.model||'模型未配置')+' · VLA / 真机：禁用';
  $('planner-status').textContent=plannerStatuses[p.status]||p.status;$('planner-error').textContent=p.error||'';
  const decision=p.decision,assessment=p.assessment;currentDecisionId=p.decision_id;
  $('proposal-panel').hidden=!decision;
  if(['ready','unsupported'].includes(p.status)&&previous!==p.status)$('planner-panel').open=true;
  if(p.status==='executing'&&previous!==p.status){$('planner-panel').open=false;window.scrollTo(0,0);}
  $('planner-decision').textContent=decision?(decision.decision==='unsupported'?'不支持完整请求，未执行：':'模型建议执行固定技能：')+decision.summary:'尚无实际模型提案。';
  $('proposal-id').textContent=!decision||p.decision_id==null?'':'当前提案 ID：'+p.decision_id;
  $('proposal-task').textContent=decision?'本提案绑定的任务：'+p.task:'';
  $('decision-details').textContent=decision?JSON.stringify(decision,null,2):'';
  $('planner-assessment').textContent=assessment?assessment.summary:'尚无实际模型反馈；不能据此宣称规划运行成功。';
  $('assessment-details').textContent=assessment?JSON.stringify(assessment,null,2):'';
  $('local-outcome').textContent=p.local_outcome?'本地实测结果：'+(statuses[p.local_outcome]||p.local_outcome):'尚无本次规划的实际终态结果。';
  $('planner-history').textContent=JSON.stringify(p,null,2);
  $('planner-start').disabled=sending||locked||active||blocked||done||!s.planner_configured||!$('planner-task').value.trim();$('planner-task').disabled=sending||locked;
  $('planner-execute').disabled=sending||p.status!=='ready'||decision?.decision!=='execute'||!Number.isSafeInteger(currentDecisionId)||blocked;
  $('planner-cancel').disabled=sending||!locked||p.status==='stopping';
}
async function send(url,payload){if(sending)return;sending=true;actionError='';if(currentState)show(currentState);try{const options={method:'POST'};if(payload!==undefined){options.headers={'Content-Type':'application/json'};options.body=JSON.stringify(payload);}show(await json(url,options));}catch(e){actionError=e.message;}finally{sending=false;if(currentState)show(currentState);}}
for(const action of ['resume','pause','step','reset'])$(action).onclick=()=>send('/control/'+action);
for(const action of ['start','stop'])$(action).onclick=()=>send('/skill/'+action);
$('planner-task').oninput=()=>{if(currentState)show(currentState);};
$('planner-start').onclick=()=>send('/planner/start',{task:$('planner-task').value});
$('planner-execute').onclick=()=>send('/planner/execute',{decision_id:currentDecisionId});
$('planner-cancel').onclick=()=>send('/planner/cancel');
let previousImage=null;
async function refresh(){try{if(!document.hidden){show(await json('/state'));const r=await fetch('/frame.jpg?camera='+encodeURIComponent($('camera').value),{cache:'no-store'});if(!r.ok){const d=await r.json();throw Error(d.error||'渲染失败');}const next=URL.createObjectURL(await r.blob());$('frame').src=next;if(previousImage)URL.revokeObjectURL(previousImage);previousImage=next;}}catch(e){$('error').textContent=e.message;}finally{setTimeout(refresh,document.hidden?1000:200);}}
refresh();
</script></body></html>"""


class ScenePreview:
    def __init__(self, source: Path | None = None):
        activate_source(source)
        self.scene = importlib.import_module("stack_scene")
        self.model = self.scene.build_model()
        if abs(self.model.opt.timestep - 0.002) > 1e-9:
            raise ValueError("This observer requires the frozen 0.002 s MuJoCo timestep")
        self.data = mujoco.MjData(self.model)
        self.scene.reset_to_reference_pose(self.model, self.data)
        self.skill = ArmRaiseSkill(self.model, self.data, source=source)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = (0.35, 0.0, 0.82)
        self.camera.distance = 2.3
        self.camera.azimuth = 145
        self.camera.elevation = -22
        self.paused = True
        self.error = None
        self.render_error = None
        self.scene_revision = 0
        self.planner = G1PlanningSession(
            Path(__file__).resolve().parent,
            self.skill,
            lambda: make_observation(
                self.model,
                self.data,
                self.skill,
                scene_revision=self.scene_revision,
                execution_error=self.error,
            ),
        )
        self.planner_response_observed = False

    def state(self):
        planner = self.planner.state()
        self.planner_response_observed |= self.has_response_thread(planner)
        return {
            "time_s": float(self.data.time) if np.isfinite(self.data.time) else None,
            "contacts": int(self.data.ncon),
            "paused": self.paused,
            "error": self.error or self.render_error,
            "task_prompt": self.scene.TASK_PROMPT,
            "skill": self.skill.state(),
            "executor_kind": "scripted_skill",
            "planner": planner,
            "planner_configured": planner["configured"],
            "planner_connected": self.planner_response_observed,
            "policy_connected": False,
            "vla_enabled": False,
            "hardware_execution_enabled": False,
            "reset_allowed": (
                not self.skill.active and not self.planner.locks_scene
            ) or bool(self.error and self.paused),
        }

    @staticmethod
    def has_response_thread(value):
        if isinstance(value, dict):
            if isinstance(value.get("thread_id"), str) and value["thread_id"]:
                return True
            return any(ScenePreview.has_response_thread(item) for item in value.values())
        if isinstance(value, list):
            return any(ScenePreview.has_response_thread(item) for item in value)
        return False

    def advance(self, steps):
        for _ in range(steps):
            if self.planner.locks_scene and not self.skill.active:
                self.planner.advance()
                self.paused = True
                break
            self.skill.step()
            self.planner.advance()
            if self.skill.state()["status"] in {"succeeded", "stopped", "failed", "rejected"}:
                self.paused = True
                break

    async def tick(self):
        while True:
            if not self.paused:
                try:
                    self.advance(10)
                except Exception as exc:
                    self.paused = True
                    self.error = f"物理推进异常，未证明安全停止；需要复位或重启：{type(exc).__name__}: {exc}"
                    if self.planner.state()["status"] in {"executing", "stopping"}:
                        self.planner.fail_execution(self.error)
            await asyncio.sleep(0.02)

    async def control(self, request):
        action = request.match_info["action"]
        if action not in {"resume", "pause", "step", "reset"}:
            return web.json_response({"error": "未知场景控制"}, status=400)
        fault_reset = action == "reset" and self.error and self.paused
        if self.planner.locks_scene and not fault_reset:
            return web.json_response(
                {"error": "规划、审批、执行或结果评估期间锁定场景；请先取消，执行中请物理停止。"},
                status=409,
            )
        if self.skill.active and not (action == "reset" and self.error and self.paused):
            return web.json_response(
                {"error": "技能执行或物理停止验证期间禁止暂停、复位、单步和手动播放；请使用物理停止。"},
                status=409,
            )
        if action != "reset":
            if self.error or self.skill.requires_reset:
                return web.json_response({"error": self.error or "技能故障，需要复位。"}, status=409)
            if self.skill.state()["status"] in {"succeeded", "stopped", "failed", "rejected"}:
                return web.json_response({"error": "技能已经终止，请先复位再预览。"}, status=409)
        try:
            if action == "reset":
                self.paused = True
                self.planner.reset()
                self.scene.reset_to_reference_pose(self.model, self.data)
                self.skill.reset()
                self.scene_revision += 1
                self.error = None
                self.render_error = None
            elif action == "pause":
                self.paused = True
            elif action == "resume":
                self.paused = False
            else:
                if not self.paused:
                    return web.json_response({"error": "请先暂停空闲预览，再单步。"}, status=409)
                self.advance(1)
        except Exception as exc:
            self.paused = True
            self.error = f"场景控制异常：{type(exc).__name__}: {exc}"
            return web.json_response({"error": self.error}, status=500)
        return web.json_response(self.state())

    async def start_skill(self, request):
        if self.planner.locks_scene:
            return web.json_response({"error": "规划会话锁定场景，不能启动手动基线。"}, status=409)
        if self.skill.active:
            return web.json_response({"error": "技能正在执行或停止验证中。", "skill": self.skill.state()}, status=409)
        if self.error or self.skill.requires_reset:
            return web.json_response({"error": self.error or "需要先复位。", "skill": self.skill.state()}, status=409)
        try:
            skill = self.skill.start()
            self.paused = not self.skill.active
            if skill["status"] == "rejected" or not self.skill.active:
                return web.json_response(
                    {"error": skill.get("reason") or "技能未启动。", "skill": skill},
                    status=409,
                )
            return web.json_response(self.state())
        except Exception as exc:
            self.paused = True
            self.error = f"技能启动异常：{type(exc).__name__}: {exc}"
            return web.json_response({"error": self.error}, status=500)

    async def stop_skill(self, request):
        if not self.skill.active:
            return web.json_response({"error": "当前没有可停止的活动技能。", "skill": self.skill.state()}, status=409)
        if self.error:
            return web.json_response({"error": self.error, "skill": self.skill.state()}, status=409)
        try:
            if self.planner.state()["status"] in {"executing", "stopping"}:
                self.planner.cancel()
            else:
                self.skill.request_stop()
            self.paused = not self.skill.active
            return web.json_response(self.state())
        except Exception as exc:
            self.paused = True
            self.error = f"物理停止请求异常，未证明安全停止：{type(exc).__name__}: {exc}"
            if self.planner.state()["status"] in {"executing", "stopping"}:
                self.planner.fail_execution(self.error)
            return web.json_response({"error": self.error}, status=500)

    async def planner_action(self, request, action, key=None):
        if key is not None:
            try:
                payload = await request.json()
            except web.HTTPException:
                raise
            except (ValueError, TypeError):
                return web.json_response({"error": "请求必须是有效 JSON。"}, status=400)
            if not isinstance(payload, dict) or set(payload) != {key}:
                return web.json_response({"error": f"请求必须且只能包含 {key} 字段。"}, status=400)
            value = payload[key]
        try:
            if action == "start":
                if self.error or self.skill.requires_reset or self.skill.active:
                    return web.json_response({"error": self.error or "技能尚未空闲；请先停止或复位。"}, status=409)
                self.planner.start(value)
                self.paused = True
            elif action == "execute":
                if self.error:
                    return web.json_response({"error": self.error}, status=409)
                self.planner.execute(value)
                self.paused = not self.skill.active
            else:
                locked = self.planner.locks_scene
                running = self.planner.state()["status"] in {"executing", "stopping"}
                self.planner.cancel()
                if locked:
                    self.paused = not (running and self.skill.active and not self.error)
            return web.json_response(self.state())
        except (ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except (PlannerError, RuntimeError) as exc:
            return web.json_response({"error": str(exc), "planner": self.planner.state()}, status=409)
        except Exception as exc:
            self.paused = True
            self.error = f"规划控制异常，未证明安全停止：{type(exc).__name__}: {exc}"
            if self.planner.state()["status"] in {"executing", "stopping"}:
                self.planner.fail_execution(self.error)
            return web.json_response({"error": self.error}, status=500)

    async def start_planner(self, request):
        return await self.planner_action(request, "start", "task")

    async def execute_planner(self, request):
        return await self.planner_action(request, "execute", "decision_id")

    async def cancel_planner(self, request):
        return await self.planner_action(request, "cancel")

    async def planner_report(self, request):
        try:
            report = json.dumps(self.planner.report(), ensure_ascii=False, allow_nan=False, indent=2)
        except Exception as exc:
            return web.json_response({"error": f"规划报告导出失败：{type(exc).__name__}: {exc}"}, status=500)
        return web.Response(
            text=report,
            content_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="g1-planning-report.json"',
                "Cache-Control": "no-store",
            },
        )

    async def skill_result(self, request):
        try:
            report = json.dumps(self.skill.result(), ensure_ascii=False, allow_nan=False, indent=2)
        except Exception as exc:
            return web.json_response(
                {"error": f"结果导出失败：{type(exc).__name__}: {exc}"},
                status=500,
            )
        return web.Response(
            text=report,
            content_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="g1-scripted-skill-result.json"',
                "Cache-Control": "no-store",
            },
        )

    async def frame(self, request):
        name = request.query.get("camera", "overview")
        if name != "overview" and name not in self.scene.CAMERA_NAMES:
            return web.json_response({"error": "未知相机"}, status=400)
        if not (
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(self.data.xpos).all()
            and np.isfinite(self.data.xmat).all()
        ):
            return web.json_response({"error": "仿真状态非有限，无法安全渲染；请复位。"}, status=503)
        try:
            self.renderer.update_scene(self.data, camera=self.camera if name == "overview" else name)
            image = Image.fromarray(self.renderer.render())
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=88)
            self.render_error = None
            return web.Response(body=buffer.getvalue(), content_type="image/jpeg", headers={"Cache-Control": "no-store"})
        except Exception as exc:
            self.render_error = f"渲染失败（不会冒充物理停止）：{type(exc).__name__}: {exc}"
            return web.json_response({"error": self.render_error}, status=500)

    async def lifecycle(self, app):
        task = asyncio.create_task(self.tick())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                try:
                    await self.planner.close()
                finally:
                    self.renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g1-source",
        type=Path,
        help="Explicit verified G1 source override; defaults to the pinned repository submodule",
    )
    parser.add_argument("--port", type=int, default=8778)
    args = parser.parse_args()
    preview = ScenePreview(args.g1_source)
    app = web.Application()

    async def index(request):
        return web.Response(text=PAGE, content_type="text/html")

    async def state(request):
        return web.json_response(preview.state())

    app.router.add_get("/", index)
    app.router.add_get("/state", state)
    app.router.add_get("/frame.jpg", preview.frame)
    app.router.add_post("/control/{action}", preview.control)
    app.router.add_post("/skill/start", preview.start_skill)
    app.router.add_post("/skill/stop", preview.stop_skill)
    app.router.add_get("/skill/result", preview.skill_result)
    app.router.add_post("/planner/start", preview.start_planner)
    app.router.add_post("/planner/execute", preview.execute_planner)
    app.router.add_post("/planner/cancel", preview.cancel_planner)
    app.router.add_get("/planner/report", preview.planner_report)
    app.cleanup_ctx.append(preview.lifecycle)
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
