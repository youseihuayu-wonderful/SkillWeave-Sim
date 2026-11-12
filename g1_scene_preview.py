"""Local G1 scripted arm-skill observer, without policy inference or robot transport."""

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
from g1_source import activate_source


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SkillWeave Sim · G1 人工编码技能基线</title>
<style>
:root{color-scheme:dark;font:15px/1.6 system-ui,sans-serif;background:#10151d;color:#e5edf7}
body{margin:0;padding:12px;max-width:1100px;margin-inline:auto}h1{margin:0;font-size:24px}
p{margin:8px 0}.muted{color:#a9b8cc}.notice{padding:6px 10px;border:1px solid #b58a38;border-radius:10px;background:#302819;margin:8px 0}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}button,select{font:inherit;padding:8px 12px;border-radius:7px;background:#213047;border:1px solid #465b78;color:inherit}
button{cursor:pointer}button:focus-visible,select:focus-visible{outline:3px solid #78baff}button:disabled{opacity:.45;cursor:default}
img{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;background:#080c12;border-radius:10px}
.stats{display:flex;gap:20px;flex-wrap:wrap;padding:12px 0;font-variant-numeric:tabular-nums}
#error{color:#ffb2b2;white-space:pre-wrap}code{overflow-wrap:anywhere}footer{margin-top:16px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:8px 20px;margin:12px 0}.metrics div{padding:8px;background:#172130;border-radius:7px}.metrics dt{color:#a9b8cc;overflow-wrap:anywhere}.metrics dd{margin:0;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}a{color:#99caff}
</style></head><body>
<h1>G1 人工编码技能基线</h1><p class="muted">受限抬臂—保持—回位 · MuJoCo 真实物理</p>
<div class="notice"><strong>人工编码技能基线 · 非VLA · 真机禁用</strong></div>
<div class="controls">
<button id="start">开始抬臂技能</button><button id="stop">物理停止并验证稳定</button><button id="reset">复位并暂停</button></div>
<p><strong id="status">等待状态</strong> · <span id="phase">—</span> · 仿真时长 <output id="elapsed">0.000</output> s</p>
<p id="error" role="alert"></p><img id="frame" alt="当前 G1 MuJoCo 场景实际渲染">
<div class="stats"><span id="mode">初始化中</span><span>仿真时钟 <output id="time">0.000</output> s</span>
<span>接触对 <output id="contacts">0</output></span><span>规划器 / VLA：未连接</span><span>真机：禁用</span></div>
<details><summary>空闲物理与相机设置</summary><div class="controls">
<button id="resume">播放空闲物理</button><button id="pause">暂停空闲预览</button><button id="step">单步 2 ms</button>
<label>视角 <select id="camera"><option value="overview">外部观察</option><option value="cam_left_high">头部相机</option><option value="cam_left_wrist">左腕相机</option><option value="cam_right_wrist">右腕相机</option></select></label></div></details>
<section aria-labelledby="skill-title"><h2 id="skill-title">实际执行状态</h2>
<p id="reason" role="status"></p><dl class="metrics" id="metrics"></dl>
<a href="/skill/result" download="g1-scripted-skill-result.json">下载真实技能结果 JSON（含配置、实测轨迹与阶段历史）</a>
<p class="muted">“物理停止”会取消原轨迹并继续推进物理，只有实测速度持续满足稳定条件才会报告已停止；暂停画面不等于完成停止验证。终态自动暂停，复位后才能重新预览。</p></section>
<footer class="muted"><p>当前人工技能：<strong>受限抬臂—保持—回参考位姿</strong></p><p>未连接规划器或策略，不创建真机通信或命令发布器。结果只来自本地仿真的实际测量；不是学习策略、VLA 成功率或硬件安全认证。</p>原模型训练任务（未执行）：<code id="prompt"></code><p>站姿、碰撞物理与相机来自经校验的指定 G1 源工程；本轮不执行叠块任务，成功、停止和失败均按抬臂引擎实测结果展示。</p></footer>
<script>
const $=id=>document.getElementById(id);
const statuses={idle:'空闲',running:'执行中',stopping:'物理停止验证中',succeeded:'成功（实测验收通过）',stopped:'已物理停止（非成功）',failed:'失败（需要复位）',rejected:'启动被拒绝'};
const phases={idle:'空闲',initial_hold:'初始保持与稳定验证',raise:'抬升',hold:'保持',return:'返回原始位姿',settle:'返回后稳定验证',stopping:'物理停止验证',terminal:'终态'};
const metricNames={right_elevation_m:'右末端当前抬升（m）',max_right_elevation_m:'右末端最大抬升（m）',tracking_error_m:'实际末端跟踪误差（m）',orientation_tracking_error_rad:'实际末端姿态跟踪误差（rad）',left_return_error_m:'左末端返回误差（m）',right_return_error_m:'右末端返回误差（m）',return_orientation_error_rad:'末端返回姿态误差（rad）',robot_max_abs_velocity:'机器人最大绝对广义速度（平移 m/s；转动 rad/s）',eef_speed_m_s:'实际末端速度（m/s）',base_xy_drift_m:'基座水平漂移（m）',pelvis_height_m:'骨盆高度（m）',base_tilt_rad:'基座倾角（rad）',hold_above_minimum_s:'达到最低抬升的实际保持时长（s）',settled_for_s:'持续满足低速稳定条件（s）',stop_settled:'物理停止稳定验证通过'};
metricNames.initial_settled_for_s='初始参考位姿持续满足稳定条件（s）';
const terminal=status=>['succeeded','stopped','failed','rejected'].includes(status);
const number=(value,digits=4)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(digits):value==null?'—':String(value);
async function json(url,options){const r=await fetch(url,{cache:'no-store',...options});const data=await r.json();if(!r.ok)throw Error(data.error||data.skill?.reason||r.statusText);return data;}
function showMetrics(metrics){const rows=[];function visit(value,path){if(value&&typeof value==='object'&&!Array.isArray(value)){for(const [key,item] of Object.entries(value))visit(item,path?path+'.'+key:key);return;}const row=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=metricNames[path]||path;dd.textContent=Array.isArray(value)?value.map(item=>number(item)).join(', '):typeof value==='boolean'?(value?'是':'否'):number(value);row.append(dt,dd);rows.push(row);}visit(metrics||{},'');$('metrics').replaceChildren(...rows);}
function show(s){const skill=s.skill,active=skill.active,blocked=!!s.error||skill.requires_reset,done=terminal(skill.status);$('time').textContent=number(s.time_s,3);$('contacts').textContent=s.contacts;$('mode').textContent=active?'技能物理推进中':s.paused?'已暂停':'空闲物理预览中';$('prompt').textContent=s.task_prompt;$('error').textContent=s.error||'';$('status').textContent=statuses[skill.status]||skill.status;$('phase').textContent=phases[skill.phase]||skill.phase;$('elapsed').textContent=number(skill.elapsed_s,3);$('reason').textContent=skill.reason?'原因 / 结果：'+skill.reason:'尚无终态结论';showMetrics(skill.metrics);$('start').disabled=active||blocked||done;$('stop').disabled=!active||skill.status==='stopping';$('resume').disabled=active||!s.paused||blocked||done;$('pause').disabled=active||s.paused;$('step').disabled=active||!s.paused||blocked||done;$('reset').disabled=!s.reset_allowed;}
for(const action of ['resume','pause','step','reset'])$(action).onclick=async()=>{try{show(await json('/control/'+action,{method:'POST'}));}catch(e){$('error').textContent=e.message;}};
for(const action of ['start','stop'])$(action).onclick=async()=>{try{show(await json('/skill/'+action,{method:'POST'}));}catch(e){$('error').textContent=e.message;}};
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

    def state(self):
        return {
            "time_s": float(self.data.time) if np.isfinite(self.data.time) else None,
            "contacts": int(self.data.ncon),
            "paused": self.paused,
            "error": self.error or self.render_error,
            "task_prompt": self.scene.TASK_PROMPT,
            "skill": self.skill.state(),
            "executor_kind": "scripted_skill",
            "planner_connected": False,
            "policy_connected": False,
            "hardware_execution_enabled": False,
            "reset_allowed": not self.skill.active or bool(self.error and self.paused),
        }

    def advance(self, steps):
        for _ in range(steps):
            self.skill.step()
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
            await asyncio.sleep(0.02)

    async def control(self, request):
        action = request.match_info["action"]
        if action not in {"resume", "pause", "step", "reset"}:
            return web.json_response({"error": "未知场景控制"}, status=400)
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
                self.scene.reset_to_reference_pose(self.model, self.data)
                self.skill.reset()
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
            self.skill.request_stop()
            self.paused = not self.skill.active
            return web.json_response(self.state())
        except Exception as exc:
            self.paused = True
            self.error = f"物理停止请求异常，未证明安全停止：{type(exc).__name__}: {exc}"
            return web.json_response({"error": self.error}, status=500)

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
    app.cleanup_ctx.append(preview.lifecycle)
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
