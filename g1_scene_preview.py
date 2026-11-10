"""Local G1 MuJoCo scene inspection, without policy inference or robot transport."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import io
from pathlib import Path
import sys

from aiohttp import web
import mujoco
import numpy as np
from PIL import Image


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SkillWeave Sim · G1 场景观察</title>
<style>
:root{color-scheme:dark;font:15px/1.6 system-ui,sans-serif;background:#10151d;color:#e5edf7}
body{margin:0;padding:20px;max-width:1100px;margin-inline:auto}h1{margin:0;font-size:26px}
p{margin:8px 0}.muted{color:#a9b8cc}.notice{padding:12px 16px;border:1px solid #b58a38;border-radius:10px;background:#302819;margin:16px 0}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}button,select{font:inherit;padding:8px 12px;border-radius:7px;background:#213047;border:1px solid #465b78;color:inherit}
button{cursor:pointer}button:focus-visible,select:focus-visible{outline:3px solid #78baff}button:disabled{opacity:.45;cursor:default}
img{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;background:#080c12;border-radius:10px}
.stats{display:flex;gap:20px;flex-wrap:wrap;padding:12px 0;font-variant-numeric:tabular-nums}
#error{color:#ffb2b2;white-space:pre-wrap}code{overflow-wrap:anywhere}footer{margin-top:16px}
</style></head><body>
<h1>G1 + MuJoCo</h1><p class="muted">SkillWeave Sim · 固定站姿、双 Dex1、叠方块场景</p>
<div class="notice"><strong>仅场景观察，不是 VLA 任务成功演示。</strong><br>
未加载动作模型；不创建真机通信或命令发布器。播放仅推进本地 MuJoCo 物理，维持原场景的参考站姿控制。</div>
<div class="controls">
<button id="resume">播放物理预览</button><button id="pause">暂停</button>
<button id="step">单步 2 ms</button><button id="reset">复位并暂停</button>
<label>视角 <select id="camera"><option value="overview">外部观察</option>
<option value="cam_left_high">头部相机</option><option value="cam_left_wrist">左腕相机</option>
<option value="cam_right_wrist">右腕相机</option></select></label></div>
<div class="stats"><span id="mode">初始化中</span><span>仿真时钟 <output id="time">0.000</output> s</span>
<span>接触对 <output id="contacts">0</output></span><span>VLA：未连接</span><span>真机：禁用</span></div>
<p id="error" role="alert"></p><img id="frame" alt="当前 G1 MuJoCo 场景渲染">
<footer class="muted">当前任务指令：<code id="prompt"></code><p>站姿与相机来自指定 G1 源工程。观察画面不代表标定、碰撞门槛或任务基线已通过。</p></footer>
<script>
const $=id=>document.getElementById(id);
async function json(url,options){const r=await fetch(url,options);const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data;}
function show(s){$('time').textContent=s.time_s.toFixed(3);$('contacts').textContent=s.contacts;$('mode').textContent=s.paused?'已暂停':'物理预览运行中';$('prompt').textContent=s.task_prompt;$('error').textContent=s.error||'';$('resume').disabled=!s.paused||!!s.error;$('pause').disabled=s.paused;$('step').disabled=!s.paused||!!s.error;}
for(const action of ['resume','pause','step','reset'])$(action).onclick=async()=>{try{show(await json('/control/'+action,{method:'POST'}));}catch(e){$('error').textContent=e.message;}};
let previousImage=null;
async function refresh(){try{if(!document.hidden){show(await json('/state'));const r=await fetch('/frame.jpg?camera='+encodeURIComponent($('camera').value),{cache:'no-store'});if(!r.ok){const d=await r.json();throw Error(d.error||'渲染失败');}const next=URL.createObjectURL(await r.blob());$('frame').src=next;if(previousImage)URL.revokeObjectURL(previousImage);previousImage=next;}}catch(e){$('error').textContent=e.message;}finally{setTimeout(refresh,document.hidden?1000:200);}}
refresh();
</script></body></html>"""


class ScenePreview:
    def __init__(self, source: Path):
        source = source.resolve()
        if not (source / "stack_scene.py").is_file():
            raise FileNotFoundError(f"G1 scene source missing: {source / 'stack_scene.py'}")
        sys.path.insert(0, str(source))
        self.scene = importlib.import_module("stack_scene")
        self.model = self.scene.build_model()
        if abs(self.model.opt.timestep - 0.002) > 1e-9:
            raise ValueError("This observer requires the frozen 0.002 s MuJoCo timestep")
        self.data = mujoco.MjData(self.model)
        self.scene.reset_to_reference_pose(self.model, self.data)
        self.hold = self.data.ctrl.copy()
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = (0.35, 0.0, 0.82)
        self.camera.distance = 2.3
        self.camera.azimuth = 145
        self.camera.elevation = -22
        self.paused = True
        self.error = None

    def state(self):
        return {
            "time_s": float(self.data.time),
            "contacts": int(self.data.ncon),
            "paused": self.paused,
            "error": self.error,
            "task_prompt": self.scene.TASK_PROMPT,
            "policy_connected": False,
            "hardware_execution_enabled": False,
        }

    def advance(self, steps):
        for _ in range(steps):
            self.data.ctrl[:] = self.hold
            mujoco.mj_step(self.model, self.data)
            if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
                raise RuntimeError("Non-finite simulation state; reset is required")

    async def tick(self):
        while True:
            if not self.paused:
                try:
                    self.advance(10)
                except Exception as exc:
                    self.paused = True
                    self.error = f"Physics stopped: {type(exc).__name__}: {exc}"
            await asyncio.sleep(0.02)

    async def control(self, request):
        action = request.match_info["action"]
        if action not in {"resume", "pause", "step", "reset"}:
            return web.json_response({"error": "Unknown scene control"}, status=400)
        if self.error and action != "reset":
            return web.json_response({"error": self.error}, status=409)
        if action == "reset":
            self.paused = True
            self.scene.reset_to_reference_pose(self.model, self.data)
            self.hold = self.data.ctrl.copy()
            self.error = None
        elif action == "pause":
            self.paused = True
        elif action == "resume":
            self.paused = False
        else:
            if not self.paused:
                return web.json_response({"error": "Pause before single stepping"}, status=409)
            try:
                self.advance(1)
            except Exception as exc:
                self.paused = True
                self.error = f"Physics stopped: {type(exc).__name__}: {exc}"
                return web.json_response({"error": self.error}, status=500)
        return web.json_response(self.state())

    async def frame(self, request):
        name = request.query.get("camera", "overview")
        if name != "overview" and name not in self.scene.CAMERA_NAMES:
            return web.json_response({"error": "Unknown camera"}, status=400)
        if self.error:
            return web.json_response({"error": self.error}, status=503)
        try:
            self.renderer.update_scene(self.data, camera=self.camera if name == "overview" else name)
            image = Image.fromarray(self.renderer.render())
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=88)
            return web.Response(body=buffer.getvalue(), content_type="image/jpeg", headers={"Cache-Control": "no-store"})
        except Exception as exc:
            self.paused = True
            self.error = f"Rendering stopped: {type(exc).__name__}: {exc}"
            return web.json_response({"error": self.error}, status=500)

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
    parser.add_argument("--g1-source", type=Path, required=True)
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
    app.cleanup_ctx.append(preview.lifecycle)
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
