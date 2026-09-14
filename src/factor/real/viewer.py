"""``factor-view``: live browser view of the real-robot loop. / 실기 루프 라이브 브라우저 뷰.

Serves the directory written by :class:`factor.real.telemetry.Telemetry` as a single page that
refreshes several times a second: the camera frame with the believed and estimated geometry, the
canonical view the estimator sees, every signal of the tick (fiducial pose, corrections, grasp
probability, skill phase, contact search, target) and the tail of the log. Standard library
only, so it runs in the ROS Python next to the loop.

텔레메트리 디렉터리를 한 페이지로 서빙한다: 오버레이된 카메라 프레임, 추정기 입력(정규화 시야),
틱의 모든 신호, 로그 꼬리. 표준 라이브러리만 사용한다.

Example::

    factor-view --dir logs/factor/live --port 8765   # then open http://localhost:8765
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>FACTOR live</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:13px/1.4 system-ui,sans-serif}
 header{padding:8px 14px;background:#1b1b1b;display:flex;gap:18px;align-items:baseline}
 header b{font-size:16px;color:#fff} header .phase{padding:2px 8px;border-radius:4px;background:#333}
 main{display:grid;grid-template-columns:2fr 1fr;gap:10px;padding:10px}
 .panel{background:#1b1b1b;border-radius:6px;padding:8px}
 img{width:100%;display:block;border-radius:4px;background:#000}
 table{width:100%;border-collapse:collapse} td{padding:2px 6px;border-bottom:1px solid #2a2a2a;font-variant-numeric:tabular-nums}
 td:first-child{color:#9a9a9a;width:48%}
 .bar{height:6px;background:#2a2a2a;border-radius:3px;position:relative;margin-top:2px}
 .bar i{position:absolute;top:0;bottom:0;background:#4c9;border-radius:3px}
 pre{margin:0;max-height:260px;overflow:auto;font:11px/1.35 ui-monospace,monospace;color:#bbb;white-space:pre-wrap}
 h3{margin:0 0 6px;font-size:12px;color:#9a9a9a;text-transform:uppercase;letter-spacing:.06em}
 .legend span{display:inline-block;margin-right:12px}.legend i{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:2px}
</style></head><body>
<header><b>FACTOR live</b><span id="phase" class="phase">—</span><span id="clock"></span><span id="move"></span></header>
<main>
 <div>
  <div class="panel"><h3>camera · believed scene vs. corrected estimate</h3><img id="camera" alt="camera">
   <div class="legend" style="margin-top:6px"><span><i style="background:#ffc800"></i>believed (nominal, from the ArUco anchor)</span>
   <span><i style="background:#3cff3c"></i>corrected / held-ring estimate</span><span><i style="background:#ffdc28"></i>goal</span>
   <span><i style="background:#0078ff"></i>markers</span><span><i style="background:#fff"></i>grip point</span></div></div>
  <div class="panel" style="margin-top:10px"><h3>log</h3><pre id="log"></pre></div>
 </div>
 <div>
  <div class="panel"><h3>estimator input (canonical view)</h3><img id="canonical" alt="canonical"></div>
  <div class="panel" style="margin-top:10px"><h3>signals</h3><table id="signals"></table></div>
 </div>
</main>
<script>
const fmt=(v,d=1)=>Array.isArray(v)?v.map(x=>(+x).toFixed(d)).join('  '):(typeof v==='number'?v.toFixed(d):String(v));
const ROWS=[['fiducial markers / rms px','fiducial'],['anchor in camera (m)','anchor_in_camera_t'],
 ['raw correction (mm, mm, deg)','correction_mm_deg'],['fused correction (mm, mm, deg)','fused_mm_deg'],
 ['grasped probability','grasped'],['ensemble spread (mm)','spread_mm'],['in-hand offset est. (mm)','offset_mm'],['holding (gripper cmd)','holding'],
 ['inserting','inserting'],['contact search index / stall','nudge'],['retry (lift)','retry'],
 ['grip point, base (m)','grip'],['commanded step (mm)','step_mm'],['target, base (m)','target'],['workspace clamped','clamped'],
 ['loop rate (Hz)','rate_hz']];
function bar(v,max){const w=Math.min(100,Math.abs(v)/max*100);return `<div class="bar"><i style="left:${v<0?50-w/2:50}%;width:${w/2}%"></i></div>`}
async function tick(){
 try{
  const r=await fetch('state.json?'+Date.now(),{cache:'no-store'}); if(!r.ok) return; const s=await r.json();
  document.getElementById('phase').textContent=s.phase||'—';
  document.getElementById('clock').textContent=`t ${(+s.t||0).toFixed(1)} s · step ${s.step??''}`;
  document.getElementById('move').textContent=s.move||'';
  let html='';
  for(const [name,key] of ROWS){ if(!(key in s)) continue; let v=s[key]; let extra='';
    if(key==='grasped') extra=`<div class="bar"><i style="left:0;width:${(v*100).toFixed(0)}%"></i></div>`;
    if(key==='correction_mm_deg'||key==='fused_mm_deg') extra=bar(v[0],10)+bar(v[1],10);
    html+=`<tr><td>${name}</td><td>${fmt(v,key.includes('mm')||key==='grasped'?2:3)}${extra}</td></tr>`; }
  document.getElementById('signals').innerHTML=html;
  const t=Date.now(); document.getElementById('camera').src='camera.jpg?'+t; document.getElementById('canonical').src='canonical.jpg?'+t;
  const l=await fetch('log.tail?'+t,{cache:'no-store'}); if(l.ok) document.getElementById('log').textContent=await l.text();
 }catch(e){}
}
setInterval(tick,200); tick();
</script></body></html>
"""


