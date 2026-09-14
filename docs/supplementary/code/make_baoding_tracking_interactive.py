#!/usr/bin/env python3
"""Build a self-contained interactive 3-D Baoding tracking viewer.

The input trajectory table is read from the authoritative Super-Server result
directory when this script is run there.  The generated HTML embeds only the
small trajectory table and remains usable offline in a browser.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "scenario": r["scenario"],
                "target": int(r["target"]),
                "time_s": float(r["time_s"]),
                "truth": [float(r["gps_east_m"]), float(r["gps_north_m"]), float(r["gps_up_m"])],
                "apce": [float(r["apce_east_m"]), float(r["apce_north_m"]), float(r["apce_up_m"])],
                "width_m": float(r["marginal_width_m"]),
            })
    return rows


def build_html(rows, source_path: str, source_sha: str):
    payload = {
        "source": source_path,
        "source_sha256": source_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baoding Acoustic Source Tracking · Interactive trajectories</title>
<style>
:root{font-family:Arial,sans-serif;color:#17191c;background:#fff;--lime:#e4ea48;--ink:#17191c;--muted:#65707a;--line:#d9dfe3;--blue:#1769aa;--orange:#d97706;--green:#2f855a}
*{box-sizing:border-box}body{margin:0;background:#fff}main{max-width:1920px;margin:auto;padding:30px 44px 24px}header{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:16px;flex-wrap:wrap}h1{font-size:clamp(26px,2.45vw,44px);line-height:1.13;margin:0;font-weight:700;letter-spacing:-.02em;background:var(--lime);padding:10px 17px;border-radius:10px;max-width:calc(100% - 230px)}.clock{font-size:clamp(25px,2.3vw,39px);font-variant-numeric:tabular-nums;white-space:nowrap;padding-top:8px}.clock i{font-style:italic}.clock output{display:inline-block;min-width:7ch;text-align:right}
.tabs{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 13px}.tabs button,.controlbar button,.controlbar select{font:inherit;background:#fff;border:1px solid #b8c1c8;border-radius:7px;padding:8px 13px;min-height:38px;cursor:pointer}.tabs button[aria-selected=true]{background:var(--ink);color:#fff;border-color:var(--ink)}
.controlbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:15px;padding:9px 12px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.controlbar label{font-size:14px;color:var(--muted)}.controlbar input[type=range]{flex:1;min-width:180px}.controlbar select{min-height:34px;padding:6px 10px}.status{font-size:14px;color:var(--muted);font-variant-numeric:tabular-nums}
.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.panel{border:1px solid var(--line);border-radius:9px;background:#fff;overflow:hidden}.panel-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;padding:13px 17px 8px}.panel-head h2{font-size:clamp(21px,1.7vw,31px);margin:0;line-height:1.25}.panel-head span{font-size:13px;color:var(--muted)}.stage{position:relative;height:min(58vh,650px);min-height:410px}.stage canvas{width:100%;height:100%;display:block;touch-action:none;cursor:grab}.stage canvas.dragging{cursor:grabbing}.hint{position:absolute;left:15px;top:12px;background:rgba(255,255,255,.9);padding:6px 9px;border-radius:5px;color:var(--muted);font-size:13px;pointer-events:none}.legend{display:flex;align-items:center;gap:18px;flex-wrap:wrap;padding:12px 16px;border-top:1px solid var(--line);font-size:14px;min-height:49px}.legend span{display:inline-flex;align-items:center;white-space:nowrap}.swatch{width:25px;height:0;border-top:3px solid currentColor;margin-right:7px}.swatch.dash{border-top-style:dashed}.dot{width:11px;height:11px;border:2px solid #5d6871;border-radius:50%;margin-right:7px;background:#fff}.ring{width:14px;height:14px;border:2px solid #65707a;border-radius:50%;margin-right:7px}
.footer{display:flex;justify-content:space-between;gap:14px;align-items:center;border-top:1px solid var(--line);margin-top:16px;padding-top:12px;color:#5a646d;font-size:13px}.footer button{background:#fff;border:1px solid #b8c1c8;border-radius:6px;padding:6px 10px;cursor:pointer}dialog{border:1px solid var(--line);border-radius:8px;max-width:650px;line-height:1.55}dialog::backdrop{background:#17191c55}@media(max-width:900px){main{padding:20px 16px}.panels{grid-template-columns:1fr}.stage{height:56vh;min-height:360px}h1{max-width:100%}.clock{padding-top:0}}@media(max-width:560px){.stage{min-height:310px;height:52vh}.panel-head{padding-inline:12px}.legend{gap:11px;font-size:12px}.controlbar{gap:8px}}
</style></head><body><main>
<header><h1>Baoding Acoustic Source Localization and Trajectory Tracking</h1><div class="clock"><i>t</i> = <output id="time">0.00 s</output></div></header>
<div class="tabs" role="tablist"><button data-scenario="single" aria-selected="true">Single source</button><button data-scenario="dual" aria-selected="false">Two sources</button><button data-scenario="three_source" aria-selected="false">Three sources</button></div>
<div class="controlbar"><button id="play">Play</button><button id="reset">Reset view</button><label for="frame">Sample</label><input id="frame" type="range" min="0" max="1" value="0" step="1"><span class="status" id="sample-status">1 / 1</span><label for="speed">Speed</label><select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><label for="preset">View</label><select id="preset"><option value="top" selected>Top view</option><option value="3d">3D perspective</option></select></div>
<section class="panels"><article class="panel"><div class="panel-head"><h2>Reference trajectory</h2><span>Acoustic localization</span></div><div class="stage"><canvas id="viewLeft" aria-label="Reference source trajectories"></canvas><div class="hint">Drag to rotate · scroll to zoom</div></div><div class="legend" id="legend-left"></div></article><article class="panel"><div class="panel-head"><h2>APCE tracking</h2><span>Estimate and uncertainty</span></div><div class="stage"><canvas id="viewRight" aria-label="APCE source trajectories"></canvas><div class="hint">Drag to rotate · scroll to zoom</div></div><div class="legend" id="legend-right"></div></article></section>
<div class="footer"><span>Coordinates are shown in metres; time is relative to the first recorded sample.</span><button id="help">About this view</button></div><span id="source" hidden></span>
<dialog id="about"><h3>Baoding tracking viewer</h3><p>This interactive view reuses the saved single-, two- and three-source trajectory table used for the supplementary tracking movie.</p><p>The left panel shows the reference trajectory and recorded sparse samples. The right panel shows the APCE trajectory and its saved marginal 90% position-interval width. Use the top-view / 3D switch and drag either panel to inspect the geometry.</p><button id="close">Close</button></dialog>
<script id="payload" type="application/json">''' + data_json + r'''</script>
<script>
const DATA=JSON.parse(document.getElementById('payload').textContent),rows=DATA.rows,byScenario={};for(const r of rows)(byScenario[r.scenario]??=[]).push(r);for(const k in byScenario)byScenario[k].sort((a,b)=>a.target-b.target||a.time_s-b.time_s);
const panels=[{id:'viewLeft',mode:'reference',legend:'legend-left'},{id:'viewRight',mode:'apce',legend:'legend-right'}];const S={scenario:'single',frame:0,playing:false,speed:1,preset:'top',yaw:-.72,pitch:.45,zoom:1,drag:null};const palette=['#1769aa','#d97706','#2f855a'];
const fmt=v=>Math.abs(v)<1e-8?'0':Number(v.toFixed(2)).toString();const targets=()=>[...new Set((byScenario[S.scenario]||[]).map(r=>r.target))].sort((a,b)=>a-b);const series=t=>(byScenario[S.scenario]||[]).filter(r=>r.target===t);const allRows=()=>byScenario[S.scenario]||[];
function bounds(){const a=allRows().flatMap(r=>[r.truth,r.apce]);if(!a.length)return 100;const xs=a.map(p=>p[0]),ys=a.map(p=>p[1]),zs=a.map(p=>p[2]);return Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),Math.max(...zs)-Math.min(...zs),100)*.58}
function center(){const a=allRows().flatMap(r=>[r.truth,r.apce]);if(!a.length)return [0,0,0];return [0,1,2].map(k=>(Math.min(...a.map(p=>p[k]))+Math.max(...a.map(p=>p[k])))/2)}
function project(p,c,range){let [x,y,z]=p;const ce=center();x-=ce[0];y-=ce[1];z-=ce[2];if(S.preset==='top'){const sc=Math.min(c.width,c.height)*.42;return [c.width/2+x*sc/range,c.height/2-y*sc/range,0];}const cy=Math.cos(S.yaw),sy=Math.sin(S.yaw),x1=cy*x-sy*y,y1=sy*x+cy*y,cp=Math.cos(S.pitch),sp=Math.sin(S.pitch),Y=cp*z-sp*y1,depth=sp*z+cp*y1,scale=Math.min(c.width,c.height)*.43*S.zoom/range;return [c.width/2+x1*scale,c.height/2-Y*scale,depth]}
function setupCanvas(c){const dpr=devicePixelRatio||1,rect=c.getBoundingClientRect();if(c.width!==Math.round(rect.width*dpr)||c.height!==Math.round(rect.height*dpr)){c.width=Math.round(rect.width*dpr);c.height=Math.round(rect.height*dpr)}return [c.getContext('2d'),c.width,c.height]}
function drawGrid(ctx,c,range){ctx.save();ctx.strokeStyle='#e5e9eb';ctx.lineWidth=1;for(let k=-2;k<=2;k++){const a=project([k*range*.25,-range*.5,0],c,range),b=project([k*range*.25,range*.5,0],c,range),d=project([-range*.5,k*range*.25,0],c,range),e=project([range*.5,k*range*.25,0],c,range);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.beginPath();ctx.moveTo(d[0],d[1]);ctx.lineTo(e[0],e[1]);ctx.stroke()}ctx.restore()}
function drawAxes(ctx,c,range){const o=project([0,0,0],c,range),axes=[[[range*.42,0,0],'east','#8b3a3a'],[[0,range*.42,0],'north','#2f6f3e'],[[0,0,range*.42],'up','#315a8a']];ctx.save();ctx.font='15px Arial';for(const [p,label,col] of axes){const q=project(p,c,range);ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(o[0],o[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.fillText(label,q[0]+6,q[1]-5)}ctx.restore()}
function path(ctx,c,pts,color,dash,width=3){if(pts.length<2)return;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash?[10,8]:[]);ctx.beginPath();pts.forEach((p,i)=>{const q=project(p,c,bounds());i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])});ctx.stroke();ctx.restore()}
function marker(ctx,c,p,color,r,open=false){const q=project(p,c,bounds());ctx.save();ctx.beginPath();ctx.arc(q[0],q[1],r,0,Math.PI*2);ctx.fillStyle=open?'#fff':color;ctx.fill();ctx.strokeStyle=color;ctx.lineWidth=open?2:1.5;ctx.stroke();ctx.restore()}
function drawPane(c,mode){const [ctx,W,H]=setupCanvas(c);ctx.clearRect(0,0,W,H);const range=bounds();drawGrid(ctx,c,range);if(S.preset==='3d')drawAxes(ctx,c,range);const cur=S.frame,ts=targets();for(let i=0;i<ts.length;i++){const dat=series(ts[i]),truth=dat.map(r=>r.truth),apce=dat.map(r=>r.apce),col=palette[i%palette.length],selected=mode==='reference'?truth:apce;path(ctx,c,selected,col,false,3);if(mode==='reference'){for(let j=0;j<dat.length;j+=Math.max(1,Math.floor(dat.length/16)))marker(ctx,c,dat[j].truth,col,4,true);const r=dat[Math.min(cur,dat.length-1)];if(r)marker(ctx,c,r.truth,col,7,false)}else{path(ctx,c,truth,col,true,2);for(let j=0;j<dat.length;j+=Math.max(1,Math.floor(dat.length/16)))marker(ctx,c,dat[j].apce,col,4,true);const r=dat[Math.min(cur,dat.length-1)];if(r){marker(ctx,c,r.apce,col,7,true);const q=project(r.apce,c,range),rad=Math.max(6,Math.min(48,r.width_m*.5*Math.min(c.width,c.height)*.42/range));ctx.save();ctx.strokeStyle=col;ctx.globalAlpha=.45;ctx.lineWidth=2;ctx.beginPath();ctx.arc(q[0],q[1],rad,0,Math.PI*2);ctx.stroke();ctx.restore()}}}}
function update(){const n=Math.max(1,...targets().map(t=>series(t).length));document.getElementById('frame').max=Math.max(0,n-1);S.frame=Math.min(S.frame,n-1);document.getElementById('frame').value=S.frame;document.getElementById('sample-status').textContent=`${S.frame+1} / ${n}`;const first=series(targets()[0]||1)[0],cur=series(targets()[0]||1)[S.frame];document.getElementById('time').textContent=cur&&first?`${(cur.time_s-first.time_s).toFixed(2)} s`:'0.00 s';for(const p of panels){drawPane(document.getElementById(p.id),p.mode);const l=document.getElementById(p.legend);l.innerHTML=targets().map((t,i)=>`<span><b class="swatch" style="color:${palette[i%palette.length]}"></b>Target ${t}</span>`).join('')+(p.mode==='reference'?'<span><b class="dot"></b>recorded sample</span>':'<span><b class="swatch dash" style="color:#58616a"></b>reference</span><span><b class="ring"></b>uncertainty</span>')}}
function selectScenario(name){S.scenario=name;S.frame=0;document.querySelectorAll('[data-scenario]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.scenario===name)));update()}
function step(){if(!S.playing)return;const n=Number(document.getElementById('frame').value),max=Number(document.getElementById('frame').max);if(n>=max){S.playing=false;document.getElementById('play').textContent='Play';return}S.frame=n+1;document.getElementById('frame').value=S.frame;update();setTimeout(step,Math.max(24,180/S.speed))}
document.querySelectorAll('[data-scenario]').forEach(b=>b.onclick=()=>selectScenario(b.dataset.scenario));document.getElementById('frame').oninput=e=>{S.frame=Number(e.target.value);update()};document.getElementById('speed').onchange=e=>S.speed=Number(e.target.value);document.getElementById('preset').onchange=e=>{S.preset=e.target.value;update()};document.getElementById('play').onclick=()=>{S.playing=!S.playing;document.getElementById('play').textContent=S.playing?'Pause':'Play';if(S.playing)step()};document.getElementById('reset').onclick=()=>{S.yaw=-.72;S.pitch=.45;S.zoom=1;S.preset='top';document.getElementById('preset').value='top';update()};
for(const p of panels){const c=document.getElementById(p.id);c.addEventListener('pointerdown',e=>{c.setPointerCapture(e.pointerId);S.drag=[e.clientX,e.clientY];c.classList.add('dragging')});c.addEventListener('pointermove',e=>{if(!S.drag)return;S.yaw-=(e.clientX-S.drag[0])*.008;S.pitch=Math.max(-1.45,Math.min(1.45,S.pitch+(e.clientY-S.drag[1])*.008));S.preset='3d';document.getElementById('preset').value='3d';S.drag=[e.clientX,e.clientY];update()});['pointerup','pointercancel'].forEach(t=>c.addEventListener(t,()=>{S.drag=null;c.classList.remove('dragging')}));c.addEventListener('wheel',e=>{e.preventDefault();S.zoom=Math.max(.55,Math.min(2.5,S.zoom*Math.exp(-e.deltaY*.001)));update()},{passive:false})}
document.getElementById('source').textContent=DATA.source+' · SHA-256 '+DATA.source_sha256.slice(0,16)+'…';document.getElementById('help').onclick=()=>document.getElementById('about').showModal();document.getElementById('close').onclick=()=>document.getElementById('about').close();update();
</script></main></body></html>'''
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    inp = Path(args.input)
    rows = load_rows(inp)
    sha = hashlib.sha256(inp.read_bytes()).hexdigest()
    html = build_html(rows, str(inp), sha)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(json.dumps({"output": str(out), "rows": len(rows), "sha256": sha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
