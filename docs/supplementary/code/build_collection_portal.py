"""Build the standalone purple-and-white homepage without editing scientific assets.

Run: python code/build_collection_portal.py (from docs/supplementary/)
The CSS and JavaScript are embedded so index.html works from a local folder.
Palette: purple (#63065f) and white.
"""
from pathlib import Path
import hashlib
import html
import json

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent
ESC = lambda value: html.escape(str(value), quote=True)


def span(zh, en, tag="span", attrs=""):
    return f'<{tag} data-zh="{ESC(zh)}" data-en="{ESC(en)}" {attrs}>{ESC(zh)}</{tag}>'


def build():
    manifest_path = ROOT / "catalog.json" if (ROOT / "catalog.json").is_file() else ROOT / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_entries = manifest.get("entries", manifest.get("movies", []))
    entries = {str(e.get("movie", e.get("number"))): e for e in raw_entries if e.get("movie", e.get("number")) is not None}
    titles = [
        ("三个经典不确定方程", "波、弹簧振动与热传导", "Wave, spring vibration and heat conduction", "dynamics"),
        ("五个 ODE 系统", "化学反应、药代动力学、摆、FHN 与 Robertson 动力学", "Chemical reaction, pharmacokinetics, pendulum, FHN and Robertson kinetics", "dynamics"),
        ("Lorenz–96 动力学", "稀疏同化与观测中断后的演化", "Sparse assimilation and evolution after observation blackout", "dynamics"),
        ("Kuramoto–Sivashinsky 方程", "一维波形重建与观测中断后的预测", "One-dimensional waveform reconstruction and post-blackout forecast", "dynamics"),
        ("Kolmogorov 湍流", "速度场、涡量与观测中断后的预测", "Velocity fields, vorticity and post-blackout forecast", "flow"),
        ("VIV–PIV 实验", "五个约化速度工况下的稀疏观测与场重建", "Sparse observations and field reconstruction across five reduced velocities", "flow"),
        ("MeshRIR 三维声场", "三维声场观测与重建", "Three-dimensional acoustic field observations and reconstruction", "acoustics"),
        ("声源定位与轨迹跟踪", "单源、双源和三源的定位与轨迹跟踪", "Single-, two- and three-source localization and trajectory tracking", "acoustics"),
    ]
    movies = []
    for number, (zh, desc_zh, desc_en, group) in enumerate(titles, 1):
        e = entries[str(number)]
        en = e.get("title_en", e.get("title", "").split(": ", 1)[-1])
        poster = e.get("poster", f"assets/posters/movie-{number}.png")
        delivery_file = e.get("delivery_file", e.get("file"))
        assert delivery_file and (ROOT / poster).is_file() and (ROOT / delivery_file).is_file()
        duration = f'{e["duration_s"]:.2f}'.rstrip("0").rstrip(".")
        movie = dict(number=number, zh=zh, en=en, description_zh=desc_zh, description_en=desc_en, group=group, file=delivery_file, poster=poster, duration=duration)
        if number == 7:
            movie['variants'] = e['variants']
            assert all((ROOT / v['file']).is_file() and (ROOT / v['poster']).is_file() for v in movie['variants'])
            movie['file'] = movie['variants'][0]['file']
            movie['poster'] = movie['variants'][0]['poster']
            revision = movie['variants'][0].get('revision')
            if revision:
                movie['playback_src'] = movie['file'] + '?v=' + revision
                movie['poster'] += '?v=' + revision
            movie['description_zh'] = '等值面与切片展示，可在同一播放时刻切换。'
            movie['description_en'] = 'Switch between isosurfaces and slices at the same playback time.'
        movies.append(movie)

    components = [
        ("1D", "稀疏观测与中断", "Sparse observations & blackout", "查看实际传感器布局、观测时刻和中断后的预测。", "Inspect sensor layouts, observation times and forecasts after blackout.", "Supplementary_Sensor_Blackout_Explorer.html"),
        ("PIV", "VIV–PIV 五工况", "VIV–PIV: five regimes", "联动查看稀疏 PIV 测量、重建场与探针频谱。", "Explore sparse PIV measurements, reconstructed fields and probe spectra.", "VIV_PIV_5regimes_interactive.html"),
        ("3D", "MeshRIR 三维声场", "MeshRIR 3D acoustic field", "切换等值面与切片，同步旋转观测、APCE与参考声场。", "Switch isosurfaces and slices; rotate observations, APCE and reference together.", "MeshRIR_3D_interactive.html"),
        ("XYZ", "声源定位与轨迹跟踪", "Source localization & tracking", "查看单源、双源和三源的三维轨迹与不确定度。", "Explore 3D trajectories and uncertainty for one, two and three sources.", "Baoding_Tracking_3D_interactive.html"),
    ]
    diagnostics = [
        ("APCE 机制检查器", "APCE mechanism inspector", "候选权重、熵、α 估计与配对预测。", "Candidate weights, entropy, α estimates and paired forecasts.", "Supplementary_APCE_Inspector.html"),
        ("全运行校准图谱", "Full-run calibration", "参数误差、预测误差、覆盖率与区间宽度。", "Parameter error, forecast error, coverage and interval width.", "Supplementary_FullRun_Calibration_Atlas.html"),
        ("运行代价与预测效果", "Runtime & forecast skill", "比较各案例中记录的运行时间与预测误差。", "Compare recorded runtime and forecast error within each case.", "Supplementary_Runtime_Pareto.html"),
        ("跨案例预测来源", "Cross-case forecast sources", "比较 18 个案例在不同观测频率下的预测来源。", "Compare forecast sources across 18 cases and observation frequencies.", "Supplementary_CrossCase_ForecastSource_Atlas.html"),
    ]
    assert all((ROOT / c[-1]).is_file() for c in components + diagnostics)
    css = (CODE / "collection_portal.css").read_text(encoding="utf-8")
    js = (CODE / "collection_portal.js").read_text(encoding="utf-8")
    out = [f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Supplementary movies and interactive experiments for paired cumulative predictive evidence (PCE) and adaptive PCE."><title>Supplementary Movies</title><style>\n{css}\n</style></head><body>']
    out.append('<a class="skip" href="#movies">'+span("跳到视频", "Skip to videos")+'</a>')
    out.append('<header class="masthead"><div class="container masthead-inner"><a class="brand" href="#top"><span class="brand-mark" aria-hidden="true">PCE</span><div>'+span("补充材料", "Supplements", attrs='class="brand-name"')+span("视频与交互演示", "Movies & interactives", attrs='class="brand-caption"')+'</div></a><nav class="navigation" data-aria-zh="页面导航" data-aria-en="Page navigation">')
    for anchor, zh, en in [("movies", "视频案例", "Movies"), ("interactives", "交互组件", "Interactives"), ("diagnostics", "校准与来源", "Calibration")]:
        out.append(f'<a href="#{anchor}">'+span(zh,en)+'</a>')
    out.append('</nav><div class="language" id="lang-switcher" role="group" data-aria-zh="页面语言" data-aria-en="Page language"><button type="button" data-language="zh" aria-pressed="false">'+span("中文","Chinese")+'</button><button type="button" data-language="en" aria-pressed="true">English</button></div></div></header>')
    out.append('<main id="top"><div class="container"><section class="intro"><div><div class="intro-label">'+span("补充材料", "SUPPLEMENTARY MATERIALS")+'</div>'+span("动态重建与不确定性", "Dynamic reconstruction & uncertainty", "h1")+span("从稀疏观测到中断预测，浏览动力系统、湍流与声场实验。", "Explore sparse reconstruction and post-blackout forecasting across dynamical systems, turbulent flow and acoustics.", "p", 'class="intro-lead"')+'<div class="collection-facts"><span><b>8</b>'+span("部补充视频", "supplementary movies")+'</span><span><b>8</b>'+span("个交互组件", "interactive components")+'</span><a href="#interactives">'+span("探索交互组件 →", "Explore interactives →")+'</a></div></div><aside class="intro-aside">'+span("本页汇集 APCE 方法的可复现实验演示：视频展示时间演化，交互组件用于检查观测、重建和不确定度。", "A compact, reproducible view of APCE: videos show temporal evolution; interactive components expose observations, reconstructions and uncertainty.", "p")+'<a href="https://github.com/BrianZhu1999/PCE-APCE" target="_blank" rel="noreferrer">'+span("访问 APCE 项目 ↗", "Open the APCE repository ↗")+'</a></aside></section>')
    out.append('<section class="section" id="movies"><div class="section-heading">'+span("视频案例", "Supplementary movies", "h2")+span("可在卡片内直接播放；放大后点击「关闭播放器」或按 Esc 返回。", "Play directly in a card, or enlarge it. Close the player or press Esc to return.", "p")+'</div><div class="filters" role="group" data-aria-zh="筛选视频类别" data-aria-en="Filter movie categories">')
    for key,zh,en in [("all","全部视频","All movies"),("dynamics","动力系统","Dynamical systems"),("flow","湍流与流动","Turbulent flow"),("acoustics","声场与跟踪","Acoustics & tracking")]:
        out.append(f'<button type="button" data-filter="{key}" aria-pressed="{str(key=="all").lower()}">'+span(zh,en)+'</button>')
    out.append('</div><span class="visually-hidden" id="filter-status" aria-live="polite"></span><div class="movie-grid">')
    for index, m in enumerate(movies):
        n=m['number']
        out.append(f'<article class="movie-card" data-group="{m["group"]}" id="movie-{n}"><div class="media"><video class="inline-video" controls playsinline preload="none" src="{ESC(m.get("playback_src",m["file"]))}" poster="{ESC(m["poster"])}"></video><p class="video-error" role="status" hidden>'+span("视频暂时无法播放，请确认视频文件仍在本目录。", "The video could not be played. Check that its file is still in this folder.")+'</p></div><div class="movie-body"><div class="movie-meta">'+span(f"补充视频 {n}",f"Supplementary Movie {n}",attrs='class="movie-id"')+f'<span>{m["duration"]} s</span></div>'+span(m['zh'],m['en'],"h3")+span(m['description_zh'],m['description_en'],"p",'class="movie-description"')+'<div class="movie-actions">'+f'<button type="button" class="action primary" data-play-movie="{index}">播放预览</button>'+f'<button type="button" class="action" data-open-movie="{index}" aria-haspopup="dialog">'+span("放大播放 ↗", "Enlarge player ↗")+'</button></div></div></article>')
    out.append('</div></section></div><section class="interactive-section" id="interactives"><div class="container"><div class="section-heading">'+span("核心交互组件", "Interactive experiments", "h2")+span("选择一个案例，自由查看观测、重建和时空演化。", "Choose an experiment to explore observations, reconstruction and evolution.", "p")+'</div><div class="component-grid">')
    for symbol,zh,en,dzh,den,file in components:
        out.append(f'<a class="component" href="{file}"><span class="component-symbol" aria-hidden="true">{symbol}</span><div>'+span(zh,en,"h3")+span(dzh,den,"p")+'</div><span class="component-arrow" aria-hidden="true">→</span></a>')
    out.append('</div></div></section><div class="container"><section class="section" id="diagnostics"><div class="section-heading">'+span("校准与来源", "Calibration & provenance", "h2")+span("从方法机制到跨案例对照，核查已有实验结果。", "Inspect existing results, from method mechanisms to comparisons across cases.", "p")+'</div><div class="diagnostic-grid">')
    for zh,en,dzh,den,file in diagnostics:
        out.append('<article class="diagnostic">'+span(zh,en,"h3")+span(dzh,den,"p")+f'<a href="{file}">'+span("查看组件 →", "Open component →")+'</a></article>')
    out.append('</div></section><footer class="footer">'+span("科研补充材料", "Research supplements")+'<div class="footer-links"><a href="README.md">'+span("文件说明", "README")+'</a><a href="catalog.json">'+span("来源与校验", "Sources & verification")+'</a><a href="#top">'+span("返回顶部 ↑", "Back to top ↑")+'</a></div></footer></div></main>')
    out.append('<dialog class="player-dialog" id="video-dialog" aria-labelledby="player-title"><div class="player-shell"><header class="player-top"><div><span class="movie-id" id="player-number"></span><h2 id="player-title"></h2></div><button class="close-player" type="button" id="close-player" autofocus>'+span("✕ 关闭播放器", "✕ Close player")+'</button></header><div class="player-stage"><video class="dialog-video" id="dialog-video" controls playsinline preload="none"></video><p class="video-error" id="player-error" role="status" hidden>'+span("暂时无法播放，可下载视频后查看。", "Unable to play this video. You can download it to watch locally.")+'</p></div><footer class="player-bottom"><div class="player-pagination"><button type="button" id="previous-movie">'+span("← 上一部", "← Previous")+'</button><span class="player-counter" id="player-counter"></span><button type="button" id="next-movie">'+span("下一部 →", "Next →")+'</button></div>'+span("按 Esc 或点击外侧空白关闭；返回后视频保持暂停。", "Press Esc or click outside to close. Playback stays paused on return.","p",'class="player-help"')+'<a id="download-movie" download>'+span("下载视频", "Download video")+'</a></footer></div></dialog>')
    out.append('<script type="application/json" id="movie-data">'+json.dumps(movies,ensure_ascii=False).replace('</','<\/')+'</script>')
    out.append('<script>\n'+js+'\n</script></body></html>\n')
    (ROOT/'index.html').write_text('\n'.join(out),encoding='utf-8',newline='\n')
    print(json.dumps({'file':'index.html','bytes':(ROOT/'index.html').stat().st_size,'movies':len(movies),'interactives':len(components)+len(diagnostics),'sha256':hashlib.sha256((ROOT/'index.html').read_bytes()).hexdigest()},ensure_ascii=False))


if __name__=='__main__':
    build()
