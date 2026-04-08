"""CyberForge Web UI — interactive AI model optimization workflow.

Mounted inside the main FastAPI app so everything runs as ONE server process.
Launch:  uvicorn apps.api.main:app --reload

All /action/* routes call the API in-process via httpx ASGITransport (zero network).
"""

from __future__ import annotations

import html as _html
import json as _json
import os
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

# Reference to the main API app — set by build_ui_app()
_api_app: FastAPI | None = None


def build_ui_app(api_app: FastAPI) -> FastAPI:
    """Create the UI FastAPI sub-app wired to call *api_app* in-process."""
    global _api_app
    _api_app = api_app
    return app


# The UI sub-application (routes registered at module level below)
app = FastAPI(title="CyberForge UI", version="0.1.0", docs_url=None, redoc_url=None)


# ── Helpers ──────────────────────────────────────────────────────

async def _api(path: str, method: str = "GET", json: dict | None = None) -> Any:
    """Call an API route in-process via ASGI transport (no network).

    The base_url is a dummy required by httpx — no real network call is made.
    We extract error details from the JSON body so the ASGI dummy URL never
    leaks into user-facing messages.
    """
    if _api_app is None:
        return {"_error": "UI not wired to API app — call build_ui_app() first"}
    transport = httpx.ASGITransport(app=_api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://asgi-internal", timeout=httpx.Timeout(600.0)) as client:
        try:
            if method == "POST":
                resp = await client.post(path, json=json or {})
            elif method == "DELETE":
                resp = await client.delete(path)
            else:
                resp = await client.get(path)
            # On error responses, try to extract the real error from the JSON body
            # instead of letting raise_for_status() produce a message with the
            # dummy ASGI base URL.
            if resp.status_code >= 400:
                try:
                    body = resp.json()
                    detail = body.get("detail") or body.get("error") or body.get("message")
                    if detail:
                        return {"_error": f"[{resp.status_code}] {detail}"}
                except Exception:
                    pass
                return {"_error": f"[{resp.status_code}] {path} failed"}
            return resp.json()
        except Exception as exc:
            # Strip any reference to the dummy ASGI base URL from error messages
            msg = str(exc).replace("http://asgi-internal", "")
            return {"_error": msg}


def _err(data: Any) -> str | None:
    if isinstance(data, dict) and "_error" in data:
        escaped = _html.escape(str(data["_error"]))
        return f'<div class="card" style="border-color:#f85149;color:#f85149">{escaped}</div>'
    return None


# ── Shared page chrome ───────────────────────────────────────────

_CSS = """\
:root{--bg:#0d1117;--fg:#c9d1d9;--accent:#58a6ff;--card:#161b22;--border:#30363d;--green:#238636;--yellow:#9e6a03;--red:#da3633;--purple:#8b5cf6;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial;background:var(--bg);color:var(--fg);line-height:1.6;}
a{color:var(--accent);text-decoration:none;}a:hover{text-decoration:underline;}
nav{background:var(--card);border-bottom:1px solid var(--border);padding:.75rem 2rem;display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;}
nav .brand{font-weight:700;font-size:1.1rem;color:var(--accent);}
nav a{color:var(--fg);font-size:.9rem;}nav a:hover{color:var(--accent);}
.container{max-width:1200px;margin:0 auto;padding:2rem;}
h1{color:var(--accent);margin-bottom:.5rem;}
h2{color:var(--accent);margin:1.5rem 0 .75rem;border-bottom:1px solid var(--border);padding-bottom:.25rem;}
h3{color:var(--fg);margin:1rem 0 .5rem;}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:1.25rem;margin:.75rem 0;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;}
.kpi{text-align:center;}.kpi .value{font-size:2rem;font-weight:700;color:var(--accent);}.kpi .label{font-size:.85rem;color:#8b949e;}
table{width:100%;border-collapse:collapse;margin:.5rem 0;}
th,td{text-align:left;padding:.5rem 1rem;border-bottom:1px solid var(--border);}
th{color:var(--accent);font-weight:600;}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-size:.8rem;background:var(--border);color:var(--fg);}
.badge-ok{background:#238636;color:#fff;}.badge-warn{background:#9e6a03;color:#fff;}.badge-err{background:#da3633;color:#fff;}
.badge-purple{background:var(--purple);color:#fff;}
footer{margin-top:3rem;text-align:center;color:#484f58;font-size:.8rem;padding:1rem;}
/* Buttons */
.btn{display:inline-block;padding:.5rem 1.25rem;border-radius:6px;border:1px solid var(--border);cursor:pointer;font-size:.9rem;font-weight:600;transition:all .15s;text-align:center;}
.btn:hover{filter:brightness(1.2);}
.btn:disabled{opacity:.4;cursor:not-allowed;}
.btn-primary{background:var(--accent);color:#0d1117;border-color:var(--accent);}
.btn-green{background:var(--green);color:#fff;border-color:var(--green);}
.btn-red{background:var(--red);color:#fff;border-color:var(--red);}
.btn-purple{background:var(--purple);color:#fff;border-color:var(--purple);}
.btn-outline{background:transparent;color:var(--accent);border-color:var(--accent);}
.btn-sm{padding:.3rem .75rem;font-size:.8rem;}
/* Forms */
select,input[type=text],input[type=number],textarea{background:var(--card);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:.5rem .75rem;font-size:.9rem;width:100%;}
select:focus,input:focus,textarea:focus{outline:none;border-color:var(--accent);}
label{display:block;margin-bottom:.25rem;font-size:.85rem;color:#8b949e;font-weight:600;}
.form-group{margin-bottom:1rem;}
/* Radio/Check cards */
.mode-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.75rem;margin:1rem 0;}
.mode-card{background:var(--card);border:2px solid var(--border);border-radius:8px;padding:1rem;cursor:pointer;text-align:center;transition:all .15s;}
.mode-card:hover{border-color:var(--accent);}
.mode-card.selected{border-color:var(--accent);background:#1a2332;}
.mode-card .icon{font-size:2rem;margin-bottom:.5rem;}
.mode-card .title{font-weight:700;font-size:1rem;}
.mode-card .desc{font-size:.8rem;color:#8b949e;margin-top:.25rem;}
/* Stepper */
.stepper{display:flex;gap:.5rem;margin:1.5rem 0;flex-wrap:wrap;}
.step-pill{padding:.4rem 1rem;border-radius:20px;font-size:.8rem;font-weight:600;background:var(--border);color:#8b949e;transition:all .2s;}
.step-pill.active{background:var(--accent);color:#0d1117;}
.step-pill.done{background:var(--green);color:#fff;}
/* Result box */
.result-box{background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:1rem;margin:1rem 0;display:none;max-height:500px;overflow-y:auto;}
.result-box.visible{display:block;}
/* Loading */
.spinner{display:inline-block;width:16px;height:16px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite;margin-right:.5rem;}
@keyframes spin{to{transform:rotate(360deg)}}
/* Comparison */
.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}
@media(max-width:768px){.compare-grid{grid-template-columns:1fr;}}
/* Chat */
.chat-box{background:#0d1117;border:1px solid var(--border);border-radius:8px;height:400px;overflow-y:auto;padding:1rem;margin:1rem 0;}
.chat-msg{margin:.5rem 0;padding:.75rem;border-radius:8px;max-width:85%;}
.chat-user{background:#1a2332;margin-left:auto;border:1px solid var(--accent);}
.chat-bot{background:var(--card);border:1px solid var(--border);}
.chat-input-row{display:flex;gap:.5rem;}
.chat-input-row input{flex:1;}
/* Progress bar */
.progress-bar{height:6px;background:var(--border);border-radius:3px;overflow:hidden;margin:.5rem 0;}
.progress-bar .fill{height:100%;background:var(--accent);transition:width .3s;}
/* Model select card */
.model-select{display:flex;align-items:center;gap:1rem;padding:.75rem;border:1px solid var(--border);border-radius:8px;margin:.5rem 0;cursor:pointer;transition:all .15s;}
.model-select:hover{border-color:var(--accent);}
.model-select.checked{border-color:var(--green);background:#0d2818;}
.model-select input[type=checkbox]{width:18px;height:18px;accent-color:var(--accent);}
"""

_NAV_LINKS = [
    ("/", "Dashboard"),
    ("/workflow", "&#9889; Workflow"),
    ("/hardware", "Hardware"),
    ("/models", "Models"),
    ("/recommend", "Recommend"),
    ("/bench", "Bench"),
    ("/optimize", "Optimize"),
    ("/cache", "Cache"),
    ("/cyber", "Cyber"),
    ("/jobs", "Jobs"),
    ("/chat", "Chat"),
    ("/docs", "API Docs &rarr;"),
]

_JS = """\
async function cyberPost(url, body={}) {
    const resp = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    return resp.json();
}
async function cyberGet(url) { const r = await fetch(url); return r.json(); }
function showLoading(id) { const el=document.getElementById(id); if(el){el.innerHTML='<span class="spinner"></span> Working...';el.classList.add('visible');} }
function hideLoading(id) { const el=document.getElementById(id); if(el) el.classList.remove('visible'); }
function showResult(id, html) { const el=document.getElementById(id); if(el){el.innerHTML=html;el.classList.add('visible');} }
function buildTable(headers, rows) {
    let h = '<table><thead><tr>' + headers.map(h=>'<th>'+h+'</th>').join('') + '</tr></thead><tbody>';
    rows.forEach(r => { h += '<tr>' + r.map(c=>'<td>'+c+'</td>').join('') + '</tr>'; });
    return h + '</tbody></table>';
}
function selectMode(el, mode) {
    document.querySelectorAll('.mode-card').forEach(c=>c.classList.remove('selected'));
    el.classList.add('selected');
    window._selectedMode = mode;
}
function toggleModelSelect(el) {
    const cb = el.querySelector('input[type=checkbox]');
    cb.checked = !cb.checked;
    el.classList.toggle('checked', cb.checked);
}
// ── Global selected-model state (persists across page navigations) ──
const CyberForge = {
    getSelected() {
        try { return JSON.parse(sessionStorage.getItem('cf_selected_model') || 'null'); } catch { return null; }
    },
    setSelected(model) {
        // model = {id, display_name, ollama_tag, hf_repo, benchmark_safe, requires_quantization, suggested_quant, fit_note}
        sessionStorage.setItem('cf_selected_model', JSON.stringify(model));
        window.dispatchEvent(new CustomEvent('cf:model-selected', {detail: model}));
    },
    clearSelected() {
        sessionStorage.removeItem('cf_selected_model');
    },
    getModelId() {
        const m = this.getSelected();
        if (!m) return '';
        return m.ollama_tag || m.hf_repo || m.id || '';
    },
    getMachineClass() {
        try { return JSON.parse(sessionStorage.getItem('cf_machine_class') || 'null'); } catch { return null; }
    },
    setMachineClass(mc) {
        sessionStorage.setItem('cf_machine_class', JSON.stringify(mc));
    },
    // ── Universal preflight: blocks unsafe actions per-backend ──
    async preflight(action, backend, modelParamsB, modelSizeMb) {
        try {
            const resp = await cyberPost('/action/preflight', {
                action: action, backend: backend,
                model_params_b: modelParamsB || 0, model_size_mb: modelSizeMb || 0
            });
            return resp;
        } catch(e) { return {allowed: true, reason: 'Preflight unavailable', category: 'unknown'}; }
    },
    showPreflightBanner(targetId, result) {
        const el = document.getElementById(targetId);
        if (!el) return;
        if (!result || result.allowed) { el.style.display = 'none'; return; }
        const color = 'var(--red)';
        el.innerHTML = '<span style="color:' + color + '">&#9888; ' + (result.reason||'Action blocked') + '</span>' +
            (result.suggestion ? '<br><span style="color:#8b949e;font-size:.85rem;">' + result.suggestion + '</span>' : '');
        el.style.borderColor = color;
        el.style.display = '';
    },
    getSystemCapability() {
        try { return JSON.parse(sessionStorage.getItem('cf_system_cap') || 'null'); } catch { return null; }
    },
    setSystemCapability(sc) {
        sessionStorage.setItem('cf_system_cap', JSON.stringify(sc));
    }
};
// Fetch machine class and system capability on every page load
(async function() {
    try {
        const mc = await cyberGet('/action/machine-class');
        if (mc && !mc._error) CyberForge.setMachineClass(mc);
    } catch {}
    try {
        const sc = await cyberGet('/action/system-capability');
        if (sc && !sc._error) CyberForge.setSystemCapability(sc);
    } catch {}
})();
"""


def _page(title: str, body: str, active: str = "/", extra_js: str = "") -> HTMLResponse:
    nav_items = ""
    for href, label in _NAV_LINKS:
        extra = ' style="color:var(--accent);font-weight:600"' if href == active else ""
        target = f' target="_blank"' if href == "/docs" else ""
        real_href = "/docs" if href == "/docs" else href
        nav_items += f'<a href="{real_href}"{extra}{target}>{label}</a>'
    html = f"""\
<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — CyberForge</title><style>{_CSS}</style></head><body>
<nav><span class="brand">&#x1f6e1; CyberForge</span>{nav_items}</nav>
<div class="container">{body}</div>
<footer>CyberForge v0.1.0 &mdash; Local AI Cyber Workbench</footer>
<script>{_JS}{extra_js}</script>
</body></html>"""
    return HTMLResponse(html)


# ══════════════════════════════════════════════════════════════════
# ACTION API ENDPOINTS — JSON responses for frontend AJAX calls
# ══════════════════════════════════════════════════════════════════

@app.get("/action/hardware")
async def action_hardware():
    return JSONResponse(await _api("/api/hardware/profile"))

@app.get("/action/machine-class")
async def action_machine_class():
    return JSONResponse(await _api("/api/hardware/machine-class"))

@app.get("/action/models")
async def action_models():
    return JSONResponse(await _api("/api/models/registry"))

@app.get("/action/ollama-models")
async def action_ollama_models():
    return JSONResponse(await _api("/api/serve/ollama/models"))


@app.get("/action/ollama-status")
async def action_ollama_status():
    return JSONResponse(await _api("/api/serve/ollama/status"))


@app.post("/action/ollama-start")
async def action_ollama_start():
    """Attempt to start Ollama serve as a background process."""
    import platform
    import shutil
    import subprocess
    exe = shutil.which("ollama")
    if not exe:
        return JSONResponse({"started": False, "error": "Ollama executable not found on PATH. Install from https://ollama.com/download"})
    try:
        kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen([exe, "serve"], **kwargs)
        # Wait briefly and check if it came up
        import asyncio
        await asyncio.sleep(2)
        status = await _api("/api/serve/ollama/status")
        if isinstance(status, dict) and status.get("available"):
            return JSONResponse({"started": True})
        return JSONResponse({"started": True, "warning": "Process launched but Ollama may still be starting up. Refresh in a few seconds."})
    except Exception as exc:
        return JSONResponse({"started": False, "error": str(exc)})

@app.get("/action/bench-cards")
async def action_bench_cards():
    return JSONResponse(await _api("/api/bench/cards"))

@app.get("/action/cache")
async def action_cache():
    return JSONResponse(await _api("/api/lifecycle/cache"))

@app.get("/action/saved")
async def action_saved():
    return JSONResponse(await _api("/api/lifecycle/saved"))

@app.get("/action/disk-usage")
async def action_disk():
    return JSONResponse(await _api("/api/lifecycle/disk-usage"))

@app.post("/action/recommend")
async def action_recommend(request: Request):
    body = await request.json()
    data = await _api("/api/recommend/", method="POST", json=body)
    return JSONResponse(data if isinstance(data, list) else [data] if isinstance(data, dict) and "_error" not in data else data)

@app.post("/action/self-test/{test_type}")
async def action_self_test(test_type: str):
    if test_type == "coding":
        return JSONResponse(await _api("/api/bench/coding/self-test", method="POST"))
    elif test_type == "cyber":
        return JSONResponse(await _api("/api/bench/cyber/self-test", method="POST"))
    elif test_type == "ids":
        return JSONResponse(await _api("/api/bench/ids/quick", method="POST"))
    return JSONResponse({"_error": f"Unknown test type: {test_type}"})

@app.post("/action/benchmark")
async def action_benchmark(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/bench/run", method="POST", json=body))

@app.post("/action/compare")
async def action_compare(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/bench/compare", method="POST", json=body))

@app.post("/action/quantize")
async def action_quantize(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/optimize/quantize", method="POST", json=body))

@app.get("/action/quantize-status")
async def action_quant_status():
    return JSONResponse(await _api("/api/optimize/quantize/status"))

@app.post("/action/quantize-compare")
async def action_quant_compare(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/optimize/quantize/compare", method="POST", json=body))

@app.post("/action/quantize-preflight")
async def action_quant_preflight(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/optimize/quantize/preflight", method="POST", json=body))

@app.post("/action/model-info")
async def action_model_info(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/optimize/model-info", method="POST", json=body))

@app.post("/action/cache-cleanup")
async def action_cache_cleanup():
    return JSONResponse(await _api("/api/lifecycle/cleanup/volatile", method="POST"))

@app.post("/action/cache-stale")
async def action_cache_stale():
    return JSONResponse(await _api("/api/lifecycle/cleanup/stale-cache", method="POST"))

@app.post("/action/discard")
async def action_discard(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/lifecycle/discard", method="POST", json=body))

@app.post("/action/save-artifact")
async def action_save(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/lifecycle/save", method="POST", json=body))

@app.post("/action/chat")
async def action_chat(request: Request):
    body = await request.json()
    # Chat needs longer timeout — model may need to load into VRAM on first call
    if _api_app is None:
        return JSONResponse({"_error": "UI not wired to API app"})
    transport = httpx.ASGITransport(app=_api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://asgi-internal", timeout=300) as client:
        try:
            resp = await client.post("/api/serve/ollama/chat", json=body)
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", "")
                except Exception:
                    detail = ""
                return JSONResponse({"_error": f"[{resp.status_code}] {detail or 'Chat request failed'}"})
            return JSONResponse(resp.json())
        except Exception as exc:
            msg = str(exc).replace("http://asgi-internal", "")
            return JSONResponse({"_error": msg})


@app.post("/action/ollama-pull")
async def action_ollama_pull(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/serve/ollama/pull", method="POST", json=body))


@app.post("/action/model-fit")
async def action_model_fit(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/serve/ollama/model-fit", method="POST", json=body))


@app.post("/action/web-discover")
async def action_web_discover(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/models/web-discover", method="POST", json=body))


@app.post("/action/model-select")
async def action_model_select(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/models/select", method="POST", json=body))


@app.post("/action/prune")
async def action_prune(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/optimize/prune", method="POST", json=body))


@app.get("/action/prune-methods")
async def action_prune_methods():
    return JSONResponse(await _api("/api/optimize/prune/methods"))


@app.post("/action/prune-suggest")
async def action_prune_suggest(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/optimize/prune/suggest", method="POST", json=body))


@app.post("/action/prune-iterative")
async def action_prune_iterative(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/optimize/prune/iterative", method="POST", json=body))


@app.get("/action/distill-methods")
async def action_distill_methods():
    return JSONResponse(await _api("/api/optimize/distill/methods"))


@app.post("/action/distill")
async def action_distill(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/optimize/distill", method="POST", json=body))


@app.post("/action/distill-suggest")
async def action_distill_suggest(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/optimize/distill/suggest", method="POST", json=body))


@app.get("/action/edit-operations")
async def action_edit_operations():
    return JSONResponse(await _api("/api/optimize/edit/operations"))


@app.post("/action/edit")
async def action_edit(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/optimize/edit", method="POST", json=body))


@app.post("/action/edit-suggest")
async def action_edit_suggest(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/optimize/edit/suggest", method="POST", json=body))


@app.get("/action/route-status")
async def action_route_status():
    return JSONResponse(await _api("/api/optimize/route/status"))


@app.post("/action/route")
async def action_route(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/optimize/route", method="POST", json=body))


@app.get("/action/system-capability")
async def action_system_capability():
    return JSONResponse(await _api("/api/hardware/system-capability"))


@app.post("/action/preflight")
async def action_preflight(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/hardware/preflight", method="POST", json=body))


# ══════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════

# ── Dashboard ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    health, hw, registry = await _api("/health"), await _api("/api/hardware/profile"), await _api("/api/models/registry")
    err = _err(health) or _err(hw) or _err(registry)
    if err:
        return _page("Dashboard", f"<h1>Dashboard</h1>{err}", "/")

    status = health.get("status", "unknown") if isinstance(health, dict) else "unknown"
    badge_cls = "badge-ok" if status == "ok" else "badge-err"
    cpu_name = hw.get("cpu_model", "?") if isinstance(hw, dict) else "?"
    gpu_list = hw.get("gpus", []) if isinstance(hw, dict) else []
    gpu0 = gpu_list[0] if gpu_list else {}
    gpu_name = gpu0.get("name", "N/A")
    ram_gb = round(hw.get("ram_total_mb", 0) / 1024, 1) if isinstance(hw, dict) else "?"
    vram = round(gpu0.get("vram_total_mb", 0) / 1024, 1) if gpu0 else "N/A"
    models_list = registry if isinstance(registry, list) else registry.get("models", [])
    model_count = len(models_list)

    body = f"""\
<h1>Dashboard</h1>
<div class="grid">
  <div class="card kpi"><div class="value"><span class="badge {badge_cls}">{status.upper()}</span></div><div class="label">API Status</div></div>
  <div class="card kpi"><div class="value">{cpu_name}</div><div class="label">CPU</div></div>
  <div class="card kpi"><div class="value">{ram_gb} GB</div><div class="label">RAM</div></div>
  <div class="card kpi"><div class="value">{gpu_name}</div><div class="label">GPU</div></div>
  <div class="card kpi"><div class="value">{vram} GB</div><div class="label">VRAM</div></div>
  <div class="card kpi"><div class="value">{model_count}</div><div class="label">Registry Models</div></div>
</div>

<h2>&#9889; Start Optimization Workflow</h2>
<div class="card" style="border-color:var(--accent);">
  <p style="font-size:1.1rem;margin-bottom:1rem;">Don't know which model to use? Follow the guided workflow to find and optimize the best model for your hardware and use case.</p>
  <a href="/workflow" class="btn btn-primary" style="text-decoration:none;">Launch Workflow &rarr;</a>
  <a href="/chat" class="btn btn-outline" style="text-decoration:none;margin-left:.5rem;">Ask CyberForge Chat</a>
</div>

<h2>Quick Links</h2>
<div class="grid">
  <div class="card"><a href="/hardware">&#128295; Hardware Profile</a> — system specs</div>
  <div class="card"><a href="/models">&#128230; Model Registry</a> — browse models</div>
  <div class="card"><a href="/recommend">&#127775; Recommendations</a> — model picks</div>
  <div class="card"><a href="/bench">&#9201; Benchmarks</a> — run & compare tests</div>
  <div class="card"><a href="/optimize">&#128640; Optimize</a> — quantize & compress</div>
  <div class="card"><a href="/cache">&#128451; Cache</a> — manage storage</div>
  <div class="card"><a href="/cyber">&#128737; Cyber Lab</a> — validate Sigma/YARA</div>
  <div class="card"><a href="/jobs">&#9881; Jobs</a> — background queue</div>
</div>"""
    return _page("Dashboard", body, "/")


# ── Workflow Wizard ──────────────────────────────────────────────

@app.get("/workflow", response_class=HTMLResponse)
async def workflow_page():
    hw = await _api("/api/hardware/profile")
    hw_err = _err(hw)
    if hw_err:
        hw_html = hw_err
    else:
        cpu_name = hw.get("cpu_model", "?")
        gpu_list = hw.get("gpus", [])
        gpu0 = gpu_list[0] if gpu_list else {}
        ram_gb = round(hw.get("ram_total_mb", 0) / 1024, 1)
        vram = round(gpu0.get("vram_total_mb", 0) / 1024, 1) if gpu0 else 0
        hw_html = f"""
<div class="grid">
  <div class="card kpi"><div class="value">{cpu_name}</div><div class="label">CPU</div></div>
  <div class="card kpi"><div class="value">{ram_gb} GB</div><div class="label">RAM</div></div>
  <div class="card kpi"><div class="value">{gpu0.get('name', 'N/A')}</div><div class="label">GPU</div></div>
  <div class="card kpi"><div class="value">{vram} GB</div><div class="label">VRAM</div></div>
</div>"""

    body = f"""\
<h1>&#9889; Model Optimization Workflow</h1>
<p style="color:#8b949e;margin-bottom:1rem;">Follow these steps to find, test, and optimize the best AI model for your hardware and use case.</p>

<div class="stepper">
  <span class="step-pill done" id="sp1">1. Hardware</span>
  <span class="step-pill" id="sp2">2. Mode</span>
  <span class="step-pill" id="sp3">3. Test</span>
  <span class="step-pill" id="sp4">4. Models</span>
  <span class="step-pill" id="sp5">5. Cache</span>
  <span class="step-pill" id="sp6">6. Overdrive</span>
  <span class="step-pill" id="sp7">7. Benchmark</span>
  <span class="step-pill" id="sp8">8. Results</span>
</div>

<!-- STEP 1: Hardware -->
<div class="card" id="step1">
  <h2>Step 1 — Hardware Detected</h2>
  {hw_html}
  <p style="color:var(--green);margin-top:.75rem;">&#10003; Hardware profiled automatically.</p>
</div>

<!-- STEP 2: Mode Selection -->
<div class="card" id="step2">
  <h2>Step 2 — Select Task Mode</h2>
  <p style="color:#8b949e;">What will you use the model for?</p>
  <div class="mode-grid">
    <div class="mode-card" onclick="selectMode(this,'general')"><div class="icon">&#129302;</div><div class="title">General</div><div class="desc">Chat, summarization, Q&amp;A</div></div>
    <div class="mode-card" onclick="selectMode(this,'coding')"><div class="icon">&#128187;</div><div class="title">Coding</div><div class="desc">Code generation &amp; review</div></div>
    <div class="mode-card" onclick="selectMode(this,'cyber')"><div class="icon">&#128737;</div><div class="title">Cyber Security</div><div class="desc">Sigma/YARA, threat analysis</div></div>
    <div class="mode-card" onclick="selectMode(this,'ids')"><div class="icon">&#128270;</div><div class="title">IDS / Detection</div><div class="desc">Intrusion detection systems</div></div>
  </div>
</div>

<!-- STEP 3: Run Tests -->
<div class="card" id="step3">
  <h2>Step 3 — Run Capability Tests</h2>
  <p style="color:#8b949e;">Run self-tests to verify your system can execute benchmarks.</p>
  <div style="display:flex;gap:.75rem;flex-wrap:wrap;margin:1rem 0;">
    <button class="btn btn-primary" onclick="runSelfTest('coding')">&#9654; Coding Self-Test</button>
    <button class="btn btn-primary" onclick="runSelfTest('cyber')">&#9654; Cyber Self-Test</button>
    <button class="btn btn-primary" onclick="runSelfTest('ids')">&#9654; IDS Quick Test</button>
  </div>
  <div class="result-box" id="test-result"></div>
</div>

<!-- STEP 4: Model Recommendations -->
<div class="card" id="step4">
  <h2>Step 4 — Model Recommendations</h2>
  <p style="color:#8b949e;">Get AI model suggestions ranked for your hardware and selected mode.</p>
  <button class="btn btn-green" onclick="getRecommendations()" id="rec-btn">&#127775; Get Recommendations</button>
  <div class="result-box" id="rec-result"></div>
</div>

<!-- STEP 5: Cache Management -->
<div class="card" id="step5">
  <h2>Step 5 — Cache Management</h2>
  <p style="color:#8b949e;">Free up space by removing unused cached models and artifacts.</p>
  <div style="display:flex;gap:.75rem;flex-wrap:wrap;margin:1rem 0;">
    <button class="btn btn-red" onclick="cleanupCache()">&#128465; Cleanup Volatile Cache</button>
    <button class="btn btn-outline" onclick="cleanupStale()">&#128336; Remove Stale (&gt;24h)</button>
  </div>
  <div class="result-box" id="cache-result"></div>
</div>

<!-- STEP 6: Overdrive — Quantize & Prune -->
<div class="card" id="step6" style="border-color:var(--purple);">
  <h2 style="color:var(--purple);">Step 6 — Overdrive: Quantize &amp; Optimize</h2>
  <p style="color:#8b949e;">Apply cutting-edge compression techniques to maximize performance on your hardware.</p>
  <div id="od-env-status" class="card" style="background:#1a1025;border-color:var(--border);"><span class="spinner"></span> Checking quantization backends...</div>

  <!-- Quick Mode Preset -->
  <div class="card" style="background:#1a1025;border-color:var(--purple);margin:1rem 0;">
    <label style="color:var(--purple);font-size:1rem;">&#9889; Quick Mode (pick one)</label>
    <div class="mode-grid" style="margin-top:.5rem;">
      <div class="mode-card" onclick="applyPreset(this,'light')" style="border-color:var(--green);">
        <div class="icon">&#127811;</div><div class="title">Light</div>
        <div class="desc">Ollama GGUF Q8 — minimal quality loss, larger file</div>
      </div>
      <div class="mode-card selected" onclick="applyPreset(this,'balanced')" style="border-color:var(--accent);">
        <div class="icon">&#9878;&#65039;</div><div class="title">Balanced</div>
        <div class="desc">Ollama GGUF Q4_K_M — best size/quality tradeoff</div>
      </div>
      <div class="mode-card" onclick="applyPreset(this,'maximum')" style="border-color:var(--red);">
        <div class="icon">&#128293;</div><div class="title">Maximum</div>
        <div class="desc">Ollama GGUF Q3_K_M — smallest file, some quality loss</div>
      </div>
    </div>
  </div>

  <!-- HF Token (for non-Ollama backends) -->
  <div class="card" id="hf-token-section" style="background:#1a1025;border-color:var(--border);display:none;">
    <label>&#128273; HuggingFace Token (optional &mdash; needed for gated/private models)</label>
    <input type="text" id="hf-token" placeholder="hf_xxxxx..." style="margin-top:.25rem;" />
    <p style="font-size:.8rem;color:#8b949e;margin-top:.25rem;">Public models work without a token. Gated models (e.g. Llama, Mistral) require one. Get yours at <a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</a>. Token is sent to your local API server only.</p>
  </div>

  <div class="grid" style="margin:1rem 0;">
    <div class="form-group">
      <label>Source Model</label>
      <input type="text" id="od-model" placeholder="Auto-filled from recommendations" />
      <p id="od-model-hint" style="font-size:.75rem;color:#8b949e;margin-top:.2rem;"></p>
    </div>
    <div class="form-group">
      <label>Quantization Backend</label>
      <select id="od-backend" onchange="onBackendChange()">
        <option value="ollama">Ollama GGUF (recommended — works locally)</option>
        <option value="bnb_4bit">bitsandbytes NF4 (needs PyTorch CUDA)</option>
        <option value="bnb_8bit">bitsandbytes INT8 (needs PyTorch CUDA)</option>
        <option value="awq">AWQ — Activation-aware (needs PyTorch CUDA)</option>
        <option value="gptq">GPTQ — Post-Training (needs PyTorch CUDA)</option>
      </select>
    </div>
    <div class="form-group" id="gguf-level-group">
      <label>GGUF Quant Level (Ollama only)</label>
      <select id="od-level">
        <option value="q4_k_m">Q4_K_M — 4-bit balanced (recommended)</option>
        <option value="q5_k_m">Q5_K_M — 5-bit higher quality</option>
        <option value="q8_0">Q8_0 — 8-bit near-lossless</option>
        <option value="q4_0">Q4_0 — 4-bit fastest</option>
        <option value="q3_k_m">Q3_K_M — 3-bit aggressive compression</option>
      </select>
    </div>
  </div>

  <div class="card" style="background:#1a1025;border-color:var(--purple);">
    <h3 style="color:var(--purple);">&#128218; How It Works</h3>
    <table>
      <tr><th>Method</th><th>Technique</th><th>Best For</th></tr>
      <tr><td><strong>GGUF</strong></td><td>K-quant mixed precision (llama.cpp)</td><td>CPU+GPU inference, Ollama</td></tr>
      <tr><td><strong>AWQ</strong></td><td>Activation-aware channel scaling</td><td>GPU inference, best generalization</td></tr>
      <tr><td><strong>GPTQ</strong></td><td>Hessian-based post-training quantization</td><td>GPU inference, fast one-shot</td></tr>
      <tr><td><strong>NF4</strong></td><td>4-bit NormalFloat + double quantization</td><td>QLoRA fine-tuning, memory-constrained</td></tr>
    </table>
    <p style="color:#8b949e;font-size:.8rem;margin-top:.5rem;"><strong>Pruning</strong> (SparseGPT/Wanda) and <strong>Knowledge Distillation</strong> are available on the <a href="/optimize">full Optimize page</a>.</p>
  </div>

  <div style="display:flex;gap:.75rem;flex-wrap:wrap;margin:1rem 0;">
    <button class="btn btn-purple" onclick="runQuantize()">&#128640; Quantize Model</button>
    <button class="btn btn-outline" onclick="runQuantCompare()" style="border-color:var(--purple);color:var(--purple);">&#128200; Compare All Methods</button>
    <a href="/optimize" class="btn btn-outline" style="text-decoration:none;">&#9881; Pruning &amp; Distillation &rarr;</a>
  </div>
  <div class="result-box" id="quant-result"></div>
</div>

<!-- STEP 7: Benchmark Comparison -->
<div class="card" id="step7">
  <h2>Step 7 — Benchmark: Original vs Optimized</h2>
  <p style="color:#8b949e;">Run task-specific benchmarks to measure the impact of optimization.</p>
  <div class="grid" style="margin:1rem 0;">
    <div class="form-group">
      <label>Original Model ID</label>
      <input type="text" id="bench-orig" placeholder="e.g. qwen2.5:7b" />
    </div>
    <div class="form-group">
      <label>Optimized Model ID</label>
      <input type="text" id="bench-opt" placeholder="e.g. qwen2.5:7b-q4_k_m" />
    </div>
    <div class="form-group">
      <label>Task Mode</label>
      <select id="bench-mode">
        <option value="general">General</option>
        <option value="coding">Coding</option>
        <option value="cyber">Cyber Security</option>
      </select>
    </div>
  </div>
  <div style="display:flex;gap:.75rem;flex-wrap:wrap;margin:1rem 0;">
    <button class="btn btn-primary" onclick="runBenchmark('original')">&#9654; Benchmark Original</button>
    <button class="btn btn-green" onclick="runBenchmark('optimized')">&#9654; Benchmark Optimized</button>
    <button class="btn btn-outline" onclick="compareBenchmarks()">&#128200; Compare Results</button>
  </div>
  <div class="result-box" id="bench-result"></div>
</div>

<!-- STEP 8: Final Results -->
<div class="card" id="step8" style="border-color:var(--green);">
  <h2 style="color:var(--green);">Step 8 — Final Recommendation</h2>
  <p style="color:#8b949e;">Based on your hardware, task mode, and benchmark results, here's the final analysis.</p>
  <div class="result-box" id="final-result"></div>
  <button class="btn btn-green" onclick="generateFinalReport()">&#128203; Generate Final Report</button>
  <a href="/chat" class="btn btn-outline" style="text-decoration:none;margin-left:.5rem;">Still unsure? Ask Chat &rarr;</a>
</div>
"""

    workflow_js = """
window._selectedMode = 'general';
window._benchCards = {};
window._selectedOllamaTag = '';
window._selectedHfRepo = '';

function setStep(n) {
    for (let i = 1; i <= 8; i++) {
        const sp = document.getElementById('sp' + i);
        if (i < n) sp.className = 'step-pill done';
        else if (i === n) sp.className = 'step-pill active';
        else sp.className = 'step-pill';
    }
}

function selectRecommendedModel(el) {
    // Uncheck all others, check this one
    document.querySelectorAll('#rec-result .model-select').forEach(c => {
        c.classList.remove('checked');
        const cb = c.querySelector('input[type=checkbox]');
        if (cb) cb.checked = false;
    });
    el.classList.add('checked');
    const cb = el.querySelector('input[type=checkbox]');
    if (cb) cb.checked = true;
    // Store both tags
    window._selectedOllamaTag = el.dataset.ollama || '';
    window._selectedHfRepo = el.dataset.hf || '';
    _syncModelField();
    // Fill benchmark fields
    document.getElementById('bench-orig').value = window._selectedOllamaTag || window._selectedHfRepo || '';
}

function _syncModelField() {
    // Fill od-model based on current backend selection
    const backend = document.getElementById('od-backend').value;
    const modelEl = document.getElementById('od-model');
    const hintEl = document.getElementById('od-model-hint');
    if (backend === 'ollama') {
        modelEl.value = window._selectedOllamaTag || window._selectedHfRepo || '';
        if (hintEl) hintEl.textContent = window._selectedOllamaTag ? 'Ollama tag: ' + window._selectedOllamaTag : '';
    } else {
        modelEl.value = window._selectedHfRepo || window._selectedOllamaTag || '';
        if (hintEl) hintEl.textContent = window._selectedHfRepo ? 'HF repo: ' + window._selectedHfRepo : (window._selectedOllamaTag ? 'No HF repo mapped — check registry.yaml' : '');
    }
}

function applyPreset(el, level) {
    // Highlight selected preset card
    document.querySelectorAll('#step6 .mode-card').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
    // Set backend to Ollama and apply the GGUF level
    document.getElementById('od-backend').value = 'ollama';
    const levelMap = {light: 'q8_0', balanced: 'q4_k_m', maximum: 'q3_k_m'};
    document.getElementById('od-level').value = levelMap[level] || 'q4_k_m';
    onBackendChange();
}

function onBackendChange() {
    const backend = document.getElementById('od-backend').value;
    const ggufGroup = document.getElementById('gguf-level-group');
    const hfSection = document.getElementById('hf-token-section');
    // Show GGUF level selector only for Ollama
    if (ggufGroup) ggufGroup.style.display = (backend === 'ollama') ? '' : 'none';
    // Show HF token input for non-Ollama backends
    if (hfSection) hfSection.style.display = (backend === 'ollama') ? 'none' : 'block';
    // Sync model field to correct ID format
    _syncModelField();
}
// Initialize backend visibility on page load
document.addEventListener('DOMContentLoaded', function() { onBackendChange(); });

// ── Load env status for Step 6 on page load ──
(async function() {
    try {
        const status = await cyberGet('/action/quantize-status');
        const el = document.getElementById('od-env-status');
        if (!el) return;
        if (status._error) { el.innerHTML = '<span style="color:#f85149">' + status._error + '</span>'; return; }
        const hw = status._hardware || {};
        const labels = {ollama_gguf:'Ollama GGUF', bnb_8bit:'INT8', bnb_4bit:'NF4', awq:'AWQ', gptq:'GPTQ'};
        let html = '<strong>Backend Availability</strong>';
        if (hw.gpu_name) html += '<span style="margin-left:.5rem;" class="badge badge-ok">' + hw.gpu_name + '</span>';
        if (hw.nvidia_gpu_detected && !hw.torch_cuda_available) html += '<span style="margin-left:.25rem;" class="badge badge-warn">PyTorch CUDA: No</span>';
        html += '<div style="display:flex;gap:.75rem;flex-wrap:wrap;margin-top:.5rem;">';
        for (const [m, info] of Object.entries(status)) {
            if (m.startsWith('_')) continue;
            const ok = info.available;
            const badge = ok ? 'badge-ok' : (info.status === 'dependency_missing' ? 'badge-warn' : 'badge-err');
            html += '<span class="badge ' + badge + '" title="' + (ok?'Ready':((info.reason||'')+'. '+(info.install||''))) + '">' + (labels[m]||m) + ': ' + (ok?'Ready':'Missing') + '</span>';
        }
        html += '</div>';
        el.innerHTML = html;
    } catch(e) { const el = document.getElementById('od-env-status'); if(el) el.innerHTML = '<span style="color:#8b949e">Could not check backends</span>'; }
})();

async function runSelfTest(type) {
    setStep(3);
    showLoading('test-result');
    try {
        const data = await cyberPost('/action/self-test/' + type);
        if (data._error) { showResult('test-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        let html = '<h3 style="color:var(--green)">&#10003; ' + type.toUpperCase() + ' Self-Test Complete</h3>';
        html += '<pre style="color:var(--fg);white-space:pre-wrap;font-size:.85rem;">' + JSON.stringify(data, null, 2) + '</pre>';
        showResult('test-result', html);
    } catch (e) { showResult('test-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function getRecommendations() {
    setStep(4);
    showLoading('rec-result');
    try {
        const data = await cyberPost('/action/recommend', {task_mode: window._selectedMode, top_k: 10});
        if (data._error) { showResult('rec-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        const picks = Array.isArray(data) ? data : (data.recommendations || data.picks || []);
        if (!picks.length) { showResult('rec-result', 'No recommendations returned.'); return; }
        let html = '<h3>Ranked Models for "' + window._selectedMode + '"</h3>';
        picks.forEach((p, i) => {
            const m = p.model || p;
            const name = m.display_name || m.model_id || m.id || '?';
            const ollamaTag = m.ollama_tag || '';
            const hfRepo = m.hf_repo || '';
            const modelKey = ollamaTag || hfRepo || m.id || name;
            const rawScore = p.composite_score || p.score || 0;
            const score = typeof rawScore === 'object' ? (rawScore.final || 0).toFixed(3) : (typeof rawScore === 'number' ? rawScore.toFixed(3) : rawScore);
            const reason = p.reason || '';
            html += '<div class="model-select" onclick="selectRecommendedModel(this)" data-ollama="' + ollamaTag + '" data-hf="' + hfRepo + '" data-name="' + name + '">';
            html += '<input type="checkbox" data-model="' + modelKey + '" data-display="' + name + '" ' + (i === 0 ? 'checked' : '') + ' />';
            html += '<div><strong>#' + (i+1) + ' ' + name + '</strong> — Score: ' + score;
            if (ollamaTag) html += '<br><span style="font-size:.8rem;color:var(--accent);">Ollama: ' + ollamaTag + '</span>';
            if (hfRepo) html += ' <span style="font-size:.8rem;color:var(--purple);">HF: ' + hfRepo + '</span>';
            if (reason) html += '<br><span style="color:#8b949e;font-size:.85rem;">' + reason + '</span>';
            html += '</div></div>';
        });
        // Auto-fill overdrive model — prefer a model with an ollama_tag (actually runnable)
        if (picks.length > 0) {
            const m0 = picks[0].model || picks[0];
            window._selectedOllamaTag = m0.ollama_tag || '';
            window._selectedHfRepo = m0.hf_repo || '';
            _syncModelField();
            document.getElementById('bench-orig').value = window._selectedOllamaTag || window._selectedHfRepo || '';
        }
        showResult('rec-result', html);
    } catch (e) { showResult('rec-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function cleanupCache() {
    setStep(5);
    showLoading('cache-result');
    try {
        const data = await cyberPost('/action/cache-cleanup');
        showResult('cache-result', '<span style="color:var(--green)">&#10003; Cleaned up ' + (data.deleted_count || 0) + ' volatile artifact(s).</span>');
    } catch (e) { showResult('cache-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function cleanupStale() {
    setStep(5);
    showLoading('cache-result');
    try {
        const data = await cyberPost('/action/cache-stale');
        showResult('cache-result', '<span style="color:var(--green)">&#10003; Removed ' + (data.deleted_count || 0) + ' stale cached artifact(s).</span>');
    } catch (e) { showResult('cache-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function runQuantize() {
    setStep(6);
    showLoading('quant-result');
    const model = document.getElementById('od-model').value;
    const backend = document.getElementById('od-backend').value;
    const level = document.getElementById('od-level').value;
    if (!model) { showResult('quant-result', '<span style="color:#f85149">Enter a source model name.</span>'); return; }

    // For Ollama backend, check if model exists and pull if needed
    if (backend === 'ollama') {
        showResult('quant-result', '<span class="spinner"></span> Checking if model is available locally...');
        try {
            const models = await cyberGet('/action/ollama-models');
            const list = Array.isArray(models) ? models : [];
            const found = list.some(m => m.name === model || m.name.startsWith(model + ':'));
            if (!found) {
                showResult('quant-result', '<span class="spinner"></span> Model not found locally — pulling <strong>' + model + '</strong> from Ollama (this may take a few minutes)...');
                const pullResult = await cyberPost('/action/ollama-pull', {model_name: model});
                if (pullResult._error) {
                    showResult('quant-result', '<span style="color:#f85149">Pull failed: ' + pullResult._error + '</span>');
                    return;
                }
            }
        } catch (e) { /* proceed anyway, server-side will also try to pull */ }
    }

    showResult('quant-result', '<span class="spinner"></span> Quantizing <strong>' + model + '</strong> with ' + backend.toUpperCase() + '...');
    try {
        const payload = {source_model: model, backend: backend, quant_method: level};
        const hfToken = (document.getElementById('hf-token') || {}).value;
        if (hfToken && backend !== 'ollama') payload.hf_token = hfToken;
        const data = await cyberPost('/action/quantize', payload);
        if (data._error || data.error) {
            showResult('quant-result', '<span style="color:#f85149">' + (data._error || data.error) + '</span>');
            return;
        }
        let html = '<h3 style="color:var(--green)">&#10003; Quantization Complete</h3>';
        html += buildTable(['Field', 'Value'], [
            ['Output Model', data.output_model || 'N/A'],
            ['Method', data.method || backend],
            ['Output Size', data.size_bytes ? (data.size_bytes / 1024 / 1024).toFixed(1) + ' MB' : 'N/A'],
            ['Duration', (data.duration_seconds || 0).toFixed(1) + 's'],
            ['Success', data.success ? '&#10003; Yes' : '&#10007; No'],
        ]);
        // Auto-fill benchmark optimized model
        if (data.output_model) document.getElementById('bench-opt').value = data.output_model;
        showResult('quant-result', html);
    } catch (e) { showResult('quant-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function runQuantCompare() {
    setStep(6);
    showLoading('quant-result');
    const model = document.getElementById('od-model').value;
    if (!model) { showResult('quant-result', '<span style="color:#f85149">Enter a source model name.</span>'); return; }
    try {
        const data = await cyberPost('/action/quantize-compare', {source_model: model});
        if (data._error) { showResult('quant-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        let html = '<h3 style="color:var(--purple)">Quantization Method Comparison</h3>';
        const entries = data.entries || [];
        if (entries.length) {
            const statusBadge = (e) => {
                if (e.success) return '<span class="badge badge-ok">&#10003; OK</span>';
                const cat = e.status || 'config_error';
                const catLabels = {unsupported_hardware:'Unsupported HW', service_unavailable:'Service Down', dependency_missing:'Missing Dep', config_error:'Config Error', model_error:'Model Error'};
                const catColors = {unsupported_hardware:'#f0883e', service_unavailable:'#da3633', dependency_missing:'#8b949e', config_error:'#f85149', model_error:'#f85149'};
                const label = catLabels[cat] || 'Failed';
                const color = catColors[cat] || '#f85149';
                return '<span class="badge" style="background:' + color + '22;color:' + color + ';border:1px solid ' + color + '55;" title="' + (e.error||'').replace(/"/g,'&quot;') + '">' + label + '</span><br><span style="font-size:.7rem;color:#8b949e;">' + (e.error||'') + '</span>';
            };
            const rows = entries.map(e => [e.method, e.quant_level || '', e.size_bytes ? (e.size_bytes/1024/1024).toFixed(1)+'MB' : '—', e.size_reduction_pct ? e.size_reduction_pct.toFixed(1)+'%' : '—', e.success ? (e.duration_seconds||0).toFixed(1)+'s' : '—', statusBadge(e)]);
            html += buildTable(['Method', 'Level', 'Size', 'Reduction', 'Time', 'Status'], rows);
        }
        if (data.best_size) html += '<p style="margin-top:.5rem;">Best size: <strong>' + data.best_size + '</strong></p>';
        if (data.best_speed) html += '<p>Fastest: <strong>' + data.best_speed + '</strong></p>';
        showResult('quant-result', html);
    } catch (e) { showResult('quant-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function runBenchmark(label) {
    setStep(7);
    showLoading('bench-result');
    const modelId = label === 'original' ? document.getElementById('bench-orig').value : document.getElementById('bench-opt').value;
    const mode = document.getElementById('bench-mode').value;
    if (!modelId) { showResult('bench-result', '<span style="color:#f85149">Enter a model ID.</span>'); return; }
    try {
        const data = await cyberPost('/action/benchmark', {model_id: modelId, task_mode: mode, label: label === 'original' ? 'baseline' : label, backend: 'ollama'});
        if (data._error || data.error) { showResult('bench-result', '<span style="color:#f85149">' + (data._error || data.error) + '</span>'); return; }
        window._benchCards[label] = data;
        let html = '<h3>' + label.charAt(0).toUpperCase() + label.slice(1) + ' Benchmark — ' + (data.model_id || modelId) + '</h3>';
        const sys = data.system || {};
        const task = data.task || {};
        html += buildTable(['Metric', 'Value'], [
            ['Latency', (sys.latency_ms||0).toFixed(1) + ' ms'],
            ['Throughput', (sys.throughput_tok_s||0).toFixed(1) + ' tok/s'],
            ['VRAM Peak', (sys.vram_peak_mb||0).toFixed(0) + ' MB'],
            ['Load Time', (sys.load_time_ms||0).toFixed(0) + ' ms'],
            ['Model Size', (sys.model_size_mb||0).toFixed(0) + ' MB'],
            ['Pass@k', task.pass_at_k != null ? task.pass_at_k.toFixed(3) : 'N/A'],
            ['Exact Match', task.exact_match != null ? task.exact_match.toFixed(3) : 'N/A'],
            ['F1', task.f1 != null ? task.f1.toFixed(3) : 'N/A'],
        ]);
        showResult('bench-result', html);
    } catch (e) { showResult('bench-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function compareBenchmarks() {
    setStep(7);
    const cards = await cyberGet('/action/bench-cards');
    if (!Array.isArray(cards) || cards.length < 2) {
        showResult('bench-result', '<span style="color:#f85149">Need at least 2 benchmark cards. Run both original and optimized benchmarks first.</span>');
        return;
    }
    showLoading('bench-result');
    try {
        const data = await cyberPost('/action/compare', {baseline_file: cards[cards.length-2], optimized_file: cards[cards.length-1], save_report: true});
        if (data._error) { showResult('bench-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        const cmp = data.comparison || data;
        const d = cmp.deltas || {};
        const bLabel = (cmp.baseline || {}).model_id || 'Baseline';
        const oLabel = (cmp.optimized || {}).model_id || 'Optimized';
        let html = '<h3 style="color:var(--green)">&#128200; Benchmark Comparison</h3>';
        html += '<p style="color:#8b949e;margin-bottom:1rem;"><strong>' + bLabel + '</strong> vs <strong style="color:var(--green)">' + oLabel + '</strong></p>';

        function barChart(metricName, baseVal, optVal, unit, lowerIsBetter) {
            if (baseVal == null || optVal == null) return '';
            const max = Math.max(baseVal, optVal, 0.001);
            const bPct = (baseVal / max * 100).toFixed(1);
            const oPct = (optVal / max * 100).toFixed(1);
            const diff = optVal - baseVal;
            const pct = baseVal !== 0 ? ((diff / baseVal) * 100).toFixed(1) : '0.0';
            const improved = lowerIsBetter ? diff < 0 : diff > 0;
            const arrow = improved ? '&#9650;' : (diff === 0 ? '&#9644;' : '&#9660;');
            const color = improved ? 'var(--green)' : (diff === 0 ? '#8b949e' : '#f85149');
            let h = '<div style="margin-bottom:1.2rem;">';
            h += '<div style="display:flex;justify-content:space-between;margin-bottom:.3rem;"><strong>' + metricName + '</strong><span style="color:' + color + ';font-weight:600;">' + arrow + ' ' + Math.abs(pct) + '%</span></div>';
            h += '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.2rem;"><span style="width:80px;font-size:.8rem;color:#8b949e;">' + bLabel.split(':')[0].slice(0,12) + '</span>';
            h += '<div style="flex:1;background:#21262d;border-radius:4px;height:22px;overflow:hidden;">';
            h += '<div style="width:' + bPct + '%;height:100%;background:#58a6ff;border-radius:4px;display:flex;align-items:center;padding-left:6px;font-size:.75rem;color:#0d1117;font-weight:600;">' + baseVal.toFixed(1) + unit + '</div></div></div>';
            h += '<div style="display:flex;align-items:center;gap:.5rem;"><span style="width:80px;font-size:.8rem;color:#8b949e;">' + oLabel.split(':')[0].slice(0,12) + '</span>';
            h += '<div style="flex:1;background:#21262d;border-radius:4px;height:22px;overflow:hidden;">';
            h += '<div style="width:' + oPct + '%;height:100%;background:var(--green);border-radius:4px;display:flex;align-items:center;padding-left:6px;font-size:.75rem;color:#0d1117;font-weight:600;">' + optVal.toFixed(1) + unit + '</div></div></div>';
            h += '</div>';
            return h;
        }

        html += '<div class="card" style="border-color:var(--green);">';
        html += '<h3>&#9889; Performance Metrics</h3>';
        if (d.latency_ms) html += barChart('Latency', d.latency_ms.baseline, d.latency_ms.optimized, ' ms', true);
        if (d.throughput_tok_s) html += barChart('Throughput', d.throughput_tok_s.baseline, d.throughput_tok_s.optimized, ' tok/s', false);
        if (d.load_time_ms) html += barChart('Load Time', d.load_time_ms.baseline, d.load_time_ms.optimized, ' ms', true);
        html += '</div>';

        html += '<div class="card" style="border-color:var(--purple);margin-top:1rem;">';
        html += '<h3>&#128190; Resource Usage</h3>';
        if (d.vram_peak_mb) html += barChart('VRAM Peak', d.vram_peak_mb.baseline, d.vram_peak_mb.optimized, ' MB', true);
        if (d.ram_peak_mb) html += barChart('RAM Peak', d.ram_peak_mb.baseline, d.ram_peak_mb.optimized, ' MB', true);
        html += '</div>';

        html += '<div class="card" style="border-color:#58a6ff;margin-top:1rem;">';
        html += '<h3>&#127919; Quality Metrics</h3>';
        if (d.exact_match) html += barChart('Exact Match', d.exact_match.baseline, d.exact_match.optimized, '', false);
        if (d.verifier_pass_rate && d.verifier_pass_rate.baseline != null) html += barChart('Verifier Pass', d.verifier_pass_rate.baseline, d.verifier_pass_rate.optimized, '', false);
        if (d.structured_output_validity) html += barChart('Structured Output', d.structured_output_validity.baseline, d.structured_output_validity.optimized, '', false);
        if (d.syntax_error_rate && d.syntax_error_rate.baseline != null) html += barChart('Syntax Error Rate', d.syntax_error_rate.baseline, d.syntax_error_rate.optimized, '', true);
        html += '</div>';

        if (data.report_path) html += '<p style="color:#8b949e;margin-top:.5rem;">Report saved: ' + data.report_path + '</p>';
        showResult('bench-result', html);
    } catch (e) { showResult('bench-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function generateFinalReport() {
    setStep(8);
    let html = '<h3 style="color:var(--green)">&#127942; Optimization Summary</h3>';
    html += '<p><strong>Task Mode:</strong> ' + window._selectedMode + '</p>';
    const orig = window._benchCards['original'];
    const opt = window._benchCards['optimized'];
    if (orig && opt) {
        const origSys = orig.system || {};
        const optSys = opt.system || {};

        function miniBar(label, baseVal, optVal, unit, lowerIsBetter) {
            if (!baseVal && !optVal) return '';
            const max = Math.max(baseVal, optVal, 0.001);
            const bPct = (baseVal / max * 100).toFixed(1);
            const oPct = (optVal / max * 100).toFixed(1);
            const diff = optVal - baseVal;
            const pct = baseVal !== 0 ? ((diff / baseVal) * 100).toFixed(1) : '0.0';
            const improved = lowerIsBetter ? diff < 0 : diff > 0;
            const color = improved ? 'var(--green)' : (diff === 0 ? '#8b949e' : '#f85149');
            const arrow = improved ? '&#9650;' : (diff === 0 ? '&#9644;' : '&#9660;');
            let h = '<div style="margin-bottom:1rem;">';
            h += '<div style="display:flex;justify-content:space-between;margin-bottom:.2rem;"><strong>' + label + '</strong><span style="color:' + color + ';font-weight:600;">' + arrow + ' ' + Math.abs(pct) + '%</span></div>';
            h += '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.15rem;"><span style="width:70px;font-size:.75rem;color:#8b949e;">Original</span>';
            h += '<div style="flex:1;background:#21262d;border-radius:4px;height:20px;overflow:hidden;">';
            h += '<div style="width:' + bPct + '%;height:100%;background:#58a6ff;border-radius:4px;display:flex;align-items:center;padding-left:5px;font-size:.7rem;color:#0d1117;font-weight:600;">' + baseVal.toFixed(1) + unit + '</div></div></div>';
            h += '<div style="display:flex;align-items:center;gap:.5rem;"><span style="width:70px;font-size:.75rem;color:#8b949e;">Optimized</span>';
            h += '<div style="flex:1;background:#21262d;border-radius:4px;height:20px;overflow:hidden;">';
            h += '<div style="width:' + oPct + '%;height:100%;background:var(--green);border-radius:4px;display:flex;align-items:center;padding-left:5px;font-size:.7rem;color:#0d1117;font-weight:600;">' + optVal.toFixed(1) + unit + '</div></div></div>';
            h += '</div>';
            return h;
        }

        html += '<div class="card" style="border-color:var(--green);">';
        html += '<h3>&#128200; Performance Comparison</h3>';
        html += '<p style="color:#8b949e;margin-bottom:.8rem;"><strong>' + (orig.model_id || '?') + '</strong> vs <strong style="color:var(--green)">' + (opt.model_id || '?') + '</strong></p>';
        html += miniBar('Latency', origSys.latency_ms||0, optSys.latency_ms||0, ' ms', true);
        html += miniBar('Throughput', origSys.throughput_tok_s||0, optSys.throughput_tok_s||0, ' tok/s', false);
        html += miniBar('VRAM Peak', origSys.vram_peak_mb||0, optSys.vram_peak_mb||0, ' MB', true);
        html += miniBar('Model Size', origSys.model_size_mb||0, optSys.model_size_mb||0, ' MB', true);
        html += miniBar('Load Time', origSys.load_time_ms||0, optSys.load_time_ms||0, ' ms', true);
        html += '</div>';
    } else {
        html += '<p style="color:#8b949e;">Run benchmarks on both original and optimized models (Step 7) to see a detailed comparison here.</p>';
    }
    html += '<p style="margin-top:1rem;color:#8b949e;">Tip: Not satisfied? Try a different quantization method in Step 6, or <a href="/chat">ask CyberForge Chat</a> for personalized advice.</p>';
    showResult('final-result', html);
}

setStep(2);
"""
    return _page("Workflow", body, "/workflow", extra_js=workflow_js)


# ── Hardware ─────────────────────────────────────────────────────

@app.get("/hardware", response_class=HTMLResponse)
async def hardware_page():
    hw = await _api("/api/hardware/profile")
    err = _err(hw)
    if err:
        return _page("Hardware", f"<h1>Hardware Profile</h1>{err}", "/hardware")

    cpu_name = hw.get("cpu_model", "?")
    gpu_list = hw.get("gpus", [])
    gpu0 = gpu_list[0] if gpu_list else {}
    ram_total = hw.get("ram_total_mb", 0)
    ram_avail = hw.get("ram_available_mb", 0)
    disk_free = hw.get("disk_free_mb", 0)

    rows = [
        ("CPU", cpu_name),
        ("CPU Cores", hw.get("cpu_cores", "?")),
        ("CPU Threads", hw.get("cpu_threads", "?")),
        ("RAM Total", f'{round(ram_total / 1024, 1)} GB'),
        ("RAM Available", f'{round(ram_avail / 1024, 1)} GB'),
        ("GPU", gpu0.get("name", "N/A")),
        ("VRAM", f'{round(gpu0.get("vram_total_mb", 0) / 1024, 1)} GB' if gpu0 else "N/A"),
        ("GPU Driver", gpu0.get("driver_version", "N/A")),
        ("CUDA Version", gpu0.get("cuda_version", "N/A")),
        ("Disk Free", f'{round(disk_free / 1024, 1)} GB'),
    ]
    table_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)

    body = f"""\
<h1>Hardware Profile</h1>
<div class="card">
<table>{table_rows}</table>
</div>
<p style="margin-top:1rem;color:#8b949e">POST <code>/api/hardware/profile/refresh</code> to re-scan hardware.</p>"""
    return _page("Hardware", body, "/hardware")


# ── Models ───────────────────────────────────────────────────────

@app.get("/models", response_class=HTMLResponse)
async def models_page():
    registry = await _api("/api/models/registry")
    err = _err(registry)
    if err:
        return _page("Models", f"<h1>Model Registry</h1>{err}", "/models")

    models_list = registry if isinstance(registry, list) else registry.get("models", [])
    rows = ""
    for m in models_list:
        name = m.get("display_name", m.get("model_id", m.get("id", "?")))
        params = m.get("params", m.get("parameters", m.get("param_count", "?")))
        quant = m.get("quantization", m.get("format", "—"))
        backend = m.get("backend", m.get("source", "—"))
        rows += f"<tr><td>{name}</td><td>{params}</td><td>{quant}</td><td>{backend}</td></tr>"

    body = f"""\
<h1>Model Registry</h1>
<div class="card">
<table>
<thead><tr><th>Model</th><th>Parameters</th><th>Quantization</th><th>Backend</th></tr></thead>
<tbody>{rows if rows else '<tr><td colspan="4">No models registered yet.</td></tr>'}</tbody>
</table>
</div>

<h2>&#127760; Web Discovery</h2>
<p style="color:#8b949e;margin-bottom:.75rem;">Search HuggingFace Hub for models that match your hardware. Shows what you can run <em>natively</em> and what you can run <strong>after quantization/pruning</strong>.</p>
<div class="grid" style="margin:1rem 0;">
  <div class="form-group">
    <label>Task Mode</label>
    <select id="wd-task">
      <option value="general">General</option>
      <option value="coding">Coding</option>
      <option value="cyber">Cyber Security</option>
    </select>
  </div>
  <div class="form-group">
    <label>Max Results</label>
    <select id="wd-limit">
      <option value="20">20</option>
      <option value="30" selected>30</option>
      <option value="50">50</option>
    </select>
  </div>
</div>
<button class="btn btn-primary" onclick="doWebDiscover()">&#128269; Discover Models</button>
<div class="result-box" id="wd-result"></div>

<h2>&#128203; Select &amp; Cleanup</h2>
<p style="color:#8b949e;margin-bottom:.75rem;">Select a model to keep and delete all other cached models.</p>
<div class="form-group" style="max-width:400px;">
  <label>Model to Keep</label>
  <input type="text" id="sel-model" placeholder="e.g. qwen2.5-7b-instruct" />
</div>
<label style="display:flex;align-items:center;gap:.5rem;margin:.5rem 0;"><input type="checkbox" id="sel-ollama" /> Also delete other Ollama models</label>
<button class="btn btn-red" onclick="doSelectModel()">&#128465; Select &amp; Cleanup</button>
<div class="result-box" id="sel-result"></div>"""

    models_js = """
async function doWebDiscover() {
    showLoading('wd-result');
    try {
        const data = await cyberPost('/action/web-discover', {
            task_mode: document.getElementById('wd-task').value,
            limit: parseInt(document.getElementById('wd-limit').value)
        });
        if (data._error) { showResult('wd-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }

        let html = '<p style="color:#8b949e;">' + (data.hardware_summary||'') + ' &bull; Found: ' + (data.total_found||0) + '</p>';

        // Local installed
        if (data.local_installed && data.local_installed.length) {
            html += '<h3 style="color:var(--green)">&#9989; Already Installed</h3><div class="grid">';
            data.local_installed.forEach(m => {
                html += '<div class="card"><strong>' + (m.name||m.model_id||'?') + '</strong><br>'
                    + '<span class="badge badge-ok">Installed</span></div>';
            });
            html += '</div>';
        }

        // Native tier
        const native = data.native || [];
        if (native.length) {
            html += '<h3 style="color:var(--accent)">&#9889; Native (runs as-is)</h3>';
            html += buildTable(['Model','Params','FP16 VRAM','Downloads',''],
                native.map(m => [
                    '<a href="' + m.url + '" target="_blank">' + m.display_name + '</a>',
                    m.params_b + 'B',
                    (m.estimated_vram_mb/1024).toFixed(1) + ' GB',
                    (m.downloads||0).toLocaleString(),
                    '<button class="btn btn-green" style="padding:.25rem .75rem;font-size:.8rem;" onclick="pickModel(\\'' + m.model_id.replace(/'/g,"\\\\'") + '\\')">Select</button>'
                ]));
        }

        // Post-quant tier
        const pq = data.post_quant || [];
        if (pq.length) {
            html += '<h3 style="color:var(--purple)">&#128640; After Quantization</h3>';
            html += buildTable(['Model','Params','Q4 VRAM','Suggested','Downloads',''],
                pq.map(m => [
                    '<a href="' + m.url + '" target="_blank">' + m.display_name + '</a>',
                    m.params_b + 'B',
                    (m.estimated_quant_vram_mb/1024).toFixed(1) + ' GB',
                    '<span class="badge badge-purple">' + (m.suggested_quant||'q4_K_M') + '</span>',
                    (m.downloads||0).toLocaleString(),
                    '<button class="btn btn-purple" style="padding:.25rem .75rem;font-size:.8rem;" onclick="pickModel(\\'' + m.model_id.replace(/'/g,"\\\\'") + '\\')">Select</button>'
                ]));
        }

        if (!native.length && !pq.length) html += '<p>No models found matching your hardware.</p>';
        showResult('wd-result', html);
    } catch (e) { showResult('wd-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}

function pickModel(modelId) {
    document.getElementById('sel-model').value = modelId;
    document.getElementById('sel-model').scrollIntoView({behavior:'smooth'});
}

async function doSelectModel() {
    const model = document.getElementById('sel-model').value;
    if (!model) { showResult('sel-result', '<span style="color:#f85149">Enter a model name</span>'); return; }
    if (!confirm('This will DELETE all other cached models. Continue?')) return;
    showLoading('sel-result');
    try {
        const data = await cyberPost('/action/model-select', {
            selected_model: model,
            delete_ollama_others: document.getElementById('sel-ollama').checked
        });
        if (data._error || data.error) { showResult('sel-result', '<span style="color:#f85149">' + (data._error||data.error) + '</span>'); return; }
        showResult('sel-result',
            '<h3 style="color:var(--green)">&#10003; Selected: ' + data.selected_model + '</h3>' +
            '<p>Deleted ' + data.deleted_count + ' models, freed ' + data.freed_mb + ' MB</p>' +
            (data.deleted_names.length ? '<ul>' + data.deleted_names.map(n => '<li>' + n + '</li>').join('') + '</ul>' : ''));
    } catch (e) { showResult('sel-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}
"""
    return _page("Models", body, "/models", extra_js=models_js)


# ── Recommend ────────────────────────────────────────────────────

@app.get("/recommend", response_class=HTMLResponse)
async def recommend_page():
    # Get hardware profile for context
    hw = await _api("/api/hardware/profile")
    hw_err = _err(hw)
    gpu_name = ""
    vram_total = 0
    vram_free = 0
    ram_total = 0
    if not hw_err and isinstance(hw, dict):
        gpus = hw.get("gpus", [])
        if gpus:
            gpu_name = gpus[0].get("name", "")
            vram_total = gpus[0].get("vram_total_mb", 0)
            vram_free = gpus[0].get("vram_free_mb", 0)
        ram_total = hw.get("ram_total_mb", 0)

    data = await _api("/api/recommend/", method="POST", json={"task_mode": "general", "include_post_quant": True, "include_not_recommended": True, "top_k": 15})
    err = _err(data)

    body = '<h1>Recommendations</h1>'

    # Hardware summary for novice users
    if not hw_err and isinstance(hw, dict):
        body += '<div class="card" style="border-color:var(--accent);margin-bottom:1rem;">'
        body += '<strong style="color:var(--accent);">Your Hardware</strong><div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.5rem;margin-top:.5rem;">'
        if gpu_name:
            body += f'<div style="text-align:center;"><span class="badge badge-ok">{_html.escape(gpu_name)}</span></div>'
        body += f'<div style="text-align:center;"><small>VRAM</small><br><strong>{round(vram_total/1024,1)} GB</strong> ({round(vram_free/1024,1)} GB free)</div>'
        body += f'<div style="text-align:center;"><small>RAM</small><br><strong>{round(ram_total/1024,1)} GB</strong></div>'
        # Capacity guidance for novice users
        if vram_total > 0:
            max_q4_params = round(vram_total / 600, 1)  # rough: Q4 needs ~0.6 GB per B params
            max_fp16_params = round(vram_total / 2000, 1)  # rough: FP16 needs ~2 GB per B params
            body += f'<div style="text-align:center;"><small>Estimated Capacity</small><br>'
            body += f'<strong>~{max_fp16_params}B</strong> FP16<br><strong>~{max_q4_params}B</strong> Q4</div>'
        body += '</div></div>'

    if err:
        body += err
        return _page("Recommend", body, "/recommend")

    body += '<p style="color:#8b949e;margin-bottom:1rem;">Models ranked by composite score for your hardware. <strong>Runs Now</strong> models work immediately. <strong>After Optimization</strong> need compression first. <strong>Not Recommended</strong> models are too large for this machine class.</p>'
    body += '<div id="selected-model-banner" class="card" style="border-color:var(--green);display:none;margin-bottom:1rem;"><strong style="color:var(--green);">&#10003; Selected Model:</strong> <span id="selected-model-name"></span></div>'

    picks = data if isinstance(data, list) else data.get("recommendations", data.get("picks", []))
    if not picks:
        body = '<h1>Recommendations</h1><div class="card">No recommendations returned. Check hardware profile and model registry.</div>'
        return _page("Recommend", body, "/recommend")

    native_rows = ""
    pq_rows = ""
    nr_rows = ""
    for i, p in enumerate(picks, 1):
        model = p.get("model", p) if isinstance(p, dict) else p
        name = model.get("display_name", model.get("model_id", model.get("id", "?"))) if isinstance(model, dict) else str(model)
        # Extract the real runnable identifier: ollama_tag or hf_repo or id
        model_id = ""
        ollama_tag = ""
        hf_repo = ""
        if isinstance(model, dict):
            model_id = model.get("ollama_tag") or model.get("hf_repo") or model.get("id", "")
            ollama_tag = model.get("ollama_tag", "")
            hf_repo = model.get("hf_repo", "")
        raw_score = p.get("composite_score", p.get("score", "?")) if isinstance(p, dict) else "?"
        if isinstance(raw_score, dict):
            score = f"{raw_score.get('final', 0):.3f}"
        elif isinstance(raw_score, float):
            score = f"{raw_score:.3f}"
        else:
            score = str(raw_score)
        reason = p.get("reason", "") if isinstance(p, dict) else ""
        req_quant = p.get("requires_quantization", False) if isinstance(p, dict) else False
        sugg_q = p.get("suggested_quant", "") if isinstance(p, dict) else ""
        bench_safe = p.get("benchmark_safe", True) if isinstance(p, dict) else True
        fit_note = p.get("fit_note", "") if isinstance(p, dict) else ""
        tier = p.get("tier", "runs_now") if isinstance(p, dict) else "runs_now"
        mid_escaped = _html.escape(model_id).replace("'", "&#39;")
        id_badge = f'<br><code style="font-size:.75rem;color:#8b949e;">{_html.escape(model_id)}</code>' if model_id else ""
        safe_badge = ""
        if not bench_safe:
            safe_badge = ' <span class="badge badge-warn" style="font-size:.65rem;">NOT BENCHMARK-SAFE</span>'
        fit_line = ""
        if fit_note:
            fit_line = f'<br><span style="font-size:.75rem;color:var(--yellow);">{_html.escape(fit_note)}</span>'
        # Build the JSON for CyberForge.setSelected()
        import json as _json_mod
        sel_obj = _json_mod.dumps({
            "id": model.get("id", "") if isinstance(model, dict) else "",
            "display_name": name,
            "ollama_tag": ollama_tag,
            "hf_repo": hf_repo,
            "benchmark_safe": bench_safe,
            "requires_quantization": req_quant,
            "suggested_quant": sugg_q,
            "fit_note": fit_note,
            "tier": tier,
        })
        sel_escaped = _html.escape(sel_obj).replace("'", "&#39;")
        select_btn = f'''<button class="btn btn-sm btn-green" onclick="selectRec('{sel_escaped}', this)">Select</button>''' if tier != "not_recommended" else ""

        if tier == "not_recommended":
            row = f"<tr style='opacity:.6;'><td>{i}</td><td>{name}{safe_badge}{id_badge}{fit_line}</td><td>{score}</td><td style='font-size:.85rem;color:#8b949e;'>{reason}</td><td></td></tr>"
            nr_rows += row
        elif req_quant or tier == "runs_after_optimization":
            badge = f'<span class="badge badge-purple">{sugg_q}</span> ' if sugg_q else ""
            row = f"<tr><td>{i}</td><td>{badge}{name}{safe_badge}{id_badge}{fit_line}</td><td>{score}</td><td style='font-size:.85rem;color:#8b949e;'>{reason}</td><td>{select_btn}</td></tr>"
            pq_rows += row
        else:
            row = f"<tr><td>{i}</td><td>{name}{safe_badge}{id_badge}{fit_line}</td><td>{score}</td><td style='font-size:.85rem;color:#8b949e;'>{reason}</td><td>{select_btn}</td></tr>"
            native_rows += row

    table_head = '<thead><tr><th>#</th><th>Model</th><th>Score</th><th>Reason</th><th></th></tr></thead>'

    if native_rows:
        body += '<h2>&#9889; Runs Now</h2>'
        body += '<p style="color:#8b949e;margin-bottom:.5rem;">These models fit your hardware now &mdash; pull with Ollama and go.</p>'
        body += f'<div class="card"><table>{table_head}<tbody>{native_rows}</tbody></table></div>'

    if pq_rows:
        body += '<h2>&#128640; After Optimization</h2>'
        body += '<p style="color:#8b949e;margin-bottom:.5rem;">These models are too large at full precision but will fit after quantization. Use the Optimize page to quantize first.</p>'
        body += f'<div class="card"><table>{table_head}<tbody>{pq_rows}</tbody></table></div>'

    if nr_rows:
        body += '<h2 style="color:#f85149;">&#128683; Not Recommended</h2>'
        body += '<p style="color:#8b949e;margin-bottom:.5rem;">These models are too large for this machine class, even after quantization.</p>'
        body += f'<div class="card" style="border-color:#f85149;opacity:.75;"><table>{table_head}<tbody>{nr_rows}</tbody></table></div>'

    if not native_rows and not pq_rows and not nr_rows:
        body += '<div class="card">No recommendations returned.</div>'

    rec_js = """
function selectRec(jsonStr, btn) {
    const model = JSON.parse(jsonStr.replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'));
    CyberForge.setSelected(model);
    document.getElementById('selected-model-name').textContent = model.display_name + ' (' + (model.ollama_tag || model.hf_repo || model.id) + ')';
    document.getElementById('selected-model-banner').style.display = '';
    // Highlight the selected row
    document.querySelectorAll('table tr.selected-row').forEach(r => r.classList.remove('selected-row'));
    btn.closest('tr').classList.add('selected-row');
}
// On load, show previously selected model if any
(function() {
    const m = CyberForge.getSelected();
    if (m) {
        document.getElementById('selected-model-name').textContent = m.display_name + ' (' + (m.ollama_tag || m.hf_repo || m.id) + ')';
        document.getElementById('selected-model-banner').style.display = '';
    }
})();
"""

    return _page("Recommend", body, "/recommend", extra_js=rec_js)


# ── Bench ────────────────────────────────────────────────────────

@app.get("/bench", response_class=HTMLResponse)
async def bench_page():
    suites = await _api("/api/bench/suites")
    err = _err(suites)
    if err:
        return _page("Bench", f"<h1>Benchmarks</h1>{err}", "/bench")

    suite_list = suites.get("suites", [])
    rows = ""
    for s in suite_list:
        name = s.get("name", s.get("suite_id", "?"))
        desc = s.get("description", "—")
        rows += f"<tr><td>{name}</td><td>{desc}</td></tr>"

    body = f"""\
<h1>Benchmarks</h1>

<h2>Available Suites</h2>
<div class="card">
<table>
<thead><tr><th>Suite</th><th>Description</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>

<h2>Run Self-Tests</h2>
<p style="color:#8b949e;margin-bottom:1rem;">Self-tests use known-good inputs — no LLM required.</p>
<div style="display:flex;gap:.75rem;flex-wrap:wrap;">
  <button class="btn btn-primary" onclick="runTest('coding')">&#9654; Coding Self-Test</button>
  <button class="btn btn-primary" onclick="runTest('cyber')">&#9654; Cyber Self-Test</button>
  <button class="btn btn-green" onclick="runTest('ids')">&#9654; IDS Quick Benchmark</button>
</div>
<div class="result-box" id="self-test-result"></div>

<h2>Run Model Benchmark</h2>
<div class="grid" style="margin:1rem 0;">
  <div class="form-group">
    <label>Model</label>
    <div style="display:flex;gap:.25rem;align-items:center;">
      <select id="bm-model" style="flex:1;"><option value="">Loading models...</option></select>
      <button class="btn btn-sm btn-outline" onclick="refreshBenchModels()">&#8635;</button>
    </div>
  </div>
  <div class="form-group">
    <label>Task Mode</label>
    <select id="bm-mode"><option value="general">General</option><option value="coding">Coding</option><option value="cyber">Cyber</option></select>
  </div>
  <div class="form-group">
    <label>Label</label>
    <input type="text" id="bm-label" placeholder="baseline or optimized tag" value="baseline" />
  </div>
</div>
<button class="btn btn-primary" onclick="runModelBench()">&#9654; Run Benchmark</button>
<div id="model-fit-banner" class="card" style="display:none;margin-top:.5rem;"></div>
<div class="result-box" id="model-bench-result"></div>

<h2>Compare Benchmark Cards</h2>
<div class="grid" style="margin:1rem 0;">
  <div class="form-group">
    <label>Baseline Card</label>
    <select id="cmp-base"><option value="">-- loading cards --</option></select>
  </div>
  <div class="form-group">
    <label>Optimized Card</label>
    <select id="cmp-opt"><option value="">-- loading cards --</option></select>
  </div>
</div>
<div style="display:flex;gap:.75rem;flex-wrap:wrap;">
  <button class="btn btn-outline" onclick="refreshCardDropdowns()">&#128451; Refresh Cards</button>
  <button class="btn btn-green" onclick="runCompare()">&#128200; Compare</button>
</div>
<div class="result-box" id="compare-result"></div>
"""

    bench_js = """
async function runTest(type) {
    showLoading('self-test-result');
    try {
        const data = await cyberPost('/action/self-test/' + type);
        if (data._error) { showResult('self-test-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        showResult('self-test-result', '<h3 style="color:var(--green)">&#10003; ' + type.toUpperCase() + ' Complete</h3><pre style="color:var(--fg);white-space:pre-wrap;font-size:.85rem;">' + JSON.stringify(data, null, 2) + '</pre>');
    } catch (e) { showResult('self-test-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}
async function runModelBench() {
    showLoading('model-bench-result');
    const model = document.getElementById('bm-model').value;
    const mode = document.getElementById('bm-mode').value;
    const label = document.getElementById('bm-label').value || 'baseline';
    if (!model) { showResult('model-bench-result', '<span style="color:#f85149">Enter a model ID</span>'); return; }
    // ── Universal preflight: check ollama backend availability ──
    { const pf = await CyberForge.preflight('benchmark', 'ollama', 0, 0);
      if (pf && !pf.allowed) { showResult('model-bench-result', '<span style="color:#f85149">&#9888; ' + (pf.reason||'Blocked') + '</span>' + (pf.suggestion ? '<br><span style="color:#8b949e">' + pf.suggestion + '</span>' : '')); return; } }
    // ── Model-fit preflight ──
    const fitBanner = document.getElementById('model-fit-banner');
    try {
        const fit = await cyberPost('/action/model-fit', {model_name: model});
        if (fit && !fit._error) {
            const catColors = {fits:'var(--green)', tight:'var(--yellow)', partial_offload:'var(--yellow)', cpu_only:'var(--yellow)', too_large:'var(--red)', service_down:'var(--red)'};
            const color = catColors[fit.category] || 'var(--fg)';
            fitBanner.innerHTML = '<span style="color:' + color + ';">' + (fit.reason||'') + '</span>' +
                (fit.model_size_mb ? '<span style="color:#8b949e;font-size:.85rem;margin-left:.5rem;">Model: ' + fit.model_size_mb + ' MB | VRAM: ' + fit.vram_free_mb + '/' + fit.vram_total_mb + ' MB</span>' : '');
            fitBanner.style.borderColor = color;
            fitBanner.style.display = '';
            if (fit.category === 'too_large') {
                showResult('model-bench-result', '<span style="color:#f85149">&#9888; ' + fit.reason + '</span>');
                return;
            }
        } else { fitBanner.style.display = 'none'; }
    } catch(e) { fitBanner.style.display = 'none'; }
    try {
        const data = await cyberPost('/action/benchmark', {model_id: model, task_mode: mode, label: label, backend: 'ollama'});
        if (data._error) { showResult('model-bench-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        const s = data.system || {};
        showResult('model-bench-result', '<h3 style="color:var(--green)">&#10003; Benchmark Complete</h3>' + buildTable(['Metric','Value'],[['Model', data.model_id],['Label', data.label],['Latency', (s.latency_ms||0).toFixed(1)+' ms'],['Throughput', (s.throughput_tok_s||0).toFixed(1)+' tok/s'],['VRAM Peak', (s.vram_peak_mb||0).toFixed(0)+' MB'],['Model Size', (s.model_size_mb||0).toFixed(0)+' MB']]));
        // Refresh card dropdowns after a new benchmark
        await refreshCardDropdowns();
    } catch (e) { showResult('model-bench-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}
async function refreshCardDropdowns() {
    try {
        const cards = await cyberGet('/action/bench-cards');
        const baseEl = document.getElementById('cmp-base');
        const optEl = document.getElementById('cmp-opt');
        const prevBase = baseEl.value;
        const prevOpt = optEl.value;
        baseEl.innerHTML = '';
        optEl.innerHTML = '';
        if (Array.isArray(cards) && cards.length) {
            baseEl.innerHTML += '<option value="">-- select baseline --</option>';
            optEl.innerHTML += '<option value="">-- select optimized --</option>';
            cards.forEach(c => {
                baseEl.innerHTML += '<option value="' + c + '">' + c + '</option>';
                optEl.innerHTML += '<option value="' + c + '">' + c + '</option>';
            });
            if (prevBase) baseEl.value = prevBase;
            if (prevOpt) optEl.value = prevOpt;
            // Auto-select: oldest → baseline, newest → optimized
            if (!baseEl.value && cards.length >= 2) baseEl.value = cards[0];
            if (!optEl.value && cards.length >= 2) optEl.value = cards[cards.length - 1];
        } else {
            baseEl.innerHTML = '<option value="">-- no cards yet --</option>';
            optEl.innerHTML = '<option value="">-- no cards yet --</option>';
        }
    } catch (e) { /* silent */ }
}
// Auto-populate card dropdowns on page load
refreshCardDropdowns();
// ── Bench model dropdown with safe pre-selection ──
async function refreshBenchModels() {
    const sel = document.getElementById('bm-model');
    const mc = CyberForge.getMachineClass();
    const safeVramBytes = mc && mc.vram_total_mb ? mc.vram_total_mb * 1024 * 1024 * 0.85 : 5 * 1024 * 1024 * 1024;
    try {
        const models = await cyberGet('/action/ollama-models');
        const list = Array.isArray(models) ? models : (models.models || []);
        if (list.length > 0) {
            sel.innerHTML = '';
            // Sort by size ascending so smaller (safer) models appear first
            list.sort((a, b) => (a.size || 0) - (b.size || 0));
            const selectedId = CyberForge.getModelId();
            let safeIdx = -1;
            let selectedIdx = -1;
            list.forEach((m, idx) => {
                const name = m.name || m;
                const sizeGB = m.size ? (m.size / 1024 / 1024 / 1024).toFixed(1) : '?';
                const safe = m.size && m.size <= safeVramBytes;
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name + ' (' + sizeGB + ' GB)' + (safe ? '' : ' \u26a0\ufe0f VRAM risk');
                sel.appendChild(opt);
                if (safe && safeIdx < 0) safeIdx = idx;
                if (selectedId && name === selectedId) selectedIdx = idx;
            });
            // Prefer the globally selected model, else first benchmark-safe model
            if (selectedIdx >= 0) sel.selectedIndex = selectedIdx;
            else if (safeIdx >= 0) sel.selectedIndex = safeIdx;
        } else {
            sel.innerHTML = '<option value="">No models found</option>';
        }
    } catch (e) {
        sel.innerHTML = '<option value="">Failed to load</option>';
    }
}
refreshBenchModels();
async function runCompare() {
    showLoading('compare-result');
    const base = document.getElementById('cmp-base').value;
    const opt = document.getElementById('cmp-opt').value;
    if (!base || !opt) { showResult('compare-result', '<span style="color:#f85149">Enter both card filenames</span>'); return; }
    if (base === opt) { showResult('compare-result', '<span style="color:#f85149">Select two different cards</span>'); return; }
    try {
        const data = await cyberPost('/action/compare', {baseline_file: base, optimized_file: opt, save_report: true});
        if (data._error) { showResult('compare-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        const cmp = data.comparison || data;
        const d = cmp.deltas || {};
        const bLabel = (cmp.baseline || {}).model_id || 'Baseline';
        const oLabel = (cmp.optimized || {}).model_id || 'Optimized';
        let html = '<h3 style="color:var(--green)">&#128200; Benchmark Comparison</h3>';
        html += '<p style="color:#8b949e;margin-bottom:1rem;"><strong>' + bLabel + '</strong> vs <strong style="color:var(--green)">' + oLabel + '</strong></p>';

        function barChart(metricName, baseVal, optVal, unit, lowerIsBetter) {
            if (baseVal == null || optVal == null) return '';
            const max = Math.max(baseVal, optVal, 0.001);
            const bPct = (baseVal / max * 100).toFixed(1);
            const oPct = (optVal / max * 100).toFixed(1);
            const diff = optVal - baseVal;
            const pct = baseVal !== 0 ? ((diff / baseVal) * 100).toFixed(1) : '0.0';
            const improved = lowerIsBetter ? diff < 0 : diff > 0;
            const arrow = improved ? '&#9650;' : (diff === 0 ? '&#9644;' : '&#9660;');
            const color = improved ? 'var(--green)' : (diff === 0 ? '#8b949e' : '#f85149');
            let h = '<div style="margin-bottom:1.2rem;">';
            h += '<div style="display:flex;justify-content:space-between;margin-bottom:.3rem;"><strong>' + metricName + '</strong><span style="color:' + color + ';font-weight:600;">' + arrow + ' ' + Math.abs(pct) + '%</span></div>';
            h += '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.2rem;"><span style="width:80px;font-size:.8rem;color:#8b949e;">' + bLabel.split(':')[0].slice(0,12) + '</span>';
            h += '<div style="flex:1;background:#21262d;border-radius:4px;height:22px;overflow:hidden;">';
            h += '<div style="width:' + bPct + '%;height:100%;background:#58a6ff;border-radius:4px;display:flex;align-items:center;padding-left:6px;font-size:.75rem;color:#0d1117;font-weight:600;">' + baseVal.toFixed(1) + unit + '</div></div></div>';
            h += '<div style="display:flex;align-items:center;gap:.5rem;"><span style="width:80px;font-size:.8rem;color:#8b949e;">' + oLabel.split(':')[0].slice(0,12) + '</span>';
            h += '<div style="flex:1;background:#21262d;border-radius:4px;height:22px;overflow:hidden;">';
            h += '<div style="width:' + oPct + '%;height:100%;background:var(--green);border-radius:4px;display:flex;align-items:center;padding-left:6px;font-size:.75rem;color:#0d1117;font-weight:600;">' + optVal.toFixed(1) + unit + '</div></div></div>';
            h += '</div>';
            return h;
        }

        html += '<div class="card" style="border-color:var(--green);">';
        html += '<h3>&#9889; Performance</h3>';
        if (d.latency_ms) html += barChart('Latency', d.latency_ms.baseline, d.latency_ms.optimized, ' ms', true);
        if (d.throughput_tok_s) html += barChart('Throughput', d.throughput_tok_s.baseline, d.throughput_tok_s.optimized, ' tok/s', false);
        if (d.load_time_ms) html += barChart('Load Time', d.load_time_ms.baseline, d.load_time_ms.optimized, ' ms', true);
        html += '</div>';

        html += '<div class="card" style="border-color:var(--purple);margin-top:1rem;">';
        html += '<h3>&#128190; Resources</h3>';
        if (d.vram_peak_mb) html += barChart('VRAM Peak', d.vram_peak_mb.baseline, d.vram_peak_mb.optimized, ' MB', true);
        if (d.ram_peak_mb) html += barChart('RAM Peak', d.ram_peak_mb.baseline, d.ram_peak_mb.optimized, ' MB', true);
        html += '</div>';

        html += '<div class="card" style="border-color:#58a6ff;margin-top:1rem;">';
        html += '<h3>&#127919; Quality</h3>';
        if (d.exact_match) html += barChart('Exact Match', d.exact_match.baseline, d.exact_match.optimized, '', false);
        if (d.verifier_pass_rate && d.verifier_pass_rate.baseline != null) html += barChart('Verifier Pass', d.verifier_pass_rate.baseline, d.verifier_pass_rate.optimized, '', false);
        if (d.structured_output_validity) html += barChart('Structured Output', d.structured_output_validity.baseline, d.structured_output_validity.optimized, '', false);
        if (d.syntax_error_rate && d.syntax_error_rate.baseline != null) html += barChart('Syntax Error Rate', d.syntax_error_rate.baseline, d.syntax_error_rate.optimized, '', true);
        html += '</div>';

        if (data.report_path) html += '<p style="color:#8b949e;margin-top:.5rem;">Report: ' + data.report_path + '</p>';
        showResult('compare-result', html);
    } catch (e) { showResult('compare-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}
"""
    return _page("Bench", body, "/bench", extra_js=bench_js)


# ── Cyber ────────────────────────────────────────────────────────

@app.get("/cyber", response_class=HTMLResponse)
async def cyber_page():
    datasets = await _api("/api/cyber/datasets")
    err = _err(datasets)
    if err:
        return _page("Cyber", f"<h1>Cyber Lab</h1>{err}", "/cyber")

    ds_list = datasets.get("datasets", [])
    rows = ""
    for d in ds_list:
        name = d.get("name", "?")
        loader = d.get("loader", "?")
        status = d.get("status", "?")
        badge = "badge-ok" if status == "available" else "badge-warn"
        rows += f'<tr><td>{name}</td><td><code>{loader}</code></td><td><span class="badge {badge}">{status}</span></td></tr>'

    body = f"""\
<h1>Cyber Lab</h1>
<h2>Datasets</h2>
<div class="card">
<table>
<thead><tr><th>Dataset</th><th>Loader</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
<h2>Validation Tools</h2>
<div class="grid">
  <div class="card"><strong>Sigma Validator</strong><br><code>POST /api/cyber/validate/sigma</code></div>
  <div class="card"><strong>YARA Validator</strong><br><code>POST /api/cyber/validate/yara</code></div>
  <div class="card"><strong>ATT&amp;CK Mapper</strong><br><code>POST /api/cyber/map/attack</code></div>
</div>
<h2>Copilot</h2>
<div class="card"><strong>Cyber Copilot Chat</strong> — <code>POST /api/cyber/copilot/chat</code><br>
Uses prompt packs (analyst, detection, report) with an Ollama-backed model.</div>"""
    return _page("Cyber", body, "/cyber")


# ── Optimize ─────────────────────────────────────────────────────

@app.get("/optimize", response_class=HTMLResponse)
async def optimize_page():
    body = """\
<h1>&#128640; Model Optimization</h1>
<p style="color:#8b949e;margin-bottom:1rem;">Quantize and compress models using state-of-the-art algorithms. Reduce memory, increase throughput, maintain quality.</p>

<h2>Environment Status</h2>
<div id="env-status" class="card"><span class="spinner"></span> Checking available quantization backends...</div>

<h2>Single Quantization</h2>
<div id="model-info-banner" style="display:none;padding:.75rem 1rem;border-radius:8px;margin-bottom:1rem;border:1px solid var(--border);"></div>
<div class="grid" style="margin:1rem 0;">
  <div class="form-group">
    <label>Source Model</label>
    <input type="text" id="q-model" placeholder="e.g. qwen2.5:7b" onblur="checkModelInfo(this.value)" />
  </div>
  <div class="form-group">
    <label>Backend</label>
    <select id="q-backend">
      <option value="ollama">Ollama GGUF</option>
      <option value="bnb_4bit">bitsandbytes NF4</option>
      <option value="bnb_8bit">bitsandbytes INT8</option>
      <option value="awq">AWQ (Activation-aware)</option>
      <option value="gptq">GPTQ (Hessian-based)</option>
    </select>
  </div>
  <div class="form-group">
    <label>GGUF Level</label>
    <select id="q-level">
      <option value="q4_k_m">Q4_K_M</option>
      <option value="q5_k_m">Q5_K_M</option>
      <option value="q8_0">Q8_0</option>
      <option value="q4_0">Q4_0</option>
      <option value="q3_k_m">Q3_K_M</option>
    </select>
  </div>
</div>
<button class="btn btn-purple" onclick="doQuantize()">&#128640; Quantize</button>
<div class="result-box" id="q-result"></div>

<h2>Compare All Methods</h2>
<p style="color:#8b949e;">Run quantization with every supported method and compare size, speed, and reduction. Unavailable methods are skipped with clear messages.</p>
<div class="form-group" style="max-width:400px;">
  <label>Source Model</label>
  <input type="text" id="qc-model" placeholder="e.g. qwen2.5:7b" />
</div>
<button class="btn btn-outline" style="border-color:var(--purple);color:var(--purple);" onclick="doQuantCompare()">&#128200; Compare All Methods</button>
<div class="result-box" id="qc-result"></div>

<h2>Algorithm Guide</h2>
<div class="card">
<table>
  <thead><tr><th>Method</th><th>Technique</th><th>Pros</th><th>Best For</th></tr></thead>
  <tbody>
    <tr><td><strong>GGUF (K-Quant)</strong></td><td>Mixed-precision quantization via llama.cpp</td><td>CPU+GPU, fast, many levels</td><td>Ollama, edge devices</td></tr>
    <tr><td><strong>AWQ</strong></td><td>Protects salient weight channels (top 1%%) based on activation distribution. Equivalent transformation scaling.</td><td>Best generalization, no backprop needed</td><td>GPU inference, multi-modal LLMs</td></tr>
    <tr><td><strong>GPTQ</strong></td><td>Layer-wise quantization using approximate second-order (Hessian) information</td><td>Very fast one-shot, proven</td><td>GPU-only inference</td></tr>
    <tr><td><strong>NF4</strong></td><td>4-bit NormalFloat from bitsandbytes with double quantization</td><td>Lowest memory, ideal for fine-tuning</td><td>QLoRA training, extreme compression</td></tr>
    <tr><td><strong>INT8</strong></td><td>LLM.int8() mixed-precision with outlier decomposition</td><td>Minimal quality loss</td><td>Quality-sensitive tasks</td></tr>
    <tr><td><strong>SparseGPT</strong></td><td>One-shot pruning using Hessian inverse. 50-60%% unstructured sparsity.</td><td>Complementary to quantization</td><td>Sparse + quantized combos</td></tr>
    <tr><td><strong>Wanda</strong></td><td>Pruning without gradients using weight and activation norms</td><td>Zero-cost, fast</td><td>Quick iteration</td></tr>
  </tbody>
</table>
</div>

<h2>&#9986; Model Pruning <span class="badge badge-purple" style="font-size:.7rem;vertical-align:middle;">HuggingFace Only</span></h2>
<p style="color:#8b949e;margin-bottom:.75rem;">Remove redundant weights to reduce model size. <strong>Requires a HuggingFace model ID</strong> (e.g. <code>Qwen/Qwen2.5-3B-Instruct</code>). Uses PyTorch &amp; Transformers &mdash; does not work with Ollama GGUF tags directly.</p>
<div class="grid" style="margin:1rem 0;">
  <div class="form-group">
    <label>Source Model (HF repo)</label>
    <input type="text" id="pr-model" placeholder="e.g. Qwen/Qwen2.5-3B-Instruct" />
  </div>
  <div class="form-group">
    <label>Pruning Method</label>
    <select id="pr-method">
      <option value="magnitude">Magnitude (Unstructured)</option>
      <option value="l1_structured">L1 Structured</option>
      <option value="random">Random (Baseline)</option>
    </select>
  </div>
  <div class="form-group">
    <label>Sparsity</label>
    <select id="pr-sparsity">
      <option value="0.2">20%%</option>
      <option value="0.3" selected>30%%</option>
      <option value="0.4">40%%</option>
      <option value="0.5">50%%</option>
      <option value="0.6">60%%</option>
    </select>
  </div>
</div>
<div style="display:flex;gap:.75rem;flex-wrap:wrap;">
  <button class="btn btn-purple" id="btn-prune" onclick="doPrune()">&#9986; Prune Model</button>
  <button class="btn btn-outline" style="border-color:var(--accent);color:var(--accent);" onclick="doPruneSuggest()">&#128161; Suggest Strategy</button>
</div>
<div class="result-box" id="pr-result"></div>

<h3>Pruning Suggestion</h3>
<p style="color:#8b949e;margin-bottom:.5rem;">Enter model size to see if pruning can help fit it into your hardware.</p>
<div class="form-group" style="max-width:300px;">
  <label>Model Size (Billions)</label>
  <input type="number" id="prs-params" placeholder="e.g. 7" step="0.1" min="0.1" />
</div>
<div class="result-box" id="prs-result"></div>

<h2>&#128260; Iterative Pruning with Verification <span class="badge badge-purple" style="font-size:.7rem;vertical-align:middle;">HuggingFace Only</span></h2>
<p style="color:#8b949e;margin-bottom:.75rem;">Sweep sparsity levels, verify perplexity at each step, auto-stop when quality degrades. <strong>Requires a HuggingFace model ID.</strong></p>
<div class="grid" style="margin:1rem 0;">
  <div class="form-group">
    <label>Source Model (HF repo)</label>
    <input type="text" id="ip-model" placeholder="e.g. Qwen/Qwen2.5-0.5B" />
  </div>
  <div class="form-group">
    <label>Method</label>
    <select id="ip-method">
      <option value="magnitude">Magnitude</option>
      <option value="l1_structured">L1 Structured</option>
      <option value="random">Random</option>
    </select>
  </div>
  <div class="form-group">
    <label>Max PPL Ratio</label>
    <select id="ip-maxppl">
      <option value="1.3">1.3x (Strict)</option>
      <option value="1.5" selected>1.5x (Balanced)</option>
      <option value="2.0">2.0x (Aggressive)</option>
    </select>
  </div>
</div>
<button class="btn btn-purple" id="btn-iter-prune" onclick="doIterativePrune()">&#128260; Run Iterative Prune</button>
<div class="result-box" id="ip-result"></div>

<h2>&#127891; Knowledge Distillation <span class="badge badge-purple" style="font-size:.7rem;vertical-align:middle;">HuggingFace Only</span></h2>
<p style="color:#8b949e;margin-bottom:.75rem;">Transfer knowledge from a large teacher model to a smaller student. <strong>Requires HuggingFace model IDs</strong> for both teacher and student. Uses PyTorch &amp; Transformers &mdash; does not work with Ollama GGUF tags.</p>
<div class="grid" style="margin:1rem 0;">
  <div class="form-group">
    <label>Teacher Model (HF repo)</label>
    <input type="text" id="ds-teacher" placeholder="e.g. Qwen/Qwen2.5-1.5B" />
  </div>
  <div class="form-group">
    <label>Student Model (HF repo)</label>
    <input type="text" id="ds-student" placeholder="e.g. sshleifer/tiny-gpt2" />
  </div>
  <div class="form-group">
    <label>Method</label>
    <select id="ds-method">
      <option value="logit">Logit (KL Divergence)</option>
      <option value="hidden">Hidden State (MSE)</option>
      <option value="progressive">Progressive (Iterative)</option>
    </select>
  </div>
  <div class="form-group">
    <label>Temperature</label>
    <input type="number" id="ds-temp" value="2.0" step="0.5" min="1" max="10" />
  </div>
  <div class="form-group">
    <label>Alpha (distill vs student loss)</label>
    <input type="number" id="ds-alpha" value="0.5" step="0.1" min="0" max="1" />
  </div>
  <div class="form-group">
    <label>Max Steps</label>
    <input type="number" id="ds-steps" value="200" step="50" min="10" max="5000" />
  </div>
</div>
<div style="display:flex;gap:.75rem;flex-wrap:wrap;">
  <button class="btn btn-purple" id="btn-distill" onclick="doDistill()">&#127891; Distill</button>
  <button class="btn btn-outline" style="border-color:var(--accent);color:var(--accent);" onclick="doDistillSuggest()">&#128161; Suggest Config</button>
</div>
<div class="result-box" id="ds-result"></div>

<h3>Distillation Suggestion</h3>
<div class="form-group" style="max-width:300px;">
  <label>Teacher Size (Billions)</label>
  <input type="number" id="dss-params" placeholder="e.g. 7" step="0.1" min="0.1" />
</div>
<div class="result-box" id="dss-result"></div>

<h2>&#128295; Model Editor <span class="badge badge-purple" style="font-size:.7rem;vertical-align:middle;">HuggingFace Only</span></h2>
<p style="color:#8b949e;margin-bottom:.75rem;">Surgical model modifications: remove layers, merge weights, resize vocabulary, prune attention heads. <strong>Requires a HuggingFace model ID.</strong></p>
<div class="grid" style="margin:1rem 0;">
  <div class="form-group">
    <label>Source Model (HF repo)</label>
    <input type="text" id="ed-model" placeholder="e.g. Qwen/Qwen2.5-0.5B" />
  </div>
  <div class="form-group">
    <label>Operation</label>
    <select id="ed-op" onchange="toggleEditFields()">
      <option value="layer_remove">Layer Removal</option>
      <option value="weight_merge">Weight Merge (SLERP/Linear)</option>
      <option value="vocab_resize">Vocabulary Resize</option>
      <option value="head_prune">Attention Head Pruning</option>
    </select>
  </div>
</div>
<div id="ed-layer-fields" class="grid" style="margin:.5rem 0;">
  <div class="form-group">
    <label>Layers to Remove (comma-separated)</label>
    <input type="text" id="ed-layers" placeholder="e.g. 12,13,14,15" />
  </div>
</div>
<div id="ed-merge-fields" class="grid" style="margin:.5rem 0;display:none;">
  <div class="form-group">
    <label>Merge Model (HF repo)</label>
    <input type="text" id="ed-merge-model" placeholder="e.g. another/model" />
  </div>
  <div class="form-group">
    <label>Method</label>
    <select id="ed-merge-method">
      <option value="linear">Linear Interpolation</option>
      <option value="slerp">SLERP</option>
    </select>
  </div>
  <div class="form-group">
    <label>Alpha</label>
    <input type="number" id="ed-merge-alpha" value="0.5" step="0.1" min="0" max="1" />
  </div>
</div>
<div id="ed-vocab-fields" class="form-group" style="max-width:300px;display:none;">
  <label>New Vocab Size</label>
  <input type="number" id="ed-vocab-size" placeholder="e.g. 32000" />
</div>
<div id="ed-head-fields" class="form-group" style="max-width:300px;display:none;">
  <label>Number of Heads to Prune (auto-detect)</label>
  <input type="number" id="ed-num-heads" placeholder="e.g. 4" min="1" />
</div>
<div style="display:flex;gap:.75rem;flex-wrap:wrap;">
  <button class="btn btn-purple" id="btn-edit" onclick="doEdit()">&#128295; Apply Edit</button>
  <button class="btn btn-outline" style="border-color:var(--accent);color:var(--accent);" onclick="doEditSuggest()">&#128161; Suggest Edits</button>
</div>
<div class="result-box" id="ed-result"></div>

<h2>&#128268; Smart Backend Router</h2>
<p style="color:#8b949e;margin-bottom:.75rem;">Intelligent routing inspired by multi-provider scoring: health-checks all optimization backends, scores by quality/cost/health, auto-fallback if preferred backend fails.</p>
<div id="route-status" class="card"><span class="spinner"></span> Checking backend health...</div>
<div class="grid" style="margin:1rem 0;">
  <div class="form-group">
    <label>Task Type</label>
    <select id="rt-task">
      <option value="quantize">Quantize</option>
      <option value="prune">Prune</option>
      <option value="distill">Distill</option>
      <option value="edit">Edit</option>
    </select>
  </div>
  <div class="form-group">
    <label>Model Size (Billions)</label>
    <input type="number" id="rt-params" placeholder="e.g. 7" step="0.1" min="0.1" />
  </div>
  <div class="form-group">
    <label>Strategy</label>
    <select id="rt-strategy">
      <option value="balanced" selected>Balanced</option>
      <option value="quality">Quality First</option>
      <option value="speed">Speed First</option>
    </select>
  </div>
</div>
<button class="btn btn-outline" style="border-color:var(--purple);color:var(--purple);" onclick="doRoute()">&#128268; Route Task</button>
<div class="result-box" id="rt-result"></div>
"""

    opt_js = """
// ── Pre-fill model inputs from global selection ──
(function() {
    const sel = CyberForge.getSelected();
    if (!sel) return;
    const ollamaId = sel.ollama_tag || '';
    const hfId = sel.hf_repo || '';
    // Quantize input: prefer ollama tag
    if (ollamaId) {
        const qm = document.getElementById('q-model');
        if (qm && !qm.value) { qm.value = ollamaId; checkModelInfo(ollamaId); }
        const qcm = document.getElementById('qc-model');
        if (qcm && !qcm.value) qcm.value = ollamaId;
    }
    // HF-based inputs: prefer hf_repo
    if (hfId) {
        ['pr-model','ip-model','ds-teacher','ed-model'].forEach(id => {
            const el = document.getElementById(id);
            if (el && !el.value) el.value = hfId;
        });
    }
    // Gate optimization backends by machine class
    const mc = CyberForge.getMachineClass();
    if (mc && mc.allowed_quant_backends) {
        const backendSelect = document.getElementById('q-backend');
        if (backendSelect) {
            for (const opt of backendSelect.options) {
                if (!mc.allowed_quant_backends.includes(opt.value)) {
                    opt.disabled = true;
                    opt.textContent += ' (not available on ' + mc.tier + ')';
                }
            }
        }
    }
    if (mc && mc.allowed_optimizations) {
        const allOps = ['prune','distill','edit'];
        allOps.forEach(op => {
            if (!mc.allowed_optimizations.includes(op)) {
                const btns = {prune: ['btn-prune','btn-iter-prune'], distill: ['btn-distill'], edit: ['btn-edit']};
                (btns[op]||[]).forEach(id => {
                    const btn = document.getElementById(id);
                    if (btn) { btn.disabled = true; btn.title = 'Not available on ' + mc.tier + ' machines'; btn.style.opacity = '0.4'; }
                });
            }
        });
    }
})();

// ── Model-aware gating ──
let _lastModelInfo = null;

async function checkModelInfo(modelName) {
    const banner = document.getElementById('model-info-banner');
    if (!modelName || !modelName.trim()) { banner.style.display = 'none'; _lastModelInfo = null; return; }
    try {
        const info = await cyberPost('/action/model-info', {model_name: modelName.trim()});
        _lastModelInfo = info;
        if (info._error) { banner.style.display = 'none'; return; }

        let html = '';
        const chips = [];

        if (info.is_ollama_tag) chips.push('<span class="badge badge-warn">Ollama Tag</span>');
        if (info.is_quantized) chips.push('<span class="badge badge-err">Already Quantized (' + (info.quant_level||'?') + ')</span>');
        if (info.has_hf_mapping) chips.push('<span class="badge badge-ok">HF: ' + info.hf_repo + '</span>');
        if (info.is_ollama_tag && !info.has_hf_mapping) chips.push('<span class="badge badge-err">No HF Mapping</span>');

        html += '<div style="display:flex;gap:.5rem;flex-wrap:wrap;align-items:center;margin-bottom:.5rem;">' + chips.join('') + '</div>';

        if (info.warnings && info.warnings.length) {
            html += info.warnings.map(w => '<div style="color:#d29922;font-size:.85rem;">&#9888; ' + w + '</div>').join('');
        }

        // Gate quantize backend options
        const backendSelect = document.getElementById('q-backend');
        if (backendSelect) {
            const allowed = info.available_backends || [];
            for (const opt of backendSelect.options) {
                opt.disabled = allowed.length > 0 && !allowed.includes(opt.value);
            }
        }

        // Gate pruning/distill/edit sections
        const needsHF = info.is_ollama_tag && !info.has_hf_mapping;
        ['pr-model','ip-model','ds-teacher','ed-model'].forEach(id => {
            const el = document.getElementById(id);
            if (el && !el.value) {
                // Auto-fill resolved HF repo
                if (info.has_hf_mapping) el.placeholder = info.hf_repo;
            }
        });

        // Show/hide feature warnings
        if (needsHF) {
            html += '<div style="color:#f85149;font-size:.85rem;margin-top:.25rem;">&#10060; Pruning, Distillation, and Editing require a HuggingFace model ID. This Ollama tag has no HF mapping.</div>';
        }

        // Disable/enable prune, distill, edit buttons based on HF availability
        ['btn-prune','btn-iter-prune','btn-distill','btn-edit'].forEach(id => {
            const btn = document.getElementById(id);
            if (btn) {
                btn.disabled = needsHF;
                btn.title = needsHF ? 'Requires a HuggingFace model (add mapping in registry.yaml or use HF repo ID)' : '';
                btn.style.opacity = needsHF ? '0.4' : '1';
            }
        });

        banner.innerHTML = html;
        banner.style.display = html ? '' : 'none';
    } catch(e) { banner.style.display = 'none'; }
}

// ── Env status on page load ──
(async function() {
    try {
        const status = await cyberGet('/action/quantize-status');
        if (status._error) { document.getElementById('env-status').innerHTML = '<span style="color:#f85149">' + status._error + '</span>'; return; }
        const hw = status._hardware || {};
        let html = '';
        // Hardware summary banner
        if (hw.gpu_name || hw.vram_total_mb) {
            html += '<div class="card" style="margin-bottom:1rem;border-color:var(--accent);">';
            html += '<strong style="color:var(--accent);">Hardware Detected</strong><div style="margin-top:.5rem;">';
            if (hw.gpu_name) html += '<span class="badge badge-ok">' + hw.gpu_name + '</span> ';
            if (hw.vram_total_mb) html += '<span class="badge">' + (hw.vram_total_mb/1024).toFixed(1) + ' GB VRAM</span> ';
            if (hw.vram_free_mb) html += '<span class="badge">' + (hw.vram_free_mb/1024).toFixed(1) + ' GB free</span> ';
            html += '<span class="badge ' + (hw.torch_cuda_available ? 'badge-ok' : 'badge-warn') + '">PyTorch CUDA: ' + (hw.torch_cuda_available ? 'Yes' : 'No') + '</span> ';
            html += '<span class="badge ' + (hw.nvidia_gpu_detected ? 'badge-ok' : 'badge-err') + '">nvidia-smi: ' + (hw.nvidia_gpu_detected ? 'Yes' : 'No') + '</span>';
            html += '</div></div>';
        }
        html += '<div class="grid">';
        const labels = {ollama_gguf:'Ollama GGUF', bnb_8bit:'bitsandbytes INT8', bnb_4bit:'bitsandbytes NF4', awq:'AWQ', gptq:'GPTQ'};
        for (const [method, info] of Object.entries(status)) {
            if (method.startsWith('_')) continue;
            const ok = info.available;
            const badge = ok ? 'badge-ok' : (info.status === 'dependency_missing' ? 'badge-warn' : 'badge-err');
            const text = ok ? 'Ready' : info.reason || 'Unavailable';
            const install = !ok && info.install ? '<br><code style="font-size:.75rem;color:#8b949e;">' + info.install + '</code>' : '';
            html += '<div class="card" style="text-align:center;"><strong>' + (labels[method]||method) + '</strong><br><span class="badge ' + badge + '" style="white-space:normal;">' + text + '</span>' + install + '</div>';
        }
        html += '</div>';
        document.getElementById('env-status').innerHTML = html;
    } catch(e) { document.getElementById('env-status').innerHTML = '<span style="color:#f85149">Failed to check status</span>'; }
})();

async function doQuantize() {
    showLoading('q-result');
    const model = document.getElementById('q-model').value;
    if (!model) { showResult('q-result', '<span style="color:#f85149">Enter a model name</span>'); return; }
    const backend = document.getElementById('q-backend').value;
    // Pre-submit gating: block already-quantized models on Ollama backend
    if (_lastModelInfo && _lastModelInfo.is_quantized && backend === 'ollama') {
        showResult('q-result', '<span style="color:#f85149">&#9888; This model is already quantized (' + (_lastModelInfo.quant_level||'unknown') + '). Ollama requires an F16/F32 source. Either pull the fp16 variant manually or use a HuggingFace backend (bnb/awq/gptq) with a mapped HF repo.</span>');
        return;
    }
    // Block Ollama-tag without HF mapping on non-ollama backends
    if (_lastModelInfo && _lastModelInfo.is_ollama_tag && !_lastModelInfo.has_hf_mapping && backend !== 'ollama') {
        showResult('q-result', '<span style="color:#f85149">&#9888; Ollama tag \'' + model + '\' has no HuggingFace mapping. The ' + backend + ' backend requires a HF repo ID. Add a mapping in registry.yaml or use a HF model name directly.</span>');
        return;
    }
    // ── Universal preflight: check backend availability + VRAM fit ──
    { const pf = await CyberForge.preflight('quantize', backend, 0, 0);
      if (pf && !pf.allowed) { showResult('q-result', '<span style="color:#f85149">&#9888; ' + (pf.reason||'Blocked') + '</span>' + (pf.suggestion ? '<br><span style="color:#8b949e">' + pf.suggestion + '</span>' : '')); return; } }
    try {
        const data = await cyberPost('/action/quantize', {source_model: model, backend: backend, quant_method: document.getElementById('q-level').value});
        if (data._error || data.error) { showResult('q-result', '<span style="color:#f85149">' + (data._error || data.error) + '</span>'); return; }
        showResult('q-result', '<h3 style="color:var(--green)">&#10003; Done</h3>' + buildTable(['Field','Value'],[['Output', data.output_model||'?'],['Method', data.method||'?'],['Size', data.size_bytes ? (data.size_bytes/1024/1024).toFixed(1)+' MB' : 'N/A'],['Duration', (data.duration_seconds||0).toFixed(1)+'s']]));
    } catch (e) { showResult('q-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}
async function doQuantCompare() {
    showLoading('qc-result');
    const model = document.getElementById('qc-model').value;
    if (!model) { showResult('qc-result', '<span style="color:#f85149">Enter a model name</span>'); return; }
    // Fetch model info if not already done for this model
    if (!_lastModelInfo || _lastModelInfo.model_name !== model.trim()) {
        try { await checkModelInfo(model); } catch(e) {}
    }
    if (_lastModelInfo && _lastModelInfo.is_quantized && _lastModelInfo.is_ollama_tag) {
        showResult('qc-result', '<span style="color:#f85149">&#9888; Model \'' + model + '\' is already quantized (' + (_lastModelInfo.quant_level||'?') + '). Compare requires an F16/F32 source. Pull the fp16 variant first or use a HuggingFace model ID.</span>');
        return;
    }
    // ── Preflight summary ──
    try {
        const pf = await cyberPost('/action/quantize-preflight', {source_model: model});
        if (pf && pf.hardware) {
            const hw = pf.hardware;
            const eligible = pf.eligible_methods || [];
            let pfHtml = '<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin-bottom:12px;">';
            pfHtml += '<strong style="color:var(--purple);">Preflight Summary</strong><div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-top:8px;">';
            pfHtml += '<div class="card" style="text-align:center;padding:8px;"><small>VRAM</small><br><strong>' + (hw.vram_total_mb ? (hw.vram_total_mb/1024).toFixed(1)+' GB' : 'No GPU') + '</strong>' + (hw.vram_free_mb ? '<br><small style="color:#8b949e">' + (hw.vram_free_mb/1024).toFixed(1)+' GB free</small>' : '') + '</div>';
            pfHtml += '<div class="card" style="text-align:center;padding:8px;"><small>RAM</small><br><strong>' + (hw.ram_total_mb/1024).toFixed(1)+' GB</strong><br><small style="color:#8b949e">' + (hw.ram_available_mb/1024).toFixed(1)+' GB free</small></div>';
            pfHtml += '<div class="card" style="text-align:center;padding:8px;"><small>OS</small><br><strong>' + (hw.os||'?') + '</strong></div>';
            pfHtml += '<div class="card" style="text-align:center;padding:8px;"><small>Eligible</small><br><strong>' + eligible.length + '/5</strong><br><small style="color:#8b949e">' + (eligible.join(', ')||'none') + '</small></div>';
            pfHtml += '</div>';
            // Show skipped methods
            const methods = pf.methods || {};
            const skipped = Object.entries(methods).filter(([_,v]) => !v.available);
            if (skipped.length) {
                pfHtml += '<div style="margin-top:8px;font-size:.8rem;color:#f0883e;">';
                skipped.forEach(([m, v]) => { pfHtml += '<div>&#9888; <strong>' + m + '</strong>: ' + (v.reason||'Unavailable') + '</div>'; });
                pfHtml += '</div>';
            }
            pfHtml += '</div>';
            document.getElementById('qc-result').innerHTML = pfHtml + '<div style="text-align:center;"><span class="loader"></span> Running comparison on ' + eligible.length + ' eligible method(s)...</div>';
        }
    } catch(pfErr) { /* preflight is optional, continue */ }

    try {
        const data = await cyberPost('/action/quantize-compare', {source_model: model});
        if (data._error) { showResult('qc-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        const entries = data.entries || [];
        if (entries.length) {
            const statusBadge = (e) => {
                if (e.success) return '<span class="badge badge-ok">&#10003; OK</span>';
                const cat = e.status || 'config_error';
                const catLabels = {unsupported_hardware:'Unsupported HW', service_unavailable:'Service Down', dependency_missing:'Missing Dep', config_error:'Config Error', model_error:'Model Error'};
                const catColors = {unsupported_hardware:'#f0883e', service_unavailable:'#da3633', dependency_missing:'#8b949e', config_error:'#f85149', model_error:'#f85149'};
                const label = catLabels[cat] || 'Failed';
                const color = catColors[cat] || '#f85149';
                return '<span class="badge" style="background:' + color + '22;color:' + color + ';border:1px solid ' + color + '55;" title="' + (e.error||'').replace(/"/g,'&quot;') + '">' + label + '</span><br><span style="font-size:.7rem;color:#8b949e;max-width:300px;display:inline-block;word-wrap:break-word;">' + (e.error||'') + '</span>';
            };
            const rows = entries.map(e => [e.method, e.quant_level||'', e.size_bytes?(e.size_bytes/1024/1024).toFixed(1)+'MB':'—', e.size_reduction_pct?e.size_reduction_pct.toFixed(1)+'%':'—', e.success?(e.duration_seconds||0).toFixed(1)+'s':'—', statusBadge(e)]);
            // Preserve preflight summary if it was shown
            const existing = document.getElementById('qc-result').innerHTML;
            const pfPart = existing.includes('Preflight Summary') ? existing.substring(0, existing.indexOf('</div><div style="text-align:center">') > -1 ? existing.lastIndexOf('<div style="text-align:center">') : existing.lastIndexOf('<div style="text-align:center;')) : '';
            showResult('qc-result', pfPart + '<h3 style="color:var(--purple)">Comparison Results</h3>' + buildTable(['Method','Level','Size','Reduction','Time','Status'], rows));
        } else { showResult('qc-result', 'No entries returned.'); }
    } catch (e) { showResult('qc-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}

async function doPrune() {
    showLoading('pr-result');
    const model = document.getElementById('pr-model').value;
    if (!model) { showResult('pr-result', '<span style="color:#f85149">Enter a model name</span>'); return; }
    // Warn if model looks like an Ollama tag (contains ':')
    if (model.includes(':') && !model.includes('/')) {
        showResult('pr-result', '<span style="color:#f85149">&#9888; Pruning requires a HuggingFace model ID (e.g. \'Qwen/Qwen2.5-7B\'). Ollama tags are not supported. Add a mapping in registry.yaml or use the HF repo name directly.</span>');
        return;
    }
    // ── Universal preflight: check transformers backend ──
    { const pf = await CyberForge.preflight('train', 'transformers', 0, 0);
      if (pf && !pf.allowed) { showResult('pr-result', '<span style="color:#f85149">&#9888; ' + (pf.reason||'Blocked') + '</span>' + (pf.suggestion ? '<br><span style="color:#8b949e">' + pf.suggestion + '</span>' : '')); return; } }
    try {
        const data = await cyberPost('/action/prune', {
            source_model: model,
            method: document.getElementById('pr-method').value,
            sparsity: parseFloat(document.getElementById('pr-sparsity').value)
        });
        if (data._error || data.error) { showResult('pr-result', '<span style="color:#f85149">' + (data._error||data.error) + '</span>'); return; }
        showResult('pr-result', '<h3 style="color:var(--green)">&#10003; Pruning Complete</h3>' +
            buildTable(['Field','Value'],[
                ['Output', data.output_model||'?'],
                ['Method', data.method||'?'],
                ['Sparsity', (data.sparsity*100).toFixed(0)+'%'],
                ['Zero Params', (data.zero_params||0).toLocaleString() + ' / ' + (data.original_params||0).toLocaleString()],
                ['Size', data.size_bytes ? (data.size_bytes/1024/1024).toFixed(1)+' MB' : 'N/A'],
                ['Reduction', (data.size_reduction_pct||0).toFixed(1)+'%'],
                ['Duration', (data.duration_seconds||0).toFixed(1)+'s']
            ]));
    } catch (e) { showResult('pr-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}

async function doPruneSuggest() {
    const params = parseFloat(document.getElementById('prs-params').value || document.getElementById('pr-model').value.match(/(\\d+\\.?\\d*)b/i)?.[1] || '0');
    if (!params) { showResult('prs-result', '<span style="color:#f85149">Enter model size in billions</span>'); return; }
    showLoading('prs-result');
    try {
        const data = await cyberPost('/action/prune-suggest', {params_b: params});
        if (data._error) { showResult('prs-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        const color = data.pruning_needed ? 'var(--yellow)' : 'var(--green)';
        showResult('prs-result', '<div style="color:' + color + ';font-weight:600;">' + data.message + '</div>' +
            (data.pruning_needed ? buildTable(['Metric','Value'],[
                ['Recommended Sparsity', (data.recommended_sparsity*100).toFixed(0)+'%'],
                ['Recommended Method', data.recommended_method],
                ['Original VRAM', data.original_vram_mb+' MB'],
                ['After Pruning', data.estimated_vram_after_mb+' MB'],
                ['Target VRAM', data.target_vram_mb+' MB']
            ]) : ''));
    } catch (e) { showResult('prs-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}

// ── Iterative Pruning ──
async function doIterativePrune() {
    showLoading('ip-result');
    const model = document.getElementById('ip-model').value;
    if (!model) { showResult('ip-result', '<span style="color:#f85149">Enter a model name</span>'); return; }
    // Warn if model looks like an Ollama tag
    if (model.includes(':') && !model.includes('/')) {
        showResult('ip-result', '<span style="color:#f85149">&#9888; Iterative pruning requires a HuggingFace model ID (e.g. \'Qwen/Qwen2.5-7B\'). Ollama tags are not supported.</span>');
        return;
    }
    // ── Universal preflight: check transformers backend ──
    { const pf = await CyberForge.preflight('train', 'transformers', 0, 0);
      if (pf && !pf.allowed) { showResult('ip-result', '<span style="color:#f85149">&#9888; ' + (pf.reason||'Blocked') + '</span>' + (pf.suggestion ? '<br><span style="color:#8b949e">' + pf.suggestion + '</span>' : '')); return; } }
    try {
        const data = await cyberPost('/action/prune-iterative', {
            source_model: model,
            method: document.getElementById('ip-method').value,
            max_perplexity_ratio: parseFloat(document.getElementById('ip-maxppl').value)
        });
        if (data._error || data.error) { showResult('ip-result', '<span style="color:#f85149">' + (data._error||data.error) + '</span>'); return; }
        let html = '<h3 style="color:var(--green)">Iterative Pruning Complete</h3>';
        html += '<div style="margin:.5rem 0;"><strong>Best sparsity:</strong> ' + (data.best_sparsity*100).toFixed(0) + '% | <strong>PPL ratio:</strong> ' + data.best_perplexity_ratio + 'x | <strong>Baseline PPL:</strong> ' + data.baseline_perplexity + '</div>';
        if (data.output_model) html += '<div><strong>Output:</strong> ' + data.output_model + '</div>';
        if (data.steps && data.steps.length) {
            const rows = data.steps.map(s => [
                (s.sparsity*100).toFixed(0)+'%',
                s.perplexity.toFixed(2),
                s.perplexity_ratio.toFixed(3)+'x',
                s.zero_params.toLocaleString(),
                s.passed ? '<span class="badge badge-ok">PASS</span>' : '<span class="badge badge-err">FAIL</span>'
            ]);
            html += buildTable(['Sparsity','Perplexity','PPL Ratio','Zero Params','Quality Gate'], rows);
        }
        html += '<div style="margin-top:.5rem;color:#8b949e;">Duration: ' + (data.duration_seconds||0).toFixed(1) + 's</div>';
        showResult('ip-result', html);
    } catch (e) { showResult('ip-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}

// ── Distillation ──
async function doDistill() {
    showLoading('ds-result');
    const teacher = document.getElementById('ds-teacher').value;
    const student = document.getElementById('ds-student').value;
    if (!teacher || !student) { showResult('ds-result', '<span style="color:#f85149">Enter both teacher and student model names</span>'); return; }
    // Warn if models look like Ollama tags
    if ((teacher.includes(':') && !teacher.includes('/')) || (student.includes(':') && !student.includes('/'))) {
        showResult('ds-result', '<span style="color:#f85149">&#9888; Distillation requires HuggingFace model IDs (e.g. \'Qwen/Qwen2.5-7B\'). Ollama tags are not supported.</span>');
        return;
    }
    // ── Universal preflight: check transformers backend ──
    { const pf = await CyberForge.preflight('train', 'transformers', 0, 0);
      if (pf && !pf.allowed) { showResult('ds-result', '<span style="color:#f85149">&#9888; ' + (pf.reason||'Blocked') + '</span>' + (pf.suggestion ? '<br><span style="color:#8b949e">' + pf.suggestion + '</span>' : '')); return; } }
    try {
        const data = await cyberPost('/action/distill', {
            teacher_model: teacher, student_model: student,
            method: document.getElementById('ds-method').value,
            temperature: parseFloat(document.getElementById('ds-temp').value),
            alpha: parseFloat(document.getElementById('ds-alpha').value),
            max_steps: parseInt(document.getElementById('ds-steps').value)
        });
        if (data._error || data.error) { showResult('ds-result', '<span style="color:#f85149">' + (data._error||data.error) + '</span>'); return; }
        showResult('ds-result', '<h3 style="color:var(--green)">&#10003; Distillation Complete</h3>' +
            buildTable(['Field','Value'],[
                ['Output', data.output_model||'?'],
                ['Method', data.method||'?'],
                ['Teacher', data.teacher_model + ' (' + (data.teacher_params||0).toLocaleString() + ' params)'],
                ['Student', data.student_model + ' (' + (data.student_params||0).toLocaleString() + ' params)'],
                ['Compression', (data.compression_ratio*100).toFixed(1) + '%'],
                ['Final Loss', (data.final_loss||0).toFixed(4)],
                ['Steps', data.steps_completed||0],
                ['Teacher PPL', (data.teacher_perplexity||0).toFixed(2)],
                ['Student PPL', (data.student_perplexity||0).toFixed(2)],
                ['PPL Ratio', (data.perplexity_ratio||0).toFixed(2) + 'x'],
                ['Size', data.size_bytes ? (data.size_bytes/1024/1024).toFixed(1)+' MB' : 'N/A'],
                ['Duration', (data.duration_seconds||0).toFixed(1)+'s']
            ]));
    } catch (e) { showResult('ds-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}

async function doDistillSuggest() {
    const params = parseFloat(document.getElementById('dss-params').value||'0');
    if (!params) { showResult('dss-result', '<span style="color:#f85149">Enter teacher size in billions</span>'); return; }
    showLoading('dss-result');
    try {
        const data = await cyberPost('/action/distill-suggest', {teacher_params_b: params});
        if (data._error) { showResult('dss-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        showResult('dss-result', '<div style="color:var(--accent);font-weight:600;">' + data.message + '</div>' +
            buildTable(['Metric','Value'],[
                ['Recommended Device', data.recommended_device],
                ['Recommended Method', data.recommended_method],
                ['Target Student Size', data.target_student_params_b + 'B'],
                ['Teacher VRAM', data.teacher_vram_mb + ' MB'],
                ['Student VRAM', data.student_vram_mb + ' MB'],
                ['Combined VRAM', data.combined_vram_mb + ' MB'],
                ['Available VRAM', data.available_vram_mb + ' MB'],
                ['Fits GPU', data.fits_gpu ? 'Yes' : 'No'],
                ['Student Suggestions', (data.student_suggestions||[]).join(', ') || 'N/A']
            ]));
    } catch (e) { showResult('dss-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}

// ── Model Editor ──
function toggleEditFields() {
    const op = document.getElementById('ed-op').value;
    document.getElementById('ed-layer-fields').style.display = op === 'layer_remove' ? '' : 'none';
    document.getElementById('ed-merge-fields').style.display = op === 'weight_merge' ? '' : 'none';
    document.getElementById('ed-vocab-fields').style.display = op === 'vocab_resize' ? '' : 'none';
    document.getElementById('ed-head-fields').style.display = op === 'head_prune' ? '' : 'none';
}

async function doEdit() {
    showLoading('ed-result');
    const model = document.getElementById('ed-model').value;
    const op = document.getElementById('ed-op').value;
    if (!model) { showResult('ed-result', '<span style="color:#f85149">Enter a model name</span>'); return; }
    // Warn if model looks like an Ollama tag
    if (model.includes(':') && !model.includes('/')) {
        showResult('ed-result', '<span style="color:#f85149">&#9888; Model editing requires a HuggingFace model ID (e.g. \'Qwen/Qwen2.5-7B\'). Ollama tags are not supported.</span>');
        return;
    }
    // ── Universal preflight: check transformers backend ──
    { const pf = await CyberForge.preflight('train', 'transformers', 0, 0);
      if (pf && !pf.allowed) { showResult('ed-result', '<span style="color:#f85149">&#9888; ' + (pf.reason||'Blocked') + '</span>' + (pf.suggestion ? '<br><span style="color:#8b949e">' + pf.suggestion + '</span>' : '')); return; } }
    let payload = {source_model: model, operation: op};
    if (op === 'layer_remove') {
        const layers = document.getElementById('ed-layers').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
        payload.layers_to_remove = layers;
    } else if (op === 'weight_merge') {
        payload.merge_model = document.getElementById('ed-merge-model').value;
        payload.merge_method = document.getElementById('ed-merge-method').value;
        payload.merge_alpha = parseFloat(document.getElementById('ed-merge-alpha').value);
    } else if (op === 'vocab_resize') {
        payload.new_vocab_size = parseInt(document.getElementById('ed-vocab-size').value);
    } else if (op === 'head_prune') {
        payload.num_heads_to_prune = parseInt(document.getElementById('ed-num-heads').value);
    }
    try {
        const data = await cyberPost('/action/edit', payload);
        if (data._error || data.error) { showResult('ed-result', '<span style="color:#f85149">' + (data._error||data.error) + '</span>'); return; }
        showResult('ed-result', '<h3 style="color:var(--green)">&#10003; Edit Complete</h3>' +
            buildTable(['Field','Value'],[
                ['Output', data.output_model||'?'],
                ['Operation', data.operation||'?'],
                ['Original Params', (data.original_params||0).toLocaleString()],
                ['Edited Params', (data.edited_params||0).toLocaleString()],
                ['Param Reduction', (data.param_reduction_pct||0).toFixed(1)+'%'],
                ['Layers', (data.original_layers||0) + ' → ' + (data.edited_layers||0)],
                ['Vocab', (data.original_vocab||0) + ' → ' + (data.edited_vocab||0)],
                ['Size', data.size_bytes ? (data.size_bytes/1024/1024).toFixed(1)+' MB' : 'N/A'],
                ['Duration', (data.duration_seconds||0).toFixed(1)+'s']
            ]));
    } catch (e) { showResult('ed-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}

async function doEditSuggest() {
    const params = parseFloat(document.getElementById('ed-model').value.match(/(\\d+\\.?\\d*)b/i)?.[1] || '0');
    if (!params) { showResult('ed-result', '<span style="color:#f85149">Cannot infer model size — use Suggest via distillation section</span>'); return; }
    showLoading('ed-result');
    try {
        const data = await cyberPost('/action/edit-suggest', {params_b: params});
        if (data._error) { showResult('ed-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        const color = data.editing_needed ? 'var(--yellow)' : 'var(--green)';
        showResult('ed-result', '<div style="color:' + color + ';font-weight:600;">' + data.message + '</div>');
    } catch (e) { showResult('ed-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}

// ── Smart Router ──
(async function() {
    try {
        const status = await cyberGet('/action/route-status');
        if (status._error) { document.getElementById('route-status').innerHTML = '<span style="color:#f85149">' + status._error + '</span>'; return; }
        let html = '<h3>Backend Health</h3><div class="grid">';
        const backends = status.backends || {};
        for (const [name, info] of Object.entries(backends)) {
            const ok = info.healthy;
            const badge = ok ? 'badge-ok' : 'badge-err';
            const supports = Object.entries(info.supports||{}).filter(([k,v]) => v).map(([k]) => k).join(', ') || 'none';
            const errText = !ok && info.last_error ? '<br><span style="font-size:.7rem;color:#f85149;">' + info.last_error + '</span>' : '';
            html += '<div class="card" style="text-align:center;padding:.75rem;"><strong>' + info.display_name + '</strong><br>';
            html += '<span class="badge ' + badge + '">' + (ok?'Healthy':'Unavailable') + '</span>';
            html += '<br><span style="font-size:.75rem;color:#8b949e;">Q:' + info.quality_score + ' C:' + info.cost_score + '</span>';
            html += '<br><span style="font-size:.7rem;color:#8b949e;">' + supports + '</span>';
            html += errText + '</div>';
        }
        html += '</div><div style="margin-top:.5rem;color:#8b949e;font-size:.85rem;">Strategy: ' + status.strategy + ' | Fallback: ' + (status.fallback_enabled?'On':'Off') + '</div>';
        document.getElementById('route-status').innerHTML = html;
    } catch(e) { document.getElementById('route-status').innerHTML = '<span style="color:#f85149">Failed to check router status</span>'; }
})();

async function doRoute() {
    showLoading('rt-result');
    try {
        const data = await cyberPost('/action/route', {
            task: document.getElementById('rt-task').value,
            model_params_b: parseFloat(document.getElementById('rt-params').value||'0'),
            strategy: document.getElementById('rt-strategy').value
        });
        if (data._error) { showResult('rt-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        let html = '<h3 style="color:var(--accent)">Route Decision</h3>';
        html += '<div style="font-size:1.1rem;font-weight:600;margin:.5rem 0;">' + (data.backend || 'No backend available') + '</div>';
        html += '<div style="color:#8b949e;">' + (data.reason||'') + '</div>';
        html += '<div style="margin-top:.5rem;">Score: ' + (data.score||0).toFixed(3) + ' | Latency: ' + (data.latency_ms||0).toFixed(0) + 'ms</div>';
        if (data.fallback_chain && data.fallback_chain.length) {
            html += '<div style="margin-top:.25rem;color:#8b949e;">Fallback chain: ' + data.fallback_chain.join(' → ') + '</div>';
        }
        showResult('rt-result', html);
    } catch (e) { showResult('rt-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}
"""
    return _page("Optimize", body, "/optimize", extra_js=opt_js)


# ── Cache ────────────────────────────────────────────────────────

@app.get("/cache", response_class=HTMLResponse)
async def cache_page():
    cache = await _api("/api/lifecycle/cache")
    saved = await _api("/api/lifecycle/saved")
    disk = await _api("/api/lifecycle/disk-usage")

    cache_list = cache if isinstance(cache, list) else []
    saved_list = saved if isinstance(saved, list) else []

    cache_rows = ""
    for c in cache_list:
        name = c.get("name", c.get("path", "?"))
        size = c.get("size_mb", c.get("size_bytes", 0))
        if isinstance(size, (int, float)) and size > 1000:
            size = f"{size / 1024:.1f} GB"
        else:
            size = f"{size} MB" if size else "?"
        cache_rows += f"<tr><td>{name}</td><td>{size}</td><td><button class='btn btn-sm btn-red' onclick=\"discardItem('{name}')\">Delete</button></td></tr>"

    saved_rows = ""
    for s in saved_list:
        name = s.get("name", s.get("path", "?"))
        size = s.get("size_mb", s.get("size_bytes", 0))
        saved_rows += f"<tr><td>{name}</td><td>{size}</td></tr>"

    disk_info = ""
    if isinstance(disk, dict):
        for k, v in disk.items():
            disk_info += f"<div class='card kpi'><div class='value'>{v}</div><div class='label'>{k}</div></div>"

    body = f"""\
<h1>&#128451; Cache &amp; Storage</h1>

<h2>Disk Usage</h2>
<div class="grid">{disk_info}</div>

<h2>Cached Artifacts</h2>
<div class="card">
{"<table><thead><tr><th>Name</th><th>Size</th><th>Action</th></tr></thead><tbody>" + cache_rows + "</tbody></table>" if cache_rows else "<p>No cached artifacts.</p>"}
</div>

<h2>Saved Artifacts</h2>
<div class="card">
{"<table><thead><tr><th>Name</th><th>Size</th></tr></thead><tbody>" + saved_rows + "</tbody></table>" if saved_rows else "<p>No saved artifacts.</p>"}
</div>

<h2>Cleanup Actions</h2>
<div style="display:flex;gap:.75rem;flex-wrap:wrap;margin:1rem 0;">
  <button class="btn btn-red" onclick="volatileClean()">&#128465; Cleanup Volatile</button>
  <button class="btn btn-outline" onclick="staleClean()">&#128336; Remove Stale (&gt;24h)</button>
</div>
<div class="result-box" id="cleanup-result"></div>
"""

    cache_js = """
async function discardItem(name) {
    if (!confirm('Delete ' + name + '?')) return;
    try {
        const data = await cyberPost('/action/discard', {cache_name: name});
        if (data._error) alert(data._error); else location.reload();
    } catch (e) { alert(e.message); }
}
async function volatileClean() {
    showLoading('cleanup-result');
    const data = await cyberPost('/action/cache-cleanup');
    showResult('cleanup-result', '<span style="color:var(--green)">Cleaned ' + (data.deleted_count||0) + ' volatile artifact(s). Reloading...</span>');
    setTimeout(() => location.reload(), 1500);
}
async function staleClean() {
    showLoading('cleanup-result');
    const data = await cyberPost('/action/cache-stale');
    showResult('cleanup-result', '<span style="color:var(--green)">Removed ' + (data.deleted_count||0) + ' stale artifact(s). Reloading...</span>');
    setTimeout(() => location.reload(), 1500);
}
"""
    return _page("Cache", body, "/cache", extra_js=cache_js)


# ── Jobs ─────────────────────────────────────────────────────────

@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page():
    data = await _api("/api/v1/jobs/")
    err = _err(data)
    if err:
        return _page("Jobs", f"<h1>Jobs</h1>{err}", "/jobs")

    job_list = data.get("jobs", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    if not job_list:
        body = '<h1>Jobs</h1><div class="card">No jobs in queue.</div>'
        return _page("Jobs", body, "/jobs")

    rows = ""
    for j in job_list:
        jid = j.get("job_id", j.get("id", "?"))
        status = j.get("status", "?")
        badge = "badge-ok" if status in ("completed", "done") else "badge-warn" if status in ("running", "pending") else "badge-err"
        jtype = j.get("job_type", j.get("type", "—"))
        rows += f'<tr><td>{jid}</td><td>{jtype}</td><td><span class="badge {badge}">{status}</span></td></tr>'

    body = f"""\
<h1>Jobs</h1>
<div class="card">
<table>
<thead><tr><th>Job ID</th><th>Type</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""
    return _page("Jobs", body, "/jobs")


# ── Chat ─────────────────────────────────────────────────────────

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    body = """\
<h1>&#128172; CyberForge Chat</h1>
<p style="color:#8b949e;margin-bottom:1rem;">Don't know which model to use? Describe your use case and get personalized recommendations. Powered by your local Ollama model.</p>

<div id="ollama-banner" class="card" style="border-color:var(--yellow);display:none;">
  <strong style="color:var(--yellow);">&#9888; Ollama Status</strong>
  <span id="ollama-banner-text"></span>
  <button class="btn btn-sm btn-green" style="margin-left:.75rem;" onclick="startOllama()">Start Ollama</button>
</div>

<div class="card">
  <div style="margin-bottom:.5rem;">
    <label>Chat Model</label>
    <div style="display:flex;gap:.5rem;">
      <select id="chat-model-select" style="flex:1;"><option value="">Loading models...</option></select>
      <button class="btn btn-sm btn-outline" onclick="refreshChatModels()">&#8635;</button>
    </div>
  </div>
  <div class="chat-box" id="chat-box">
    <div class="chat-msg chat-bot">
      <strong>CyberForge</strong><br>
      Hi! I can help you pick the best model for your use case. Tell me:<br>
      &bull; What task do you need? (coding, security analysis, general chat, etc.)<br>
      &bull; Any constraints? (max memory, speed requirements, etc.)<br>
      &bull; Or just say "recommend" and I'll analyze your hardware and suggest the best options.
    </div>
  </div>
  <div class="chat-input-row" style="margin-top:.75rem;">
    <input type="text" id="chat-input" placeholder="Describe your use case or ask a question..." onkeydown="if(event.key==='Enter')sendChat()" />
    <button class="btn btn-primary" onclick="sendChat()">Send</button>
  </div>
</div>

<h2>Quick Actions</h2>
<div class="grid">
  <div class="card" style="cursor:pointer;" onclick="quickChat('What model should I use for code generation on a 6GB VRAM GPU?')">
    <strong>&#128187; Best for Coding</strong><br><span style="color:#8b949e;">Code generation on limited VRAM</span>
  </div>
  <div class="card" style="cursor:pointer;" onclick="quickChat('I need a model for cyber security threat analysis. What do you recommend?')">
    <strong>&#128737; Best for Cyber</strong><br><span style="color:#8b949e;">Threat analysis &amp; Sigma rules</span>
  </div>
  <div class="card" style="cursor:pointer;" onclick="quickChat('Recommend the fastest model I can run locally for general chat.')">
    <strong>&#9889; Fastest Local</strong><br><span style="color:#8b949e;">Lowest latency general chat</span>
  </div>
  <div class="card" style="cursor:pointer;" onclick="quickChat('Should I quantize my model? What method gives the best quality vs size tradeoff?')">
    <strong>&#128640; Quantization Advice</strong><br><span style="color:#8b949e;">Compression strategy guidance</span>
  </div>
</div>
"""

    chat_js = """
let _chatModel = null;
async function refreshChatModels() {
    const sel = document.getElementById('chat-model-select');
    const mc = CyberForge.getMachineClass();
    const safeLimit = mc && mc.vram_total_mb ? mc.vram_total_mb * 1024 * 1024 * 0.85 : 5 * 1024 * 1024 * 1024;
    try {
        const models = await cyberGet('/action/ollama-models');
        const list = Array.isArray(models) ? models : (models.models || []);
        if (list.length > 0) {
            sel.innerHTML = '';
            // Sort ascending by size — safer (smaller) models first
            list.sort((a, b) => (a.size || 0) - (b.size || 0));
            const selectedId = CyberForge.getModelId();
            let safeIdx = -1;
            let selectedIdx = -1;
            list.forEach((m, idx) => {
                const name = m.name || m;
                const sizeGB = m.size ? (m.size / 1024 / 1024 / 1024).toFixed(1) : '?';
                const safe = m.size && m.size <= safeLimit;
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name + ' (' + sizeGB + ' GB)' + (safe ? '' : ' \u26a0\ufe0f may not fit VRAM');
                sel.appendChild(opt);
                if (safe && safeIdx < 0) safeIdx = idx;
                if (selectedId && name === selectedId) selectedIdx = idx;
            });
            // Prefer the globally selected model, else first safe model
            if (selectedIdx >= 0) sel.selectedIndex = selectedIdx;
            else if (safeIdx >= 0) sel.selectedIndex = safeIdx;
            _chatModel = sel.value;
            document.getElementById('ollama-banner').style.display = 'none';
        } else {
            sel.innerHTML = '<option value="">No models found</option>';
            _chatModel = null;
        }
    } catch (e) {
        sel.innerHTML = '<option value="">Failed to load models</option>';
        _chatModel = null;
        // Check if Ollama is even running
        checkOllamaStatus();
    }
}
async function checkOllamaStatus() {
    try {
        const status = await cyberGet('/action/ollama-status');
        if (!status.available) {
            const banner = document.getElementById('ollama-banner');
            document.getElementById('ollama-banner-text').textContent = ' Ollama is not running.';
            banner.style.display = '';
        } else {
            document.getElementById('ollama-banner').style.display = 'none';
        }
    } catch(e) {
        const banner = document.getElementById('ollama-banner');
        document.getElementById('ollama-banner-text').textContent = ' Cannot reach Ollama.';
        banner.style.display = '';
    }
}
async function startOllama() {
    const banner = document.getElementById('ollama-banner');
    document.getElementById('ollama-banner-text').innerHTML = ' <span class="spinner"></span> Starting Ollama...';
    try {
        const result = await cyberPost('/action/ollama-start');
        if (result.started) {
            document.getElementById('ollama-banner-text').textContent = ' Ollama started! Refreshing models...';
            banner.style.borderColor = 'var(--green)';
            setTimeout(async () => { await refreshChatModels(); banner.style.display = 'none'; }, 2000);
        } else {
            document.getElementById('ollama-banner-text').textContent = ' Failed: ' + (result.error || 'Unknown error');
        }
    } catch(e) { document.getElementById('ollama-banner-text').textContent = ' Error: ' + e.message; }
}
// Init: load models, check Ollama
refreshChatModels();
document.getElementById('chat-model-select').addEventListener('change', function() { _chatModel = this.value; });

async function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    // Use selected model from dropdown
    _chatModel = document.getElementById('chat-model-select').value;
    if (!_chatModel) {
        await refreshChatModels();
        _chatModel = document.getElementById('chat-model-select').value;
    }
    if (!_chatModel) {
        const box = document.getElementById('chat-box');
        box.innerHTML += '<div class="chat-msg chat-bot"><strong>CyberForge</strong><br><span style="color:#f85149">No Ollama model available. Is Ollama running? Click "Start Ollama" above or run <code>ollama serve</code> in a terminal.</span></div>';
        box.scrollTop = box.scrollHeight;
        checkOllamaStatus();
        return;
    }
    input.value = '';
    const box = document.getElementById('chat-box');
    box.innerHTML += '<div class="chat-msg chat-user"><strong>You</strong><br>' + msg.replace(/</g,'&lt;') + '</div>';
    box.scrollTop = box.scrollHeight;
    // ── Universal preflight: check ollama backend availability ──
    { const pf = await CyberForge.preflight('chat', 'ollama', 0, 0);
      if (pf && !pf.allowed) {
        box.innerHTML += '<div class="chat-msg chat-bot"><strong>CyberForge</strong><br><span style="color:#f85149">&#9888; ' + (pf.reason||'Backend unavailable') + '</span>' + (pf.suggestion ? '<br><span style="color:#8b949e">' + pf.suggestion + '</span>' : '') + '</div>';
        box.scrollTop = box.scrollHeight; return;
      } }
    // ── Model-fit preflight ──
    try {
        const fit = await cyberPost('/action/model-fit', {model_name: _chatModel});
        if (fit && !fit._error && fit.category === 'too_large') {
            box.innerHTML += '<div class="chat-msg chat-bot"><strong>CyberForge</strong><br><span style="color:#f85149">&#9888; ' + fit.reason + ' Choose a smaller model from the dropdown.</span></div>';
            box.scrollTop = box.scrollHeight;
            return;
        }
    } catch(e) { /* preflight optional */ }
    box.innerHTML += '<div class="chat-msg chat-bot" id="chat-loading"><span class="spinner"></span> Thinking with <strong>' + _chatModel + '</strong>... (first message may take longer while model loads)</div>';
    box.scrollTop = box.scrollHeight;
    try {
        const data = await cyberPost('/action/chat', {model: _chatModel, messages: [{role: 'system', content: 'You are CyberForge, an AI model optimization assistant. Help users pick the best AI model for their hardware and use case. You know about quantization (GGUF, AWQ, GPTQ, NF4), pruning (SparseGPT, Wanda), and benchmarking.'}, {role: 'user', content: msg}]});
        const el = document.getElementById('chat-loading');
        if (!el) return;
        if (data._error) {
            el.innerHTML = '<strong>CyberForge</strong><br><span style="color:#f85149">' + data._error + '</span>';
        } else {
            const reply = (data.message && typeof data.message === 'object' ? data.message.content : data.message) || data.content || (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) || JSON.stringify(data);
            el.innerHTML = '<strong>CyberForge</strong><br>' + String(reply).replace(/\\n/g, '<br>');
        }
        el.removeAttribute('id');
    } catch (e) {
        const el = document.getElementById('chat-loading');
        if (el) { el.innerHTML = '<strong>CyberForge</strong><br><span style="color:#f85149">Error: ' + e.message + '</span>'; el.removeAttribute('id'); }
    }
    box.scrollTop = box.scrollHeight;
}
function quickChat(msg) {
    document.getElementById('chat-input').value = msg;
    sendChat();
}
"""
    return _page("Chat", body, "/chat", extra_js=chat_js)
