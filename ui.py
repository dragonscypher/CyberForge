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
    ("/optimize", "Optimize &#128274;"),
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
    sessionStorage.setItem('cf_selected_mode', mode);
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


@app.post("/action/ollama-delete")
async def action_ollama_delete(request: Request):
    body = await request.json()
    name = body.get("model_name", "")
    if not name:
        return JSONResponse({"deleted": False, "error": "No model_name provided"})
    try:
        ollama = getattr(app.state, "ollama", None) or getattr(_api_app.state, "ollama", None)
        if ollama is None:
            return JSONResponse({"deleted": False, "error": "Ollama client not available"})
        ok = await ollama.delete(name)
        return JSONResponse({"deleted": ok, "model_name": name})
    except Exception as e:
        return JSONResponse({"deleted": False, "error": str(e)})

@app.post("/action/ollama-pull")
async def action_ollama_pull(request: Request):
    body = await request.json()
    return JSONResponse(await _api("/api/serve/ollama/pull", method="POST", json=body))

@app.get("/action/ollama-library-search")
async def action_ollama_library_search(q: str = ""):
    """Search the Ollama model library at ollama.com for matching models."""
    import re as _re
    if not q or len(q) < 2:
        return JSONResponse({"results": []})
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get("https://ollama.com/search", params={"q": q})
            if resp.status_code != 200:
                return JSONResponse({"results": [], "error": "Ollama library search unavailable"})
            # Extract model names from library links: /library/modelname
            matches = _re.findall(r'href="/library/([^"]+)"', resp.text)
            seen = set()
            results = []
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    results.append(m)
            return JSONResponse({"results": results[:10]})
    except Exception as exc:
        return JSONResponse({"results": [], "error": str(exc)})


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
  <div class="card"><a href="/optimize">&#128640; Optimize <span style="font-size:.75rem;background:#6e40c9;color:#fff;padding:2px 6px;border-radius:4px;">Advanced</span></a> — all backends (expert)</div>
  <div class="card"><a href="/cache">&#128451; Cache</a> — manage storage</div>
  <div class="card"><a href="/cyber">&#128737; Cyber Lab</a> — validate Sigma/YARA/Suricata</div>
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
  <span class="step-pill" id="sp5">5. Optimize</span>
  <span class="step-pill" id="sp6">6. Benchmark</span>
  <span class="step-pill" id="sp7">7. Results</span>
</div>

<!-- STEP 1: Hardware -->
<div class="card" id="step1">
  <h2>Step 1 — Hardware Detected</h2>
  {hw_html}
  <p style="color:var(--green);margin-top:.75rem;">&#10003; Hardware profiled automatically.</p>
  <button class="btn btn-primary" onclick="setStep(2)" style="margin-top:.75rem;">Continue &rarr; Select Task Mode</button>
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
  <!-- Dataset / Benchmark info per mode -->
  <div id="mode-dataset-info" class="card" style="border-color:var(--border);margin-top:.75rem;display:none;"></div>
  <div id="mode-continue" style="display:none;margin-top:.75rem;">
    <button class="btn btn-primary" onclick="setStep(3)">Continue &rarr; Run Tests</button>
    <button class="btn btn-outline" onclick="setStep(4)" style="margin-left:.5rem;">Skip Tests &rarr; Get Recommendations</button>
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

<!-- STEP 5: Optimize & Quantize -->
<div class="card" id="step5" style="border-color:var(--purple);">
  <h2 style="color:var(--purple);">Step 5 — Quantize &amp; Optimize</h2>
  <p style="color:#8b949e;">Apply GGUF quantization to compress your model for local hardware.</p>
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

  <div style="display:flex;gap:.75rem;flex-wrap:wrap;margin:1rem 0;">
    <button class="btn btn-purple" onclick="runQuantize()">&#128640; Quantize Model</button>
    <button class="btn btn-outline" onclick="runQuantCompare()" style="border-color:var(--purple);color:var(--purple);">&#128200; Compare All Methods</button>
  </div>
  <div class="result-box" id="quant-result"></div>

  <!-- ── Optimization: Quantize (Ollama GGUF) ── -->

  <!-- Quant info (shown by default) -->
  <div id="opt-panel-quant" class="card" style="background:#1a1025;border-color:var(--purple);">
    <h3 style="color:var(--purple);">&#128218; GGUF Quantization</h3>
    <p style="color:#8b949e;font-size:.85rem;">Ollama GGUF uses K-quant mixed-precision via llama.cpp &mdash; ideal for CPU+GPU inference on local hardware.</p>
  </div>

  <!-- Optional: Cache Cleanup -->
  <details style="margin-top:1rem;">
    <summary style="color:#8b949e;cursor:pointer;font-size:.85rem;">&#128465; Cache Management (optional)</summary>
    <div style="display:flex;gap:.75rem;flex-wrap:wrap;margin:.75rem 0;">
      <button class="btn btn-red" onclick="cleanupCache()" style="font-size:.8rem;">&#128465; Cleanup Volatile Cache</button>
      <button class="btn btn-outline" onclick="cleanupStale()" style="font-size:.8rem;">&#128336; Remove Stale (&gt;24h)</button>
    </div>
    <div class="result-box" id="cache-result"></div>
  </details>

</div>

<!-- STEP 6: Benchmark Comparison -->
<div class="card" id="step6">
  <h2>Step 6 — Benchmark &amp; Compare</h2>
  <p style="color:#8b949e;">Benchmark the original and optimized models, then compare the results.</p>

  <!-- Compare Mode Selector -->
  <div style="display:flex;gap:.5rem;margin:1rem 0;">
    <button class="btn btn-primary" id="cmpmode-origopt" onclick="setBenchMode('origopt')" style="flex:1;">&#128260; Original vs Optimized</button>
  </div>
  <details style="margin-bottom:.5rem;">
    <summary style="color:#8b949e;cursor:pointer;font-size:.85rem;">&#128295; Advanced Benchmark Modes</summary>
    <div style="display:flex;gap:.5rem;margin:.5rem 0;">
      <button class="btn btn-outline" id="cmpmode-modmod" onclick="setBenchMode('modmod')" style="flex:1;">&#128257; Model vs Model</button>
      <button class="btn btn-outline" id="cmpmode-baseline" onclick="setBenchMode('baseline')" style="flex:1;">&#127919; Classical Baseline</button>
    </div>
  </details>

  <!-- Mode: Original vs Optimized -->
  <div id="bench-panel-origopt">
    <p style="color:#8b949e;font-size:.9rem;margin-bottom:.5rem;">Compare the <em>same</em> base model before and after quantization.</p>
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
  </div>

  <!-- Mode: Model vs Model -->
  <div id="bench-panel-modmod" style="display:none;">
    <p style="color:#8b949e;font-size:.9rem;margin-bottom:.5rem;">Compare any two local models side-by-side. They do <strong>not</strong> need to share a base model.</p>
    <div class="grid" style="margin:1rem 0;">
      <div class="form-group">
        <label>Left Model (A)</label>
        <input type="text" id="bench-left" placeholder="e.g. qwen2.5:7b" />
      </div>
      <div class="form-group">
        <label>Right Model (B)</label>
        <input type="text" id="bench-right" placeholder="e.g. mistral:7b" />
      </div>
      <div class="form-group">
        <label>Task Mode</label>
        <select id="bench-mode2">
          <option value="general">General</option>
          <option value="coding">Coding</option>
          <option value="cyber">Cyber Security</option>
        </select>
      </div>
    </div>
    <div style="display:flex;gap:.75rem;flex-wrap:wrap;margin:1rem 0;">
      <button class="btn btn-primary" onclick="runBenchmarkMM('left')">&#9654; Benchmark Left (A)</button>
      <button class="btn btn-green" onclick="runBenchmarkMM('right')">&#9654; Benchmark Right (B)</button>
      <button class="btn btn-outline" onclick="compareBenchmarksMM()">&#128200; Compare A vs B</button>
    </div>
  </div>

  <!-- Mode: Classical Baseline -->
  <div id="bench-panel-baseline" style="display:none;">
    <p style="color:#8b949e;font-size:.9rem;margin-bottom:.5rem;">Compare a dedicated classical / structured baseline against an LLM for a given task mode.</p>
    <div class="card" style="border-color:var(--purple);margin:1rem 0;">
      <h3 style="color:var(--purple);">Available Classical Baselines</h3>
      <table><thead><tr><th>Mode</th><th>Baseline</th><th>Status</th></tr></thead><tbody>
        <tr><td>Cyber / IDS</td><td>NSL-KDD Random Forest classifier</td><td style="color:var(--green);">Available (see IDS Quick Benchmark)</td></tr>
        <tr><td>Coding</td><td>No classical baseline</td><td style="color:#8b949e;">LLMs are the baseline for code generation</td></tr>
        <tr><td>General</td><td>No classical baseline</td><td style="color:#8b949e;">LLMs are the baseline for general tasks</td></tr>
      </tbody></table>
      <p style="color:#8b949e;margin-top:.75rem;font-size:.85rem;">For <strong>Cyber / IDS</strong>: run the IDS Quick Benchmark on the <a href="/bench">Bench page</a>, then benchmark an LLM in cyber mode and compare the results.
      For other modes, use "Model vs Model" to compare two LLMs directly.</p>
    </div>
  </div>

  <div class="result-box" id="bench-result"></div>
</div>

<!-- STEP 7: Final Results -->
<div class="card" id="step7" style="border-color:var(--green);">
  <h2 style="color:var(--green);">Step 7 — Final Recommendation</h2>
  <p style="color:#8b949e;">Based on your hardware, task mode, and benchmark results, here's the final analysis.</p>
  <div class="result-box" id="final-result"></div>
  <button class="btn btn-green" onclick="generateFinalReport()">&#128203; Generate Final Report</button>
  <a href="/chat" class="btn btn-outline" style="text-decoration:none;margin-left:.5rem;">Still unsure? Ask Chat &rarr;</a>
</div>
"""

    workflow_js = """
window._selectedMode = 'general';
window._benchCards = {};
window._optimizedModel = '';
window._selectedOllamaTag = '';
window._selectedHfRepo = '';

// ── Dataset info per mode ──
const _modeDatasets = {
    general: {
        type: 'Prompt Suite',
        label: 'Prompt-based benchmark (no external dataset)',
        items: [
            {name: 'JSON Summarization', desc: 'Asks the model to summarize a concept and reply in valid JSON. Tests structured output + knowledge.'},
            {name: 'Latency Explanation', desc: 'One-sentence explanation prompt. Tests conciseness and relevance.'}
        ],
        note: 'General mode uses built-in prompt suites, not dataset files. This is an LLM-native benchmark.'
    },
    coding: {
        type: 'Prompt Suite',
        label: 'Prompt-based benchmark (no external dataset)',
        items: [
            {name: 'Python Code Generation', desc: 'Asks for a function (add_numbers). Checks def/return keywords + correctness.'},
            {name: 'JSON Type Hints', desc: 'Structured output prompt about Python type hints. Tests JSON validity.'}
        ],
        note: 'Coding mode uses built-in prompt suites, not dataset files. There is no HumanEval or MBPP dataset wired yet.'
    },
    cyber: {
        type: 'Prompt Suite + Verifier',
        label: 'Prompt-based benchmark with artifact verification',
        items: [
            {name: 'Sigma Rule Generation', desc: 'Asks the model to write a Sigma detection rule in YAML. Output is verified by SigmaVerifier.'},
            {name: 'IOC JSON Extraction', desc: 'Structured threat intelligence extraction prompt. Tests JSON output validity.'}
        ],
        verifiers: ['SigmaVerifier — validates Sigma rule YAML syntax and required fields'],
        usedIn: ['Cyber rule generation verification', 'Cyber copilot quality assessment', 'Model recommendation weighting (cyber score)'],
        note: 'Cyber mode has real artifact verification (Sigma rules). IOC extraction is prompt-based.'
    },
    ids: {
        type: 'Real Dataset',
        label: 'NSL-KDD intrusion detection dataset',
        items: [
            {name: 'KDDTrain+.txt', desc: '125,973 labeled network connections (41 features). Train set for RF classifier.'},
            {name: 'KDDTest+.txt', desc: '22,544 labeled connections. Evaluation set.'},
            {name: 'KDDTest-21.txt', desc: 'Harder subset — 21 novel attack types not in training.'},
            {name: 'KDDTrain+_20Percent.txt', desc: '20% random sample for quick experiments.'}
        ],
        classicalBaseline: 'Random Forest classifier (sklearn) — trains on KDDTrain+.txt, evaluates on KDDTest+.txt',
        metrics: ['detection_rate', 'f1', 'roc_auc', 'pr_auc', 'false_positive_rate'],
        usedIn: ['IDS benchmark (classical baseline)', 'Model recommendation weighting (IDS detection score)'],
        note: 'IDS is the only mode backed by real datasets. The NSL-KDD files are in the workspace root.'
    }
};

// Override selectMode to also show dataset info
(function() {
    const _origSelectMode = window.selectMode || function(){};
    window.selectMode = function(el, mode) {
        _origSelectMode(el, mode);
        const info = _modeDatasets[mode];
        const box = document.getElementById('mode-dataset-info');
        if (!box || !info) return;
        const typeColor = info.type === 'Real Dataset' ? 'var(--green)' : (info.type.includes('Verifier') ? 'var(--accent)' : '#8b949e');
        let h = '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem;">';
        h += '<strong style="color:var(--accent);">Benchmark Data</strong>';
        h += '<span class="badge" style="background:' + typeColor + '22;color:' + typeColor + ';border:1px solid ' + typeColor + '55;">' + info.type + '</span>';
        h += '</div>';
        h += '<p style="color:#8b949e;font-size:.85rem;margin-bottom:.5rem;">' + info.label + '</p>';
        h += '<table><thead><tr><th>Source</th><th>Description</th></tr></thead><tbody>';
        info.items.forEach(function(it) {
            h += '<tr><td><strong>' + it.name + '</strong></td><td style="color:#8b949e;font-size:.85rem;">' + it.desc + '</td></tr>';
        });
        h += '</tbody></table>';
        if (info.verifiers) {
            h += '<div style="margin-top:.5rem;"><strong style="color:var(--accent);font-size:.85rem;">Verifiers:</strong> ';
            h += info.verifiers.map(function(v) { return '<span class="badge badge-ok" style="font-size:.75rem;">' + v + '</span>'; }).join(' ');
            h += '</div>';
        }
        if (info.classicalBaseline) {
            h += '<div style="margin-top:.5rem;"><strong style="color:var(--green);font-size:.85rem;">Classical Baseline:</strong> <span style="color:#8b949e;font-size:.85rem;">' + info.classicalBaseline + '</span></div>';
        }
        if (info.metrics) {
            h += '<div style="margin-top:.25rem;"><strong style="font-size:.85rem;color:var(--accent);">Metrics:</strong> ' + info.metrics.map(function(m) { return '<code style="font-size:.8rem;">' + m + '</code>'; }).join(', ') + '</div>';
        }
        if (info.usedIn) {
            h += '<div style="margin-top:.5rem;"><strong style="font-size:.85rem;color:var(--purple);">Used In:</strong><ul style="margin:.25rem 0 0 1.25rem;font-size:.85rem;color:#8b949e;">';
            info.usedIn.forEach(function(u) { h += '<li>' + u + '</li>'; });
            h += '</ul></div>';
        }
        h += '<p style="color:#8b949e;font-size:.8rem;font-style:italic;margin-top:.5rem;">' + info.note + '</p>';
        box.innerHTML = h;
        box.style.display = '';
        // Show continue buttons after mode selection
        var cont = document.getElementById('mode-continue');
        if (cont) cont.style.display = '';
    };
})();



function setStep(n, noScroll) {
    for (let i = 1; i <= 7; i++) {
        const sp = document.getElementById('sp' + i);
        if (i < n) sp.className = 'step-pill done';
        else if (i === n) sp.className = 'step-pill active';
        else sp.className = 'step-pill';
    }
    if (!noScroll) {
        const el = document.getElementById('step' + n);
        if (el) setTimeout(function(){ el.scrollIntoView({behavior:'smooth', block:'start'}); }, 80);
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
    // Persist to sessionStorage so chat, bench, optimize pages pick it up
    CyberForge.setSelected({
        id: el.dataset.ollama || el.dataset.hf || el.dataset.name || '',
        display_name: el.dataset.name || '',
        ollama_tag: el.dataset.ollama || '',
        hf_repo: el.dataset.hf || '',
        mode: window._selectedMode || ''
    });
    _syncModelField();
    // Fill benchmark fields
    document.getElementById('bench-orig').value = window._selectedOllamaTag || window._selectedHfRepo || '';
    // Show action bar
    const bar = document.getElementById('rec-actions');
    if (bar) {
        bar.style.display = '';
        document.getElementById('rec-actions-name').textContent = el.dataset.name || el.dataset.ollama || el.dataset.hf || '';
    }
}
function recUseInWorkflow() {
    setStep(5);
}
async function recStartOptimize() {
    if (!window._selectedOllamaTag && !window._selectedHfRepo) return;
    setStep(5);
    // Auto-start quantization after a brief visual pause
    setTimeout(function() { runQuantize(); }, 400);
}
function recOpenChat() {
    // Navigate to chat — sessionStorage already has the model
    window.location.href = '/chat';
}
function recOpenBench() {
    setStep(6);
}

function _syncModelField() {
    // Fill od-model based on current backend selection
    const backend = document.getElementById('od-backend').value;
    const modelEl = document.getElementById('od-model');
    const hintEl = document.getElementById('od-model-hint');
    if (backend === 'ollama') {
        // Rule 5: NEVER send HF repo IDs to Ollama — only use ollama_tag
        modelEl.value = window._selectedOllamaTag || '';
        if (hintEl) {
            if (window._selectedOllamaTag) {
                hintEl.textContent = 'Ollama tag: ' + window._selectedOllamaTag;
            } else if (window._selectedHfRepo) {
                // Derive a search term from HF repo name
                const parts = window._selectedHfRepo.split('/');
                const raw = (parts[parts.length - 1] || '').toLowerCase()
                    .replace(/[-_](instruct|chat|it|base|hf|gguf)$/g, '')
                    .replace(/[-_](\d+\.?\d*[bm])$/g, '')
                    .replace(/[-_](4k|8k|32k|128k|1m)$/g, '')
                    .replace(/[-_.]+$/, '');
                hintEl.innerHTML = '<span style="color:#f0883e;">\u26A0 No Ollama tag for this model.</span> '
                    + '<br><input id="od-pull-name" style="margin-top:.3rem;padding:.25rem .5rem;border:1px solid var(--border);border-radius:4px;background:var(--bg);color:var(--fg);font-size:.8rem;width:180px;" placeholder="e.g. qwen2.5:3b" value="' + raw + '" /> '
                    + '<button class="btn btn-sm btn-green" onclick="searchAndPullForWorkflow()" style="font-size:.75rem;padding:.2rem .6rem;vertical-align:middle;">\uD83D\uDD0D Search &amp; Pull</button>'
                    + '<span id="od-search-result" style="display:block;margin-top:.3rem;font-size:.8rem;"></span>';
            } else {
                hintEl.textContent = '';
            }
        }
    } else {
        modelEl.value = window._selectedHfRepo || window._selectedOllamaTag || '';
        if (hintEl) hintEl.textContent = window._selectedHfRepo ? 'HF repo: ' + window._selectedHfRepo : (window._selectedOllamaTag ? 'No HF repo mapped — check registry.yaml' : '');
    }
}

async function searchAndPullForWorkflow() {
    const nameInput = document.getElementById('od-pull-name');
    const resultEl = document.getElementById('od-search-result');
    const q = (nameInput ? nameInput.value.trim() : '').replace(/[^a-z0-9._:-]/gi, '');
    if (!q) { resultEl.innerHTML = '<span style="color:#f85149">Enter a model name to search.</span>'; return; }
    resultEl.innerHTML = '<span class="spinner"></span> Searching Ollama library\u2026';
    try {
        // First check if Ollama is running, auto-start if not
        const status = await cyberGet('/action/ollama-status');
        if (!status.available) {
            resultEl.innerHTML = '<span class="spinner"></span> Ollama not running \u2014 starting\u2026';
            const startRes = await cyberPost('/action/ollama-start');
            if (!startRes.started) {
                resultEl.innerHTML = '<span style="color:#f85149">Could not start Ollama: ' + (startRes.error || 'unknown') + '. Run <code>ollama serve</code> manually.</span>';
                return;
            }
            await new Promise(r => setTimeout(r, 2000));
        }
        // Search the Ollama library for matching models
        const search = await cyberGet('/action/ollama-library-search?q=' + encodeURIComponent(q));
        const results = search.results || [];
        if (results.length === 0) {
            // No search results — try direct pull with the entered name
            resultEl.innerHTML = '<span class="spinner"></span> No library matches \u2014 trying direct pull of <strong>' + q + '</strong>\u2026';
            const pullRes = await cyberPost('/action/ollama-pull', {model_name: q});
            if (pullRes._error || pullRes.error) {
                resultEl.innerHTML = '<span style="color:#f85149">Pull failed: ' + (pullRes._error || pullRes.error) + '</span>';
            } else {
                window._selectedOllamaTag = q;
                document.getElementById('od-model').value = q;
                resultEl.innerHTML = '<span style="color:var(--green)">\u2713 Pulled <strong>' + q + '</strong>! Ready to quantize.</span>';
            }
            return;
        }
        // Show search results as clickable options
        let html = '<strong>Matches:</strong> ';
        results.forEach(function(name) {
            html += '<button class="btn btn-sm btn-outline" style="font-size:.75rem;padding:.15rem .5rem;margin:.15rem;border-color:var(--accent);color:var(--accent);" '
                + 'onclick="pullOllamaLibraryModel(\'' + name.replace(/'/g, "\\'") + '\')">' + name + '</button>';
        });
        resultEl.innerHTML = html;
    } catch(e) { resultEl.innerHTML = '<span style="color:#f85149">Error: ' + e.message + '</span>'; }
}

async function pullOllamaLibraryModel(name) {
    const resultEl = document.getElementById('od-search-result');
    resultEl.innerHTML = '<span class="spinner"></span> Pulling <strong>' + name + '</strong>\u2026 this may take a few minutes.';
    try {
        const res = await cyberPost('/action/ollama-pull', {model_name: name});
        if (res._error || res.error) {
            resultEl.innerHTML = '<span style="color:#f85149">Pull failed: ' + (res._error || res.error) + '</span>';
        } else {
            window._selectedOllamaTag = name;
            document.getElementById('od-model').value = name;
            resultEl.innerHTML = '<span style="color:var(--green)">\u2713 Pulled <strong>' + name + '</strong>! Ready to quantize.</span>';
            // Update stored selection
            const sel = CyberForge.getSelected();
            if (sel) { sel.ollama_tag = name; CyberForge.setSelected(sel); }
        }
    } catch(e) { resultEl.innerHTML = '<span style="color:#f85149">Error: ' + e.message + '</span>'; }
}

function applyPreset(el, level) {
    // Highlight selected preset card
    document.querySelectorAll('#step5 .mode-card').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
    // Set backend to Ollama and apply the GGUF level
    document.getElementById('od-backend').value = 'ollama';
    const levelMap = {light: 'q8_0', balanced: 'q4_k_m', maximum: 'q3_k_m'};
    document.getElementById('od-level').value = levelMap[level] || 'q4_k_m';
    onBackendChange();
}

function onBackendChange() {
    const ggufGroup = document.getElementById('gguf-level-group');
    if (ggufGroup) ggufGroup.style.display = '';
    _syncModelField();
}
// Initialize backend visibility on page load
document.addEventListener('DOMContentLoaded', function() { onBackendChange(); });

// ── Load env status for Step 5 on page load ──
(async function() {
    try {
        // Auto-start Ollama if not running
        const ollamaCheck = await cyberGet('/action/ollama-status');
        if (!ollamaCheck.available) {
            const el = document.getElementById('od-env-status');
            if (el) el.innerHTML = '<span class="spinner"></span> Ollama not running \u2014 auto-starting\u2026';
            const startRes = await cyberPost('/action/ollama-start');
            if (!startRes.started && !startRes.warning) {
                if (el) el.innerHTML = '<span style="color:#d29922;">\u26A0 Could not auto-start Ollama: ' + (startRes.error || 'unknown') + '</span> '
                    + '<button class="btn btn-sm btn-green" onclick="retryOllamaStart(this)" style="margin-left:.5rem;font-size:.75rem;">Retry</button>';
            }
        }
        const _timeout = new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 15000));
        const status = await Promise.race([cyberGet('/action/quantize-status'), _timeout]);
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
    } catch(e) {
        const el = document.getElementById('od-env-status');
        if(el) el.innerHTML = e.message === 'timeout'
            ? '<span style="color:#d29922">Backend check timed out &mdash; services may still be starting. <a href="javascript:location.reload()" style="color:var(--accent)">Retry</a></span>'
            : '<span style="color:#8b949e">Could not check backends</span>';
    }
})();

function retryOllamaStart(el) {
    el.parentElement.innerHTML = '<span class="spinner"></span> Retrying\u2026';
    fetch('/action/ollama-start', {method: 'POST'}).then(() => location.reload());
}

async function runSelfTest(type) {
    setStep(3, true);
    showLoading('test-result');
    try {
        const data = await cyberPost('/action/self-test/' + type);
        if (data._error) { showResult('test-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        let html = '<h3 style="color:var(--green)">&#10003; ' + type.toUpperCase() + ' Self-Test Complete</h3>';
        html += '<pre style="color:var(--fg);white-space:pre-wrap;font-size:.85rem;">' + JSON.stringify(data, null, 2) + '</pre>';
        html += '<div style="margin-top:1rem;"><button class="btn btn-green" onclick="setStep(4)">Continue &rarr; Get Model Recommendations</button></div>';
        showResult('test-result', html);
    } catch (e) { showResult('test-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function getRecommendations() {
    setStep(4, true);
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
        // Action bar (shown once a model is selected)
        html += '<div id="rec-actions" class="card" style="display:none;margin-top:1rem;border-color:var(--accent);background:#1a1025;">';
        html += '<div style="display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;">';
        html += '<strong style="color:var(--accent);">Selected: <span id="rec-actions-name"></span></strong>';
        html += '<button class="btn btn-purple" onclick="recUseInWorkflow()" title="Continue with this model in the optimization workflow">&#9654; Use in Workflow</button>';
        html += '<button class="btn btn-green" onclick="recStartOptimize()" title="Jump to Step 6 and start quantization immediately">&#128640; Start Optimization</button>';
        html += '<button class="btn btn-outline" onclick="recOpenChat()" style="border-color:var(--accent);color:var(--accent);" title="Open chat with this model">&#128172; Chat</button>';
        html += '<button class="btn btn-outline" onclick="recOpenBench()" style="border-color:var(--accent);color:var(--accent);" title="Jump to benchmark step">&#128202; Benchmark</button>';
        html += '</div></div>';
        // Auto-fill overdrive model — prefer a model with an ollama_tag (actually runnable)
        if (picks.length > 0) {
            const m0 = picks[0].model || picks[0];
            window._selectedOllamaTag = m0.ollama_tag || '';
            window._selectedHfRepo = m0.hf_repo || '';
            // Persist first pick to sessionStorage for cross-page handoff
            CyberForge.setSelected({
                id: m0.ollama_tag || m0.hf_repo || m0.id || '',
                display_name: m0.display_name || m0.model_id || m0.id || '',
                ollama_tag: m0.ollama_tag || '',
                hf_repo: m0.hf_repo || '',
                mode: window._selectedMode || ''
            });
            _syncModelField();
            document.getElementById('bench-orig').value = window._selectedOllamaTag || window._selectedHfRepo || '';
            // Show action bar for first pick
            setTimeout(function() {
                const bar = document.getElementById('rec-actions');
                if (bar) { bar.style.display = ''; document.getElementById('rec-actions-name').textContent = m0.display_name || m0.ollama_tag || m0.hf_repo || ''; }
            }, 100);
        }
        showResult('rec-result', html);
    } catch (e) { showResult('rec-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function cleanupCache() {
    setStep(5, true);
    showLoading('cache-result');
    try {
        const data = await cyberPost('/action/cache-cleanup');
        showResult('cache-result', '<span style="color:var(--green)">&#10003; Cleaned up ' + (data.deleted_count || 0) + ' volatile artifact(s).</span>');
    } catch (e) { showResult('cache-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function cleanupStale() {
    setStep(5, true);
    showLoading('cache-result');
    try {
        const data = await cyberPost('/action/cache-stale');
        showResult('cache-result', '<span style="color:var(--green)">&#10003; Removed ' + (data.deleted_count || 0) + ' stale cached artifact(s).</span>');
    } catch (e) { showResult('cache-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function runQuantize() {
    setStep(5, true);
    showLoading('quant-result');
    const model = document.getElementById('od-model').value;
    const backend = document.getElementById('od-backend').value;
    const level = document.getElementById('od-level').value;
    if (!model) { showResult('quant-result', '<span style="color:#f85149">Enter a source model name.</span>'); return; }

    // For Ollama backend, ensure Ollama is running, then check/pull model
    if (backend === 'ollama') {
        showResult('quant-result', '<span class="spinner"></span> Checking Ollama status...');
        try {
            const ollamaStatus = await cyberGet('/action/ollama-status');
            if (!ollamaStatus.available) {
                showResult('quant-result', '<span class="spinner"></span> Ollama not running \u2014 auto-starting...');
                const startRes = await cyberPost('/action/ollama-start');
                if (!startRes.started && !startRes.warning) {
                    showResult('quant-result', '<span style="color:#f85149">Could not start Ollama: ' + (startRes.error || 'unknown') + '. Run <code>ollama serve</code> manually.</span>');
                    return;
                }
                await new Promise(r => setTimeout(r, 2000));
            }
        } catch(e) { /* continue \u2014 pull attempt will surface any real issue */ }
        showResult('quant-result', '<span class="spinner"></span> Checking if model is available locally...');
        try {
            const models = await cyberGet('/action/ollama-models');
            const list = Array.isArray(models) ? models : [];
            const found = list.some(m => m.name === model || m.name.startsWith(model + ':'));
            if (!found) {
                showResult('quant-result', '<span class="spinner"></span> Model not found locally \u2014 pulling <strong>' + model + '</strong> from Ollama (this may take a few minutes)...');
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
        if (data.output_model) {
            document.getElementById('bench-opt').value = data.output_model;
            window._optimizedModel = data.output_model;
        }
        html += '<div style="margin-top:1rem;display:flex;gap:.75rem;flex-wrap:wrap;align-items:center;">';
        html += '<button class="btn btn-primary" onclick="setStep(6)">Continue &rarr; Benchmark &amp; Compare</button>';
        html += '<span style="color:#8b949e;font-size:.85rem;">Benchmark before and after to see the impact.</span>';
        html += '</div>';
        showResult('quant-result', html);
    } catch (e) { showResult('quant-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function runQuantCompare() {
    setStep(5, true);
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

// ── Compare mode switching ──
function setBenchMode(mode) {
    ['origopt','modmod','baseline'].forEach(m => {
        document.getElementById('bench-panel-' + m).style.display = m === mode ? '' : 'none';
        const btn = document.getElementById('cmpmode-' + m);
        btn.className = m === mode ? 'btn btn-primary' : 'btn btn-outline';
    });
}

// ── Shared SVG bar chart builder ──
function svgBarChart(metrics, leftLabel, rightLabel, compareMode) {
    if (!metrics || metrics.length === 0) return '<p style="color:#8b949e;font-style:italic;">No chart data — run benchmarks for both models first.</p>';
    const valid = metrics.filter(m => m.left != null && m.right != null);
    if (valid.length === 0) return '<p style="color:#8b949e;font-style:italic;">No chart data — all metrics are null. This happens when a metric is not measured for this task mode.</p>';
    const W = 600, barH = 22, gap = 50, padL = 120, padR = 80;
    const H = valid.length * gap + 30;
    let svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;max-width:600px;font-family:sans-serif;font-size:12px;">';
    valid.forEach(function(m, i) {
        const y = i * gap + 10;
        const maxVal = Math.max(m.left, m.right, 0.001);
        const bW = Math.max((m.left / maxVal) * (W - padL - padR), 2);
        const oW = Math.max((m.right / maxVal) * (W - padL - padR), 2);
        const diff = m.right - m.left;
        const pct = m.left !== 0 ? ((diff / m.left) * 100).toFixed(1) : '0.0';
        const improved = m.lowerIsBetter ? diff < 0 : diff > 0;
        const pctColor = improved ? '#3fb950' : (diff === 0 ? '#8b949e' : '#f85149');
        svg += '<text x="' + (padL - 5) + '" y="' + (y + 10) + '" text-anchor="end" fill="#c9d1d9" font-weight="600">' + m.name + '</text>';
        svg += '<rect x="' + padL + '" y="' + y + '" width="' + bW + '" height="' + barH + '" rx="3" fill="#58a6ff" opacity="0.85"/>';
        svg += '<text x="' + (padL + bW + 4) + '" y="' + (y + 15) + '" fill="#58a6ff" font-size="11">' + m.left.toFixed(1) + m.unit + '</text>';
        svg += '<rect x="' + padL + '" y="' + (y + barH + 2) + '" width="' + oW + '" height="' + barH + '" rx="3" fill="#3fb950" opacity="0.85"/>';
        svg += '<text x="' + (padL + oW + 4) + '" y="' + (y + barH + 17) + '" fill="#3fb950" font-size="11">' + m.right.toFixed(1) + m.unit + '</text>';
        const arrow = improved ? '\\u25B2' : (diff === 0 ? '\\u2500' : '\\u25BC');
        svg += '<text x="' + (W - 5) + '" y="' + (y + barH + 5) + '" text-anchor="end" fill="' + pctColor + '" font-weight="600" font-size="11">' + arrow + ' ' + Math.abs(pct) + '%</text>';
    });
    svg += '</svg>';
    svg += '<div style="display:flex;gap:1.5rem;margin-top:.5rem;font-size:.8rem;color:#8b949e;">';
    svg += '<span><span style="display:inline-block;width:12px;height:12px;background:#58a6ff;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>' + leftLabel + '</span>';
    svg += '<span><span style="display:inline-block;width:12px;height:12px;background:#3fb950;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>' + rightLabel + '</span>';
    svg += '</div>';
    return svg;
}

// ── Size reduction banner ──
function sizeReductionBanner(d) {
    if (!d || !d.model_size_mb || d.model_size_mb.baseline == null || d.model_size_mb.optimized == null) return '';
    const orig = d.model_size_mb.baseline;
    const opt = d.model_size_mb.optimized;
    if (orig === 0 && opt === 0) return '<p style="color:#8b949e;font-style:italic;margin:.5rem 0;">Model size: not reported by backend (non-Ollama or model not found).</p>';
    const redPct = orig > 0 ? ((orig - opt) / orig * 100).toFixed(1) : 0;
    const reduced = opt < orig;
    const color = reduced ? 'var(--green)' : (opt === orig ? '#8b949e' : '#f85149');
    let h = '<div class="card" style="border-color:' + color + ';margin:1rem 0;text-align:center;">';
    h += '<div style="font-size:2rem;font-weight:700;color:' + color + ';">';
    if (reduced) h += '&#9660; ' + Math.abs(redPct) + '% smaller';
    else if (opt === orig) h += '&#9644; Same size';
    else h += '&#9650; ' + Math.abs(redPct) + '% larger';
    h += '</div>';
    h += '<div style="color:#8b949e;margin-top:.25rem;">' + orig.toFixed(0) + ' MB &rarr; ' + opt.toFixed(0) + ' MB</div>';
    h += '</div>';
    return h;
}

// ── Render compare result (shared between both modes) ──
function renderCompareResult(data, targetId) {
    const cmp = data.comparison || data;
    const d = cmp.deltas || {};
    const mode = cmp.compare_mode || 'model_vs_model';
    const bInfo = cmp.baseline || {};
    const oInfo = cmp.optimized || {};
    const bLabel = bInfo.model_id || 'Model A';
    const oLabel = oInfo.model_id || 'Model B';

    let html = '';

    // ── Header with mode badge ──
    if (mode === 'original_vs_optimized') {
        html += '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:1rem;">';
        html += '<span style="background:var(--green);color:#0d1117;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600;">Original vs Optimized</span>';
        html += '</div>';
        html += '<h3 style="color:var(--green)">&#128260; Optimization Impact</h3>';
        html += '<table style="margin-bottom:1rem;"><tbody>';
        html += '<tr><td style="color:#8b949e;">Base Model</td><td><strong>' + bLabel + '</strong></td></tr>';
        html += '<tr><td style="color:#8b949e;">Optimized Artifact</td><td><strong style="color:var(--green);">' + oLabel + '</strong></td></tr>';
        html += '<tr><td style="color:#8b949e;">Original Label</td><td>' + (bInfo.label || 'baseline') + '</td></tr>';
        html += '<tr><td style="color:#8b949e;">Optimized Label</td><td>' + (oInfo.label || '—') + '</td></tr>';
        html += '</tbody></table>';
        html += sizeReductionBanner(d);
    } else {
        html += '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:1rem;">';
        html += '<span style="background:#58a6ff;color:#0d1117;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600;">Model vs Model</span>';
        html += '</div>';
        html += '<h3 style="color:#58a6ff;">&#128257; Side-by-Side Comparison</h3>';
        html += '<p style="color:#8b949e;margin-bottom:1rem;"><strong style="color:#58a6ff;">' + bLabel + '</strong> vs <strong style="color:#3fb950;">' + oLabel + '</strong></p>';
    }

    // ── Performance chart ──
    const perfMetrics = [];
    if (d.latency_ms && d.latency_ms.baseline != null) perfMetrics.push({name:'Latency', left:d.latency_ms.baseline, right:d.latency_ms.optimized, unit:' ms', lowerIsBetter:true});
    if (d.throughput_tok_s && d.throughput_tok_s.baseline != null) perfMetrics.push({name:'Throughput', left:d.throughput_tok_s.baseline, right:d.throughput_tok_s.optimized, unit:' tok/s', lowerIsBetter:false});
    if (d.load_time_ms && d.load_time_ms.baseline != null) perfMetrics.push({name:'Load Time', left:d.load_time_ms.baseline, right:d.load_time_ms.optimized, unit:' ms', lowerIsBetter:true});
    if (d.model_size_mb && d.model_size_mb.baseline != null) perfMetrics.push({name:'Size', left:d.model_size_mb.baseline, right:d.model_size_mb.optimized, unit:' MB', lowerIsBetter:true});
    html += '<div class="card" style="border-color:var(--green);margin-top:1rem;">';
    html += '<h3>&#9889; Performance &amp; Size</h3>';
    html += svgBarChart(perfMetrics, bLabel, oLabel, mode);
    html += '</div>';

    // ── Resource chart ──
    const resMetrics = [];
    if (d.vram_peak_mb && d.vram_peak_mb.baseline != null) resMetrics.push({name:'VRAM Peak', left:d.vram_peak_mb.baseline, right:d.vram_peak_mb.optimized, unit:' MB', lowerIsBetter:true});
    if (d.ram_peak_mb && d.ram_peak_mb.baseline != null) resMetrics.push({name:'RAM Peak', left:d.ram_peak_mb.baseline, right:d.ram_peak_mb.optimized, unit:' MB', lowerIsBetter:true});
    html += '<div class="card" style="border-color:var(--purple);margin-top:1rem;">';
    html += '<h3>&#128190; Resource Usage</h3>';
    html += svgBarChart(resMetrics, bLabel, oLabel, mode);
    html += '</div>';

    // ── Quality chart ──
    const qualMetrics = [];
    if (d.exact_match && d.exact_match.baseline != null) qualMetrics.push({name:'Exact Match', left:d.exact_match.baseline, right:d.exact_match.optimized, unit:'', lowerIsBetter:false});
    if (d.verifier_pass_rate && d.verifier_pass_rate.baseline != null) qualMetrics.push({name:'Verifier Pass', left:d.verifier_pass_rate.baseline, right:d.verifier_pass_rate.optimized, unit:'', lowerIsBetter:false});
    if (d.structured_output_validity && d.structured_output_validity.baseline != null) qualMetrics.push({name:'Structured Output', left:d.structured_output_validity.baseline, right:d.structured_output_validity.optimized, unit:'', lowerIsBetter:false});
    if (d.syntax_error_rate && d.syntax_error_rate.baseline != null) qualMetrics.push({name:'Syntax Error Rate', left:d.syntax_error_rate.baseline, right:d.syntax_error_rate.optimized, unit:'', lowerIsBetter:true});
    html += '<div class="card" style="border-color:#58a6ff;margin-top:1rem;">';
    html += '<h3>&#127919; Quality Metrics</h3>';
    html += svgBarChart(qualMetrics, bLabel, oLabel, mode);
    html += '</div>';

    // ── Summary table (all deltas) ──
    html += '<div class="card" style="margin-top:1rem;">';
    html += '<h3>&#128202; Full Metric Table</h3>';
    const rows = [];
    Object.keys(d).forEach(function(key) {
        const m = d[key];
        if (!m || m.baseline == null || m.optimized == null) return;
        const imp = m.improved;
        const color = imp === true ? 'var(--green)' : (imp === false ? '#f85149' : '#8b949e');
        const arrow = imp === true ? '&#9650;' : (imp === false ? '&#9660;' : '&#9644;');
        rows.push([key.replace(/_/g,' '), m.baseline.toFixed(2), m.optimized.toFixed(2), '<span style="color:' + color + ';font-weight:600;">' + arrow + ' ' + (m.percent != null ? Math.abs(m.percent).toFixed(1) + '%' : '—') + '</span>']);
    });
    if (rows.length) html += buildTable([mode === 'original_vs_optimized' ? 'Metric' : 'Metric', bLabel, oLabel, 'Change'], rows);
    else html += '<p style="color:#8b949e;font-style:italic;">No comparable metrics available.</p>';
    html += '</div>';

    if (data.report_path) html += '<p style="color:#8b949e;margin-top:.5rem;">Report saved: ' + data.report_path + '</p>';
    return html;
}

// ── Original vs Optimized ──
async function runBenchmark(label) {
    setStep(6, true);
    showLoading('bench-result');
    const modelId = label === 'original' ? document.getElementById('bench-orig').value : document.getElementById('bench-opt').value;
    const mode = document.getElementById('bench-mode').value;
    if (!modelId) { showResult('bench-result', '<span style="color:#f85149">Enter a model ID.</span>'); return; }
    try {
        const data = await cyberPost('/action/benchmark', {model_id: modelId, task_mode: mode, label: label === 'original' ? 'baseline' : label, backend: 'ollama'});
        if (data._error || data.error) { showResult('bench-result', '<span style="color:#f85149">' + (data._error || data.error) + '</span>'); return; }
        window._benchCards[label] = data;
        const _bothDone = window._benchCards['original'] && window._benchCards['optimized'];
        let html = '<h3>' + label.charAt(0).toUpperCase() + label.slice(1) + ' Benchmark — ' + (data.model_id || modelId) + '</h3>';
        const sys = data.system || {};
        const task = data.task || {};
        html += buildTable(['Metric', 'Value'], [
            ['Latency', (sys.latency_ms||0).toFixed(1) + ' ms'],
            ['Throughput', (sys.throughput_tok_s||0).toFixed(1) + ' tok/s'],
            ['VRAM Peak', (sys.vram_peak_mb||0).toFixed(0) + ' MB'],
            ['Load Time', (sys.load_time_ms||0).toFixed(0) + ' ms'],
            ['Model Size', (sys.model_size_mb||0).toFixed(0) + ' MB'],
            ['Exact Match', task.exact_match != null ? task.exact_match.toFixed(3) : 'N/A'],
            ['F1', task.f1 != null ? task.f1.toFixed(3) : 'N/A'],
        ]);
        if (_bothDone) {
            html += '<div style="margin-top:1rem;padding:.75rem;border:1px solid var(--green);border-radius:8px;background:#0d1f0d;">';
            html += '<strong style="color:var(--green);">&#10003; Both benchmarks complete!</strong> ';
            html += '<button class="btn btn-green" onclick="compareBenchmarks()" style="margin-left:.75rem;">Compare Results Now</button>';
            html += '</div>';
        } else {
            const other = label === 'original' ? 'Optimized' : 'Original';
            html += '<div style="margin-top:.75rem;color:#8b949e;">Now benchmark the <strong>' + other + '</strong> model above to compare.</div>';
        }
        showResult('bench-result', html);
    } catch (e) { showResult('bench-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function compareBenchmarks() {
    setStep(6, true);
    const cards = await cyberGet('/action/bench-cards');
    if (!Array.isArray(cards) || cards.length < 2) {
        showResult('bench-result', '<span style="color:#f85149">Need at least 2 benchmark cards. Run both "Benchmark Original" and "Benchmark Optimized" above first.</span>');
        return;
    }
    showLoading('bench-result');
    try {
        const data = await cyberPost('/action/compare', {baseline_file: cards[cards.length-2], optimized_file: cards[cards.length-1], save_report: true});
        if (data._error) { showResult('bench-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        let cmpHtml = renderCompareResult(data, 'bench-result');
        cmpHtml += '<div style="margin-top:1rem;"><button class="btn btn-green" onclick="setStep(7)">Continue &rarr; Final Report</button></div>';
        showResult('bench-result', cmpHtml);
    } catch (e) { showResult('bench-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

// ── Model vs Model ──
async function runBenchmarkMM(side) {
    setStep(6, true);
    showLoading('bench-result');
    const modelId = side === 'left' ? document.getElementById('bench-left').value : document.getElementById('bench-right').value;
    const mode = document.getElementById('bench-mode2').value;
    if (!modelId) { showResult('bench-result', '<span style="color:#f85149">Enter a model ID for ' + (side === 'left' ? 'Left (A)' : 'Right (B)') + '.</span>'); return; }
    try {
        const data = await cyberPost('/action/benchmark', {model_id: modelId, task_mode: mode, label: side === 'left' ? 'model_a' : 'model_b', backend: 'ollama'});
        if (data._error || data.error) { showResult('bench-result', '<span style="color:#f85149">' + (data._error || data.error) + '</span>'); return; }
        window._benchCards[side] = data;
        const _bothDoneMM = window._benchCards['left'] && window._benchCards['right'];
        let html = '<h3>Model ' + (side === 'left' ? 'A' : 'B') + ' — ' + (data.model_id || modelId) + '</h3>';
        const sys = data.system || {};
        const task = data.task || {};
        html += buildTable(['Metric', 'Value'], [
            ['Latency', (sys.latency_ms||0).toFixed(1) + ' ms'],
            ['Throughput', (sys.throughput_tok_s||0).toFixed(1) + ' tok/s'],
            ['VRAM Peak', (sys.vram_peak_mb||0).toFixed(0) + ' MB'],
            ['Model Size', (sys.model_size_mb||0).toFixed(0) + ' MB'],
            ['Exact Match', task.exact_match != null ? task.exact_match.toFixed(3) : 'N/A'],
        ]);
        if (_bothDoneMM) {
            html += '<div style="margin-top:1rem;padding:.75rem;border:1px solid var(--green);border-radius:8px;background:#0d1f0d;">';
            html += '<strong style="color:var(--green);">&#10003; Both benchmarks complete!</strong> ';
            html += '<button class="btn btn-green" onclick="compareBenchmarksMM()" style="margin-left:.75rem;">Compare A vs B Now</button>';
            html += '</div>';
        } else {
            const other = side === 'left' ? 'Right (B)' : 'Left (A)';
            html += '<div style="margin-top:.75rem;color:#8b949e;">Now benchmark <strong>' + other + '</strong> above to compare.</div>';
        }
        showResult('bench-result', html);
    } catch (e) { showResult('bench-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function keepOptimizedModel() {
    const btn = document.getElementById('btn-keep');
    const btnD = document.getElementById('btn-discard');
    btn.disabled = true;
    btn.textContent = '\\u2713 Kept — model stays in Ollama';
    btn.style.opacity = '0.7';
    if (btnD) { btnD.disabled = true; btnD.style.opacity = '0.4'; }
}

async function discardOptimizedModel() {
    const model = window._optimizedModel || '';
    if (!model) return;
    if (!confirm('Delete optimized model "' + model + '" from Ollama? This cannot be undone.')) return;
    const btn = document.getElementById('btn-discard');
    const btnK = document.getElementById('btn-keep');
    btn.disabled = true;
    btn.textContent = 'Deleting...';
    try {
        const data = await cyberPost('/action/ollama-delete', {model_name: model});
        if (data.deleted) {
            btn.textContent = '\\u2713 Deleted';
            btn.style.borderColor = 'var(--green)';
            btn.style.color = 'var(--green)';
            if (btnK) { btnK.disabled = true; btnK.style.opacity = '0.4'; }
            window._optimizedModel = '';
        } else {
            btn.textContent = 'Delete failed: ' + (data.error || 'unknown');
            btn.style.color = '#f85149';
            btn.disabled = false;
        }
    } catch (e) {
        btn.textContent = 'Error: ' + e.message;
        btn.disabled = false;
    }
}

async function compareBenchmarksMM() {
    setStep(6, true);
    const cards = await cyberGet('/action/bench-cards');
    if (!Array.isArray(cards) || cards.length < 2) {
        showResult('bench-result', '<span style="color:#f85149">Need at least 2 benchmark cards. Run both Left (A) and Right (B) benchmarks first.</span>');
        return;
    }
    showLoading('bench-result');
    try {
        const data = await cyberPost('/action/compare', {baseline_file: cards[cards.length-2], optimized_file: cards[cards.length-1], save_report: true});
        if (data._error) { showResult('bench-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        let cmpHtmlMM = renderCompareResult(data, 'bench-result');
        cmpHtmlMM += '<div style="margin-top:1rem;"><button class="btn btn-green" onclick="setStep(7)">Continue &rarr; Final Report</button></div>';
        showResult('bench-result', cmpHtmlMM);
    } catch (e) { showResult('bench-result', '<span style="color:#f85149">Error: ' + e.message + '</span>'); }
}

async function generateFinalReport() {
    setStep(7, true);
    let html = '<h3 style="color:var(--green)">&#127942; Optimization Summary</h3>';
    html += '<p><strong>Task Mode:</strong> ' + window._selectedMode + '</p>';
    const orig = window._benchCards['original'];
    const opt = window._benchCards['optimized'];
    if (orig && opt) {
        const origSys = orig.system || {};
        const optSys = opt.system || {};
        // Size reduction banner
        if (origSys.model_size_mb && optSys.model_size_mb) {
            const szOrig = origSys.model_size_mb;
            const szOpt = optSys.model_size_mb;
            if (szOrig > 0) {
                const redPct = ((szOrig - szOpt) / szOrig * 100).toFixed(1);
                const col = szOpt < szOrig ? 'var(--green)' : '#f85149';
                html += '<div class="card" style="border-color:' + col + ';text-align:center;margin:1rem 0;">';
                html += '<div style="font-size:1.8rem;font-weight:700;color:' + col + ';">' + (szOpt < szOrig ? '&#9660; ' + Math.abs(redPct) + '% smaller' : '&#9650; ' + Math.abs(redPct) + '% larger') + '</div>';
                html += '<div style="color:#8b949e;">' + szOrig.toFixed(0) + ' MB &rarr; ' + szOpt.toFixed(0) + ' MB</div></div>';
            }
        }
        const perfMetrics = [
            {name:'Latency', left:origSys.latency_ms||0, right:optSys.latency_ms||0, unit:' ms', lowerIsBetter:true},
            {name:'Throughput', left:origSys.throughput_tok_s||0, right:optSys.throughput_tok_s||0, unit:' tok/s', lowerIsBetter:false},
            {name:'VRAM Peak', left:origSys.vram_peak_mb||0, right:optSys.vram_peak_mb||0, unit:' MB', lowerIsBetter:true},
            {name:'Model Size', left:origSys.model_size_mb||0, right:optSys.model_size_mb||0, unit:' MB', lowerIsBetter:true},
            {name:'Load Time', left:origSys.load_time_ms||0, right:optSys.load_time_ms||0, unit:' ms', lowerIsBetter:true},
        ];
        html += '<div class="card" style="border-color:var(--green);">';
        html += '<h3>&#128200; Performance Comparison</h3>';
        html += '<p style="color:#8b949e;margin-bottom:.8rem;"><strong>' + (orig.model_id || '?') + '</strong> vs <strong style="color:var(--green)">' + (opt.model_id || '?') + '</strong></p>';
        html += svgBarChart(perfMetrics, orig.model_id || 'Original', opt.model_id || 'Optimized', 'original_vs_optimized');
        html += '</div>';
    } else {
        html += '<p style="color:#8b949e;">Run benchmarks on both original and optimized models (Step 6) to see a detailed comparison here.</p>';
    }
    // ── Save / Discard optimized model ──
    const optModel = window._optimizedModel || (opt && opt.model_id) || '';
    if (optModel) {
        html += '<div id="save-discard-bar" style="margin-top:1.5rem;padding:1rem;border:1px solid var(--border);border-radius:8px;background:var(--card-bg);">';
        html += '<strong style="font-size:1.05rem;">&#128230; Optimized Model: <code>' + optModel + '</code></strong>';
        html += '<div style="display:flex;gap:.75rem;flex-wrap:wrap;margin-top:.75rem;">';
        html += '<button id="btn-keep" class="btn btn-green" onclick="keepOptimizedModel()" style="min-width:160px;">&#10003; Keep Model</button>';
        html += '<button id="btn-discard" class="btn btn-outline" onclick="discardOptimizedModel()" style="min-width:160px;border-color:#f85149;color:#f85149;">&#128465; Discard Model</button>';
        html += '</div>';
        html += '<p style="color:#8b949e;margin-top:.5rem;font-size:.85rem;">Keep saves the optimized model in Ollama. Discard deletes it to free disk space.</p>';
        html += '</div>';
    }
    html += '<div style="margin-top:1.5rem;padding:1rem;border:1px solid var(--green);border-radius:8px;background:#0d1f0d;">';
    html += '<strong style="color:var(--green);font-size:1.1rem;">&#127881; Workflow Complete</strong>';
    html += '<div style="display:flex;gap:.75rem;flex-wrap:wrap;margin-top:.75rem;">';
    html += '<a href="/chat" class="btn btn-primary" style="text-decoration:none;">&#128172; Chat with Model</a>';
    html += '<button class="btn btn-purple" onclick="setStep(5, true)">&#9889; Try Different Optimization</button>';
    html += '<button class="btn btn-outline" onclick="setStep(2, true)">&#128260; Start Over</button>';
    html += '</div>';
    html += '<p style="color:#8b949e;margin-top:.5rem;font-size:.85rem;">Not satisfied? Try a different quantization method, switch task modes, or <a href="/chat" style="color:var(--accent);">ask CyberForge Chat</a> for advice.</p>';
    html += '</div>';
    showResult('final-result', html);
}

// ── Persist mode to sessionStorage for resume ──
(function() {
    const saved = sessionStorage.getItem('cf_selected_mode');
    if (saved) {
        window._selectedMode = saved;
        // Highlight the saved mode card
        document.querySelectorAll('#step2 .mode-card').forEach(function(c) {
            if (c.getAttribute('onclick') && c.getAttribute('onclick').indexOf(saved) !== -1) {
                c.classList.add('selected');
            }
        });
    }
})();
setStep(2, true);
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
<p style="color:#8b949e;margin-bottom:.75rem;">Search HuggingFace Hub for models that match your hardware. Shows what you can run <em>natively</em> and what may fit <strong>after GGUF quantization</strong> (estimates are heuristic).</p>
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
<div style="display:flex;gap:.5rem;margin:1rem 0;">
  <button class="btn btn-primary" id="bcmp-origopt" onclick="setBenchCmpMode('origopt')">&#128260; Original vs Optimized</button>
  <button class="btn btn-outline" id="bcmp-modmod" onclick="setBenchCmpMode('modmod')">&#128257; Model vs Model</button>
</div>

<div id="bcmp-panel-origopt">
  <p style="color:#8b949e;font-size:.9rem;margin-bottom:.5rem;">Select two cards for the same base model (one before and one after optimization).</p>
  <div class="grid" style="margin:1rem 0;">
    <div class="form-group">
      <label>Original Card</label>
      <select id="cmp-base"><option value="">-- loading cards --</option></select>
    </div>
    <div class="form-group">
      <label>Optimized Card</label>
      <select id="cmp-opt"><option value="">-- loading cards --</option></select>
    </div>
  </div>
</div>

<div id="bcmp-panel-modmod" style="display:none;">
  <p style="color:#8b949e;font-size:.9rem;margin-bottom:.5rem;">Select any two cards to compare side-by-side.</p>
  <div class="grid" style="margin:1rem 0;">
    <div class="form-group">
      <label>Left Model (A)</label>
      <select id="cmp-left"><option value="">-- loading cards --</option></select>
    </div>
    <div class="form-group">
      <label>Right Model (B)</label>
      <select id="cmp-right"><option value="">-- loading cards --</option></select>
    </div>
  </div>
</div>

<div style="display:flex;gap:.75rem;flex-wrap:wrap;">
  <button class="btn btn-outline" onclick="refreshCardDropdowns()">&#128451; Refresh Cards</button>
  <button class="btn btn-green" onclick="runCompare()">&#128200; Compare</button>
</div>
<div class="result-box" id="compare-result"></div>
"""

    bench_js = """
// ── Bench page compare mode switching ──
let _benchCmpMode = 'origopt';
function setBenchCmpMode(mode) {
    _benchCmpMode = mode;
    ['origopt','modmod'].forEach(m => {
        document.getElementById('bcmp-panel-' + m).style.display = m === mode ? '' : 'none';
        const btn = document.getElementById('bcmp-' + m);
        btn.className = m === mode ? 'btn btn-primary' : 'btn btn-outline';
    });
}

// ── SVG bar chart builder (shared) ──
function svgBarChart(metrics, leftLabel, rightLabel, compareMode) {
    if (!metrics || metrics.length === 0) return '<p style="color:#8b949e;font-style:italic;">No chart data — run benchmarks for both models first.</p>';
    const valid = metrics.filter(m => m.left != null && m.right != null);
    if (valid.length === 0) return '<p style="color:#8b949e;font-style:italic;">No chart data — all metrics are null for this task mode.</p>';
    const W = 600, barH = 22, gap = 50, padL = 120, padR = 80;
    const H = valid.length * gap + 30;
    let svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;max-width:600px;font-family:sans-serif;font-size:12px;">';
    valid.forEach(function(m, i) {
        const y = i * gap + 10;
        const maxVal = Math.max(m.left, m.right, 0.001);
        const bW = Math.max((m.left / maxVal) * (W - padL - padR), 2);
        const oW = Math.max((m.right / maxVal) * (W - padL - padR), 2);
        const diff = m.right - m.left;
        const pct = m.left !== 0 ? ((diff / m.left) * 100).toFixed(1) : '0.0';
        const improved = m.lowerIsBetter ? diff < 0 : diff > 0;
        const pctColor = improved ? '#3fb950' : (diff === 0 ? '#8b949e' : '#f85149');
        svg += '<text x="' + (padL - 5) + '" y="' + (y + 10) + '" text-anchor="end" fill="#c9d1d9" font-weight="600">' + m.name + '</text>';
        svg += '<rect x="' + padL + '" y="' + y + '" width="' + bW + '" height="' + barH + '" rx="3" fill="#58a6ff" opacity="0.85"/>';
        svg += '<text x="' + (padL + bW + 4) + '" y="' + (y + 15) + '" fill="#58a6ff" font-size="11">' + m.left.toFixed(1) + m.unit + '</text>';
        svg += '<rect x="' + padL + '" y="' + (y + barH + 2) + '" width="' + oW + '" height="' + barH + '" rx="3" fill="#3fb950" opacity="0.85"/>';
        svg += '<text x="' + (padL + oW + 4) + '" y="' + (y + barH + 17) + '" fill="#3fb950" font-size="11">' + m.right.toFixed(1) + m.unit + '</text>';
        const arrow = improved ? '\\u25B2' : (diff === 0 ? '\\u2500' : '\\u25BC');
        svg += '<text x="' + (W - 5) + '" y="' + (y + barH + 5) + '" text-anchor="end" fill="' + pctColor + '" font-weight="600" font-size="11">' + arrow + ' ' + Math.abs(pct) + '%</text>';
    });
    svg += '</svg>';
    svg += '<div style="display:flex;gap:1.5rem;margin-top:.5rem;font-size:.8rem;color:#8b949e;">';
    svg += '<span><span style="display:inline-block;width:12px;height:12px;background:#58a6ff;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>' + leftLabel + '</span>';
    svg += '<span><span style="display:inline-block;width:12px;height:12px;background:#3fb950;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>' + rightLabel + '</span>';
    svg += '</div>';
    return svg;
}

// ── Size reduction banner ──
function sizeReductionBanner(d) {
    if (!d || !d.model_size_mb || d.model_size_mb.baseline == null || d.model_size_mb.optimized == null) return '';
    const orig = d.model_size_mb.baseline;
    const opt = d.model_size_mb.optimized;
    if (orig === 0 && opt === 0) return '<p style="color:#8b949e;font-style:italic;margin:.5rem 0;">Model size: not reported by backend.</p>';
    const redPct = orig > 0 ? ((orig - opt) / orig * 100).toFixed(1) : 0;
    const reduced = opt < orig;
    const color = reduced ? 'var(--green)' : (opt === orig ? '#8b949e' : '#f85149');
    let h = '<div class="card" style="border-color:' + color + ';margin:1rem 0;text-align:center;">';
    h += '<div style="font-size:2rem;font-weight:700;color:' + color + ';">';
    if (reduced) h += '&#9660; ' + Math.abs(redPct) + '% smaller';
    else if (opt === orig) h += '&#9644; Same size';
    else h += '&#9650; ' + Math.abs(redPct) + '% larger';
    h += '</div>';
    h += '<div style="color:#8b949e;margin-top:.25rem;">' + orig.toFixed(0) + ' MB &rarr; ' + opt.toFixed(0) + ' MB</div>';
    h += '</div>';
    return h;
}

// ── Render compare result (shared) ──
function renderCompareResult(data) {
    const cmp = data.comparison || data;
    const d = cmp.deltas || {};
    const mode = cmp.compare_mode || 'model_vs_model';
    const bInfo = cmp.baseline || {};
    const oInfo = cmp.optimized || {};
    const bLabel = bInfo.model_id || 'Model A';
    const oLabel = oInfo.model_id || 'Model B';
    let html = '';

    if (mode === 'original_vs_optimized') {
        html += '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:1rem;">';
        html += '<span style="background:var(--green);color:#0d1117;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600;">Original vs Optimized</span>';
        html += '</div>';
        html += '<h3 style="color:var(--green)">&#128260; Optimization Impact</h3>';
        html += '<table style="margin-bottom:1rem;"><tbody>';
        html += '<tr><td style="color:#8b949e;">Base Model</td><td><strong>' + bLabel + '</strong></td></tr>';
        html += '<tr><td style="color:#8b949e;">Optimized Artifact</td><td><strong style="color:var(--green);">' + oLabel + '</strong></td></tr>';
        html += '<tr><td style="color:#8b949e;">Original Label</td><td>' + (bInfo.label || 'baseline') + '</td></tr>';
        html += '<tr><td style="color:#8b949e;">Optimized Label</td><td>' + (oInfo.label || '—') + '</td></tr>';
        html += '</tbody></table>';
        html += sizeReductionBanner(d);
    } else {
        html += '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:1rem;">';
        html += '<span style="background:#58a6ff;color:#0d1117;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600;">Model vs Model</span>';
        html += '</div>';
        html += '<h3 style="color:#58a6ff;">&#128257; Side-by-Side Comparison</h3>';
        html += '<p style="color:#8b949e;margin-bottom:1rem;"><strong style="color:#58a6ff;">' + bLabel + '</strong> vs <strong style="color:#3fb950;">' + oLabel + '</strong></p>';
    }

    const perfMetrics = [];
    if (d.latency_ms && d.latency_ms.baseline != null) perfMetrics.push({name:'Latency', left:d.latency_ms.baseline, right:d.latency_ms.optimized, unit:' ms', lowerIsBetter:true});
    if (d.throughput_tok_s && d.throughput_tok_s.baseline != null) perfMetrics.push({name:'Throughput', left:d.throughput_tok_s.baseline, right:d.throughput_tok_s.optimized, unit:' tok/s', lowerIsBetter:false});
    if (d.load_time_ms && d.load_time_ms.baseline != null) perfMetrics.push({name:'Load Time', left:d.load_time_ms.baseline, right:d.load_time_ms.optimized, unit:' ms', lowerIsBetter:true});
    if (d.model_size_mb && d.model_size_mb.baseline != null) perfMetrics.push({name:'Size', left:d.model_size_mb.baseline, right:d.model_size_mb.optimized, unit:' MB', lowerIsBetter:true});
    html += '<div class="card" style="border-color:var(--green);margin-top:1rem;">';
    html += '<h3>&#9889; Performance &amp; Size</h3>';
    html += svgBarChart(perfMetrics, bLabel, oLabel, mode);
    html += '</div>';

    const resMetrics = [];
    if (d.vram_peak_mb && d.vram_peak_mb.baseline != null) resMetrics.push({name:'VRAM Peak', left:d.vram_peak_mb.baseline, right:d.vram_peak_mb.optimized, unit:' MB', lowerIsBetter:true});
    if (d.ram_peak_mb && d.ram_peak_mb.baseline != null) resMetrics.push({name:'RAM Peak', left:d.ram_peak_mb.baseline, right:d.ram_peak_mb.optimized, unit:' MB', lowerIsBetter:true});
    html += '<div class="card" style="border-color:var(--purple);margin-top:1rem;">';
    html += '<h3>&#128190; Resource Usage</h3>';
    html += svgBarChart(resMetrics, bLabel, oLabel, mode);
    html += '</div>';

    const qualMetrics = [];
    if (d.exact_match && d.exact_match.baseline != null) qualMetrics.push({name:'Exact Match', left:d.exact_match.baseline, right:d.exact_match.optimized, unit:'', lowerIsBetter:false});
    if (d.verifier_pass_rate && d.verifier_pass_rate.baseline != null) qualMetrics.push({name:'Verifier Pass', left:d.verifier_pass_rate.baseline, right:d.verifier_pass_rate.optimized, unit:'', lowerIsBetter:false});
    if (d.structured_output_validity && d.structured_output_validity.baseline != null) qualMetrics.push({name:'Structured Output', left:d.structured_output_validity.baseline, right:d.structured_output_validity.optimized, unit:'', lowerIsBetter:false});
    if (d.syntax_error_rate && d.syntax_error_rate.baseline != null) qualMetrics.push({name:'Syntax Error Rate', left:d.syntax_error_rate.baseline, right:d.syntax_error_rate.optimized, unit:'', lowerIsBetter:true});
    html += '<div class="card" style="border-color:#58a6ff;margin-top:1rem;">';
    html += '<h3>&#127919; Quality Metrics</h3>';
    html += svgBarChart(qualMetrics, bLabel, oLabel, mode);
    html += '</div>';

    html += '<div class="card" style="margin-top:1rem;">';
    html += '<h3>&#128202; Full Metric Table</h3>';
    const rows = [];
    Object.keys(d).forEach(function(key) {
        const m = d[key];
        if (!m || m.baseline == null || m.optimized == null) return;
        const imp = m.improved;
        const color = imp === true ? 'var(--green)' : (imp === false ? '#f85149' : '#8b949e');
        const arrow = imp === true ? '&#9650;' : (imp === false ? '&#9660;' : '&#9644;');
        rows.push([key.replace(/_/g,' '), m.baseline.toFixed(2), m.optimized.toFixed(2), '<span style="color:' + color + ';font-weight:600;">' + arrow + ' ' + (m.percent != null ? Math.abs(m.percent).toFixed(1) + '%' : '—') + '</span>']);
    });
    if (rows.length) html += buildTable([mode === 'original_vs_optimized' ? 'Metric' : 'Metric', bLabel, oLabel, 'Change'], rows);
    else html += '<p style="color:#8b949e;font-style:italic;">No comparable metrics available.</p>';
    html += '</div>';

    if (data.report_path) html += '<p style="color:#8b949e;margin-top:.5rem;">Report: ' + data.report_path + '</p>';
    return html;
}

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
        await refreshCardDropdowns();
    } catch (e) { showResult('model-bench-result', '<span style="color:#f85149">' + e.message + '</span>'); }
}
async function refreshCardDropdowns() {
    try {
        const cards = await cyberGet('/action/bench-cards');
        const selectors = ['cmp-base','cmp-opt','cmp-left','cmp-right'];
        const prevVals = {};
        selectors.forEach(id => { const el = document.getElementById(id); if (el) prevVals[id] = el.value; });
        selectors.forEach(id => {
            const el = document.getElementById(id);
            if (!el) return;
            const isLeft = id.endsWith('-base') || id.endsWith('-left');
            const placeholder = isLeft ? '-- select card --' : '-- select card --';
            el.innerHTML = '';
            if (Array.isArray(cards) && cards.length) {
                el.innerHTML += '<option value="">' + placeholder + '</option>';
                cards.forEach(c => { el.innerHTML += '<option value="' + c + '">' + c + '</option>'; });
                if (prevVals[id]) el.value = prevVals[id];
                if (!el.value && cards.length >= 2) {
                    el.value = isLeft ? cards[cards.length - 2] : cards[cards.length - 1];
                }
            } else {
                el.innerHTML = '<option value="">-- no cards yet --</option>';
            }
        });
    } catch (e) { /* silent */ }
}
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
                opt.textContent = name + ' (' + sizeGB + ' GB)' + (safe ? '' : ' \\u26a0\\ufe0f VRAM risk');
                sel.appendChild(opt);
                if (safe && safeIdx < 0) safeIdx = idx;
                if (selectedId && name === selectedId) selectedIdx = idx;
            });
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
    let base, opt;
    if (_benchCmpMode === 'modmod') {
        base = document.getElementById('cmp-left').value;
        opt = document.getElementById('cmp-right').value;
    } else {
        base = document.getElementById('cmp-base').value;
        opt = document.getElementById('cmp-opt').value;
    }
    if (!base || !opt) { showResult('compare-result', '<span style="color:#f85149">Select both cards</span>'); return; }
    if (base === opt) { showResult('compare-result', '<span style="color:#f85149">Select two different cards</span>'); return; }
    try {
        const data = await cyberPost('/action/compare', {baseline_file: base, optimized_file: opt, save_report: true});
        if (data._error) { showResult('compare-result', '<span style="color:#f85149">' + data._error + '</span>'); return; }
        showResult('compare-result', renderCompareResult(data));
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
  <div class="card"><strong>Suricata Validator</strong><br><code>POST /api/cyber/validate/suricata</code></div>
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
<h1>&#128640; Model Optimization <span style="font-size:.9rem;background:#6e40c9;color:#fff;padding:3px 10px;border-radius:6px;vertical-align:middle;">Advanced</span></h1>
<div style="padding:.75rem 1rem;border:1px solid #6e40c9;border-radius:8px;background:#1a0d2e;margin-bottom:1rem;">
<strong style="color:#d2a8ff;">&#9888; Expert Tool</strong>
<span style="color:#8b949e;"> &mdash; This page exposes all optimization backends including HuggingFace-only methods. For the guided experience, use the <a href="/workflow" style="color:var(--accent);">Workflow</a> instead.</span>
</div>
<p style="color:#8b949e;margin-bottom:1rem;">Quantize and compress models using standard algorithms (GGUF, GPTQ, AWQ, bitsandbytes). Reduce memory, increase throughput, maintain quality.</p>

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
        const _timeout = new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 15000));
        const status = await Promise.race([cyberGet('/action/quantize-status'), _timeout]);
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
    } catch(e) {
        document.getElementById('env-status').innerHTML = e.message === 'timeout'
            ? '<span style="color:#d29922">Backend check timed out &mdash; services may still be starting. <a href="javascript:location.reload()" style="color:var(--accent)">Retry</a></span>'
            : '<span style="color:#f85149">Failed to check status</span>';
    }
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
        const _timeout = new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 15000));
        const status = await Promise.race([cyberGet('/action/route-status'), _timeout]);
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
    } catch(e) {
        document.getElementById('route-status').innerHTML = e.message === 'timeout'
            ? '<span style="color:#d29922">Router check timed out &mdash; services may still be starting. <a href="javascript:location.reload()" style="color:var(--accent)">Retry</a></span>'
            : '<span style="color:#f85149">Failed to check router status</span>';
    }
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
<p style="color:#8b949e;margin-bottom:1rem;">Chat with your locally-running model. Powered by Ollama.</p>

<div id="ollama-banner" class="card" style="border-color:var(--yellow);display:none;">
  <strong style="color:var(--yellow);">&#9888; Ollama Status</strong>
  <span id="ollama-banner-text"></span>
  <div style="margin-top:.5rem;">
    <button class="btn btn-sm btn-green" onclick="startOllama()">Start Ollama</button>
    <a href="https://ollama.com/download" target="_blank" class="btn btn-sm btn-outline" style="margin-left:.5rem;">Install Ollama</a>
  </div>
</div>

<div id="selected-model-banner" class="card" style="border-color:var(--accent);display:none;">
  <div style="display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;">
    <strong style="color:var(--accent);">Selected Model:</strong>
    <span id="sel-model-name" style="font-weight:600;"></span>
    <span id="sel-model-status" class="badge"></span>
  </div>
  <div id="sel-model-action" style="margin-top:.5rem;display:none;"></div>
</div>

<div class="card">
  <div style="margin-bottom:.5rem;">
    <label>Chat Model</label>
    <div style="display:flex;gap:.5rem;">
      <select id="chat-model-select" style="flex:1;"><option value="">Loading models...</option></select>
      <button class="btn btn-sm btn-outline" onclick="refreshChatModels()">&#8635;</button>
    </div>
    <p style="font-size:.8rem;color:#8b949e;margin:.25rem 0 0;">Only Ollama-loaded models appear here. Select a model in the <a href="/workflow" style="color:var(--accent);">Workflow</a> to use it everywhere.</p>
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
let _ollamaAvailable = false;
let _ollamaModels = [];

async function refreshChatModels() {
    const sel = document.getElementById('chat-model-select');
    const mc = CyberForge.getMachineClass();
    const safeLimit = mc && mc.vram_total_mb ? mc.vram_total_mb * 1024 * 1024 * 0.85 : 5 * 1024 * 1024 * 1024;
    try {
        const models = await cyberGet('/action/ollama-models');
        const list = Array.isArray(models) ? models : (models.models || []);
        _ollamaModels = list.map(m => m.name || m);
        if (list.length > 0) {
            _ollamaAvailable = true;
            sel.innerHTML = '';
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
            if (selectedIdx >= 0) sel.selectedIndex = selectedIdx;
            else if (safeIdx >= 0) sel.selectedIndex = safeIdx;
            _chatModel = sel.value;
            document.getElementById('ollama-banner').style.display = 'none';
        } else {
            _ollamaAvailable = true;
            sel.innerHTML = '<option value="">No models found \u2014 run: ollama pull qwen2.5:3b</option>';
            _chatModel = null;
        }
    } catch (e) {
        sel.innerHTML = '<option value="">Ollama not available</option>';
        _chatModel = null;
        _ollamaAvailable = false;
        checkOllamaStatus();
    }
    showSelectedModelBanner();
}
function showSelectedModelBanner() {
    const m = CyberForge.getSelected();
    const banner = document.getElementById('selected-model-banner');
    if (!banner) return;
    if (!m) { banner.style.display = 'none'; return; }
    const name = m.display_name || m.ollama_tag || m.hf_repo || m.id || '';
    document.getElementById('sel-model-name').textContent = name;
    const statusEl = document.getElementById('sel-model-status');
    const actionEl = document.getElementById('sel-model-action');
    const ollamaTag = m.ollama_tag || '';
    const hfRepo = m.hf_repo || '';
    const inOllama = ollamaTag && _ollamaModels.some(n => n === ollamaTag || n.startsWith(ollamaTag.split(':')[0] + ':'));
    if (inOllama) {
        statusEl.className = 'badge badge-ok'; statusEl.textContent = 'Ready in Ollama';
        actionEl.style.display = 'none';
        const sel = document.getElementById('chat-model-select');
        for (let i = 0; i < sel.options.length; i++) {
            if (sel.options[i].value === ollamaTag || sel.options[i].value.startsWith(ollamaTag.split(':')[0] + ':')) {
                sel.selectedIndex = i; _chatModel = sel.value; break;
            }
        }
    } else if (_ollamaAvailable && ollamaTag) {
        statusEl.className = 'badge badge-warn'; statusEl.textContent = 'Not pulled yet';
        actionEl.innerHTML = '<button class="btn btn-sm btn-green" onclick="pullSelectedModel()">Pull ' + ollamaTag + '</button><span style="color:#8b949e;font-size:.8rem;margin-left:.5rem;">Downloads the model (~2\u20138 GB)</span>';
        actionEl.style.display = '';
    } else if (!_ollamaAvailable) {
        statusEl.className = 'badge badge-err'; statusEl.textContent = 'Ollama not running';
        actionEl.innerHTML = '<span style="color:#8b949e;font-size:.85rem;">Start Ollama to chat with this model.</span>';
        actionEl.style.display = '';
    } else if (hfRepo && !ollamaTag) {
        statusEl.className = 'badge badge-warn'; statusEl.textContent = 'HF-only model';
        actionEl.innerHTML = '<span style="color:#8b949e;font-size:.85rem;">No Ollama tag. Use <a href="/workflow" style="color:var(--accent);">Workflow</a> to quantize into Ollama format.</span>';
        actionEl.style.display = '';
    } else { banner.style.display = 'none'; return; }
    banner.style.display = '';
}
async function pullSelectedModel() {
    const m = CyberForge.getSelected();
    if (!m || !m.ollama_tag) return;
    const actionEl = document.getElementById('sel-model-action');
    actionEl.innerHTML = '<span class="spinner"></span> Pulling <strong>' + m.ollama_tag + '</strong>\u2026 this may take several minutes.';
    try {
        const result = await cyberPost('/action/ollama-pull', {model_name: m.ollama_tag});
        if (result._error || result.error) {
            actionEl.innerHTML = '<span style="color:#f85149;">Pull failed: ' + (result._error || result.error) + '</span>';
        } else {
            actionEl.innerHTML = '<span style="color:var(--green);">\u2713 Model pulled! Refreshing\u2026</span>';
            setTimeout(async () => { await refreshChatModels(); }, 1000);
        }
    } catch(e) { actionEl.innerHTML = '<span style="color:#f85149;">Error: ' + e.message + '</span>'; }
}
async function checkOllamaStatus() {
    try {
        const status = await cyberGet('/action/ollama-status');
        if (!status.available) {
            _ollamaAvailable = false;
            const banner = document.getElementById('ollama-banner');
            document.getElementById('ollama-banner-text').innerHTML = ' Ollama is not running. Chat requires Ollama to serve models locally.';
            banner.style.display = '';
        } else {
            _ollamaAvailable = true;
            document.getElementById('ollama-banner').style.display = 'none';
        }
    } catch(e) {
        _ollamaAvailable = false;
        const banner = document.getElementById('ollama-banner');
        document.getElementById('ollama-banner-text').innerHTML = ' Cannot reach Ollama. <a href="https://ollama.com/download" target="_blank" style="color:var(--accent);">Install Ollama</a> and run <code>ollama serve</code>.';
        banner.style.display = '';
    }
}
async function startOllama() {
    document.getElementById('ollama-banner-text').innerHTML = ' <span class="spinner"></span> Starting Ollama\u2026';
    try {
        const result = await cyberPost('/action/ollama-start');
        if (result.started) {
            document.getElementById('ollama-banner-text').textContent = ' Ollama started! Refreshing\u2026';
            document.getElementById('ollama-banner').style.borderColor = 'var(--green)';
            setTimeout(async () => { await refreshChatModels(); document.getElementById('ollama-banner').style.display = 'none'; }, 2000);
        } else {
            document.getElementById('ollama-banner-text').innerHTML = ' Failed: ' + (result.error || 'Unknown') + '<br><span style="color:#8b949e;font-size:.85rem;">Try <code>ollama serve</code> manually.</span>';
        }
    } catch(e) { document.getElementById('ollama-banner-text').textContent = ' Error: ' + e.message; }
}
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
