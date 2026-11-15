"""Web server for the Disruption Response Desk (no Streamlit).

    set GROQ_API_KEY=your_key
    python server.py            ->  http://localhost:8000
"""
import asyncio
import json
import os
import pathlib
import time
import webbrowser

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from agents.errors import UserInputError
from agents.tools import (baseline_rank, clean_event_text, compute_options, inventory_tool, load_suppliers,
                          primary_affected)

ROOT = pathlib.Path(__file__).parent
EVAL_FILE = ROOT / "data" / "last_eval.json"
SCENARIOS = json.load(open(ROOT / "data" / "scenarios.json"))
RUN_TIMEOUT = 240            # seconds for one run before we give up
RUN_SLOTS = asyncio.Semaphore(3)   # at most 3 agent runs at once (Groq rate limits)
EVAL_LOCK = asyncio.Lock()


def serialise(bb):
    s = bb.state
    keys = ("event", "stock", "options", "ranked_plan", "shortlist", "shortlist_why", "retrieved",
            "inventory_note", "tradeoff_note", "ranking_source", "baseline_ranking", "warnings")
    out = {k: s.get(k) for k in keys}
    out["warnings"] = out["warnings"] or []
    out["log"] = [list(x) for x in bb.log]
    return out


def friendly(e):
    """Turn any exception into a message a user can act on. 'input' = their text, 'system' = our side."""
    if isinstance(e, UserInputError):
        return {"error": str(e), "kind": "input"}
    m = str(e).lower()
    if "401" in m or "invalid api key" in m or "authentication" in m:
        msg = "Groq rejected the API key. Check GROQ_API_KEY, restart the server, or use Demo mode."
    elif "429" in m or "rate limit" in m or "rate_limit" in m:
        msg = "Groq's rate limit was hit. Wait about a minute and try again, or use Demo mode."
    elif "timeout" in m or "timed out" in m or "connect" in m:
        msg = "Could not reach Groq (network timeout). Check the internet connection and try again, or use Demo mode."
    elif "tool_use_failed" in m or "tool call" in m:
        msg = "The model produced a malformed tool call. Please run it again."
    else:
        msg = "Something went wrong inside the agents. Please try again, or use Demo mode."
    return {"error": msg, "kind": "system", "detail": str(e)[:300]}


def ndjson(gen):
    return StreamingResponse(gen, media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


def line(obj):
    return json.dumps(obj) + "\n"


# ---------------- demo mode: same tools, no LLM (backup if Groq is unreachable) ----------------
def demo_run(event_text):
    import re
    from agents.blackboard import Blackboard
    from agents.tools import supported_goods
    text = clean_event_text(event_text)                       # same gate 1 as the real run
    sups = load_suppliers()
    low = text.lower()
    region = next((r for r in sorted({s["region"] for s in sups}) if r.lower() in low), None)
    port = next((p for p in sorted({s["port"] for s in sups}) if p.lower() in low), None)
    goods = [g for g in supported_goods() if g in low]
    if not goods:
        raise UserInputError("Demo mode could not find a known good in the text. Supported goods: " + ", ".join(supported_goods()) + ".")
    m = re.search(r"(\d+)\s*day", low)
    raw_dur = int(m.group(1)) if m else 14
    dur = max(1, min(180, raw_dur))
    good, warnings = goods[0], []
    if raw_dur != dur:
        warnings.append("Duration was limited to the range 1 to 180 days.")
    if len(goods) > 1:
        warnings.append(f"Several goods were mentioned ({', '.join(goods)}). Planning for {good}; run the others separately.")
    if not region and not port:
        warnings.append("No known port or region was recognised, so your primary supplier is assumed to be affected.")
    kinds = [("strike", "port strike"), ("typhoon", "weather"), ("flood", "weather"), ("storm", "weather"), ("hurricane", "weather"),
             ("shortage", "shortage"), ("congestion", "congestion"), ("delay", "congestion"), ("blockade", "blockade/customs"),
             ("customs", "blockade/customs"), ("fire", "fire/outage"), ("outage", "fire/outage")]
    etype = next((v for k, v in kinds if k in low), "other")
    ev = {"type": etype, "region": region, "port": port, "goods": [good], "duration_days": dur}
    ids = [s["id"] for s in sups if good in s["goods"] and s["port"] != port and s["region"] != region]
    if not ids:
        warnings.append("No alternative suppliers exist outside the disrupted area, so only the wait option is available.")
    bb = Blackboard()
    bb.write("EventInterpreter", "event", ev)
    bb.state["stock"] = {good: inventory_tool(good)}
    bb.log.append(("Inventory", "stock"))
    bb.write("SupplierAlternatives", "shortlist", ids)
    bb.write("SupplierAlternatives", "shortlist_why", "Demo mode: every supplier of this good outside the disrupted port/region.")
    bb.state["retrieved"] = {s["id"]: dict(s, rag_score=None) for s in sups if s["id"] in ids}
    opts = compute_options(good, ids, dur, event=ev)
    bb.write("CostDelay", "options", opts)
    plan = baseline_rank([dict(o) for o in opts])
    for i, o in enumerate(plan, 1):
        o["rank"] = i
        o["reason"] = f"Weighted score {o['score']} (60% cost, 40% delay); demo mode, no LLM."
    bb.write("OperationsManager", "ranked_plan", plan)
    bb.write("OperationsManager", "ranking_source", "demo")
    bb.write("OperationsManager", "baseline_ranking", [o["id"] for o in plan])
    bb.write("Inventory", "inventory_note", "Demo mode: stock read directly from the inventory tool.")
    bb.write("CostDelay", "tradeoff_note", "Demo mode: options computed directly by the cost-delay tool.")
    bb.state["warnings"] = warnings
    return bb


# ---------------- routes ----------------
async def status(request):
    return JSONResponse({"has_key": bool(os.environ.get("GROQ_API_KEY")),
                         "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")})


async def data(request):
    from agents.tools import supported_goods
    return JSONResponse({"suppliers": load_suppliers(), "inventory": {g: inventory_tool(g) for g in supported_goods()},
                         "scenarios": SCENARIOS, "goods": supported_goods()})


async def run(request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError
    except Exception:
        return JSONResponse({"error": "Malformed request.", "kind": "input"}, status_code=400)
    demo = bool(body.get("demo"))
    try:
        text = clean_event_text(body.get("event"))             # gate 1 runs before any streaming or LLM call
    except UserInputError as e:
        return JSONResponse({"error": str(e), "kind": "input"}, status_code=400)
    if not demo and not os.environ.get("GROQ_API_KEY"):
        return JSONResponse({"error": "GROQ_API_KEY is not set. Use Demo mode, or set the key and restart the server.", "kind": "system"}, status_code=400)

    async def gen():
        t0 = time.time()
        if demo:
            try:
                bb = demo_run(text)
                for name in ("EventInterpreter", "Inventory", "SupplierAlternatives", "CostDelay", "OperationsManager"):
                    yield line({"step": name, "state": "start"})
                    await asyncio.sleep(0 if body.get("fast") else 0.25)
                    yield line({"step": name, "state": "done"})
                yield line({"result": serialise(bb), "secs": round(time.time() - t0, 1), "demo": True})
            except Exception as e:
                yield line(friendly(e))
            return
        from pipeline import run_desk
        if RUN_SLOTS.locked():
            yield line({"info": "The desk is busy with other runs; yours is queued."})
        async with RUN_SLOTS:
            q = asyncio.Queue()
            task = asyncio.create_task(asyncio.wait_for(run_desk(text, on_step=lambda n, s: q.put_nowait({"step": n, "state": s})), RUN_TIMEOUT))
            while not task.done() or not q.empty():
                try:
                    yield line(await asyncio.wait_for(q.get(), timeout=0.3))
                except asyncio.TimeoutError:
                    pass
            try:
                yield line({"result": serialise(task.result()), "secs": round(time.time() - t0, 1)})
            except asyncio.TimeoutError:
                yield line({"error": f"The run took longer than {RUN_TIMEOUT} seconds and was stopped. Try again or use Demo mode.", "kind": "system"})
            except Exception as e:
                yield line(friendly(e))
    return ndjson(gen())


async def replay(request):
    """Rebuild a scenario from the saved real LLM evaluation (real Groq ranking and reasons, no network needed)."""
    sid = request.query_params.get("id", "")
    scen = next((x for x in SCENARIOS if x["id"] == sid), None)
    try:
        saved = json.load(open(EVAL_FILE)) if EVAL_FILE.exists() else []
    except Exception:
        saved = []
    rec = next((r for r in saved if r["id"] == sid and "error" not in r), None)
    if not scen or not rec:
        return JSONResponse({"error": "No saved LLM run for that scenario yet. Run the evaluation once first.", "kind": "input"}, status_code=404)
    bb = demo_run(scen["event"])
    by_id = {o["id"]: o for o in bb.read("ranked_plan")}
    plan = []
    for o in rec["plan"]:
        if o["id"] in by_id:
            plan.append(dict(by_id[o["id"]], rank=o["rank"], reason=o["reason"]))
    if len(plan) != len(by_id):
        return JSONResponse({"error": "The saved run no longer matches the current data. Run the evaluation again.", "kind": "input"}, status_code=409)
    bb.write("OperationsManager", "ranked_plan", plan)
    bb.write("OperationsManager", "ranking_source", "replay")
    bb.state["warnings"] = []
    return JSONResponse({"result": serialise(bb), "event": scen["event"]})


async def evaluate(request):
    if not os.environ.get("GROQ_API_KEY"):
        return JSONResponse({"error": "GROQ_API_KEY is not set.", "kind": "system"}, status_code=400)
    if EVAL_LOCK.locked():
        return JSONResponse({"error": "An evaluation is already running. Please wait for it to finish.", "kind": "system"}, status_code=429)
    from pipeline import run_desk

    async def gen():
        async with EVAL_LOCK:
            sem, q = asyncio.Semaphore(2), asyncio.Queue()

            async def one(s):
                async with sem:
                    t0 = time.time()
                    try:
                        bb = await asyncio.wait_for(run_desk(s["event"]), RUN_TIMEOUT)
                        plan = bb.read("ranked_plan")
                        rec = {"id": s["id"], "event": s["event"], "expected": s["expected_best"], "llm_top": plan[0]["id"],
                               "rules_top": bb.read("baseline_ranking")[0], "source": bb.read("ranking_source"),
                               "plan": [{k: o[k] for k in ("rank", "id", "label", "total_cost", "delay_days", "stockout_days", "reason")} for o in plan]}
                    except Exception as e:
                        rec = {"id": s["id"], "event": s["event"], "expected": s["expected_best"], "error": friendly(e)["error"]}
                    rec["secs"] = round(time.time() - t0, 1)
                    await q.put(rec)
                    return rec
            tasks = [asyncio.create_task(one(s)) for s in SCENARIOS]
            done = 0
            while done < len(tasks):
                rec = await q.get()
                done += 1
                yield line({"progress": done, "total": len(tasks), "record": rec})
            results = sorted((t.result() for t in tasks), key=lambda r: r["id"])
            if all("error" not in r for r in results):          # never overwrite a good saved result with a broken run
                json.dump(results, open(EVAL_FILE, "w"), indent=1)
            yield line({"results": results})
    return ndjson(gen())


async def last_eval(request):
    try:
        return JSONResponse(json.load(open(EVAL_FILE)) if EVAL_FILE.exists() else [])
    except Exception:
        return JSONResponse([])


app = Starlette(routes=[
    Route("/api/status", status), Route("/api/data", data), Route("/api/last-eval", last_eval),
    Route("/api/replay", replay), Route("/api/run", run, methods=["POST"]), Route("/api/evaluate", evaluate, methods=["POST"]),
    Mount("/", StaticFiles(directory=ROOT / "web", html=True)),
])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Disruption Response Desk -> http://localhost:{port}   (Groq key {'found' if os.environ.get('GROQ_API_KEY') else 'NOT set: demo mode only'})")
    webbrowser.open(f"http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
