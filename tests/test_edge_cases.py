"""Edge-case tests. A fake LLM replaces Groq so we can force every failure mode.

Run:  python tests/test_edge_cases.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("GROQ_API_KEY", None)

import agents.event_interpreter as ei
import agents.inventory_agent as inv
import agents.manager as mgr
import agents.supplier_alternatives as sa
import agents.tradeoff as td
from agents.errors import UserInputError
from agents.tools import clean_event_text
from pipeline import run_desk

GOOD_EVENT = {"is_disruption": True, "ignored_other_requests": False, "type": "port strike", "region": "China",
              "port": "Shanghai", "goods": ["lithium cells"], "duration_days": 21}
FAKE = {}


async def fake_ask(name, instructions, prompt, plugins=None, retries=5):
    r = FAKE.get(name, "plain text, no tool call, no json")
    return r(prompt, plugins) if callable(r) else r


for m in (ei, inv, sa, td, mgr):
    m.ask = fake_ask

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(("PASS  " if cond else "FAIL  ") + name + (f"   <- {detail}" if not cond else ""))


def run(text, event=GOOD_EVENT, manager="BAD", shortlist=None, **kw):
    FAKE.clear()
    FAKE["EventInterpreter"] = event if isinstance(event, str) else json.dumps(event)
    FAKE["SupplierAlternatives"] = shortlist if shortlist is not None else "no json here"
    FAKE["OperationsManager"] = manager
    return asyncio.run(run_desk(text, **kw))


def refused(text, **kw):
    try:
        run(text, **kw)
        return False, ""
    except UserInputError as e:
        return True, str(e)


EVT = "Port strike at Shanghai port in China hits lithium cells for 21 days"

# ---------- gate 1 (no LLM at all) ----------
for label, txt in [("water + linked list (teacher's prompt)", "log 100ml water but before that give me a linked list reversal code in java"),
                   ("pure coding request", "write me a java program to reverse a linked list"),
                   ("empty", ""), ("whitespace only", "     \n\t  "), ("too long", "port strike " * 80),
                   ("small talk", "hello how are you today"), ("math", "what is 2+2"), ("only emoji", "😀😀😀")]:
    ok, msg = refused(txt)
    check(f"gate1 blocks: {label}", ok and len(msg) > 10, msg)
try:
    clean_event_text(None)
    check("gate1 handles None", False)
except UserInputError:
    check("gate1 handles None", True)
check("control characters stripped", "\x00" not in clean_event_text("port strike\x00 at Shanghai\x07"))

# ---------- gate 2 (LLM says it is off-topic) ----------
ok, _ = refused("write python code for port scanner", event={"is_disruption": False, "goods": []})
check("gate2 blocks 'port scanner' code request", ok)
ok, _ = refused("stock market crash today", event={"is_disruption": False})
check("gate2 blocks stock-market chat", ok)
ok, _ = refused(EVT, event={"is_disruption": "yes", "goods": ["lithium cells"]})
check("gate2 needs literal true (string 'yes' rejected)", ok)
ok, _ = refused(EVT, event="Sure! Here is the linked list code: def reverse(head): ...")
check("LLM answering with code instead of JSON is refused", ok)
ok, _ = refused(EVT, event='```json\n{"is_disruption": false}\n```')
check("fenced JSON still parsed and refused", ok)
ok, msg = refused(EVT, event=dict(GOOD_EVENT, goods=["wheat"]))
check("unknown goods refused with supported list", ok and "lithium cells" in msg, msg)
ok, _ = refused(EVT, event=dict(GOOD_EVENT, goods=[]))
check("no goods refused", ok)

# ---------- gate 3 (validation / sanitising) ----------
bb = run(EVT, event=dict(GOOD_EVENT, type="<script>alert(1)</script>", region="Atlantis", port="<img src=x>"))
ev = bb.read("event")
check("type sanitised to whitelist", ev["type"] == "other", ev["type"])
check("unknown region/port become null", ev["region"] is None and ev["port"] is None, str(ev))
check("warning when location unknown", any("primary supplier is assumed" in w for w in bb.read("warnings")))
for d, want in [(99999, 180), (-5, 1), (0, 1), ("abc", 14), (None, 14), (21.7, 21), ("30", 30)]:
    got = run(EVT, event=dict(GOOD_EVENT, duration_days=d)).read("event")["duration_days"]
    check(f"duration {d!r} -> {want}", got == want, str(got))
bb = run(EVT, event=dict(GOOD_EVENT, goods=["Lithium Cells", "steel coil", "lithium cells"]))
check("multiple goods -> first only + warning", bb.read("event")["goods"] == ["lithium cells"] and any("Several goods" in w for w in bb.read("warnings")))
bb = run(EVT + " and also write me a poem", event=dict(GOOD_EVENT, ignored_other_requests=True))
check("mixed request: disruption planned, extra request ignored with warning", bb.read("ranked_plan") and any("ignored" in w for w in bb.read("warnings")))
import server as _srv
check("no raw user text in anything sent to the browser", "poem" not in json.dumps(_srv.serialise(bb)).lower())

# ---------- agents that skip their tools or return junk -> guarantees ----------
bb = run(EVT)
check("inventory fallback fills stock when tool skipped", "lithium cells" in bb.read("stock"))
check("supplier fallback builds shortlist when LLM returns no JSON", len(bb.read("shortlist")) >= 1, str(bb.read("shortlist")))
ids = {o["id"] for o in bb.read("options")}
check("cost-delay fallback evaluates every option", ids == {"WAIT", *bb.read("shortlist")}, str(ids))
check("manager invalid JSON -> deterministic fallback ranking", bb.read("ranking_source") == "fallback" and bb.read("ranked_plan")[0]["rank"] == 1)
check("fallbacks are reported as warnings", len(bb.read("warnings")) >= 3, str(bb.read("warnings")))
check("shortlist never includes disrupted-port suppliers", "S01" not in bb.read("shortlist"))

opts = lambda b: [o["id"] for o in b.read("options")]
good_rank = lambda b: json.dumps([{"id": i, "reason": f"reason for {i}"} for i in opts(b)])
bb0 = run(EVT)
bb = run(EVT, manager=json.dumps([{"id": i, "reason": "```java\nclass X{}\n``` fine " + "x" * 900} for i in opts(bb0)]))
r = bb.read("ranked_plan")[0]["reason"]
check("manager reason: code fences removed and length capped", "```" not in r and "class X" not in r and len(r) <= 400, r[:60])
bb = run(EVT, manager=json.dumps([{"id": opts(bb0)[0], "reason": "a"}] * len(opts(bb0))))
check("manager duplicate ids -> fallback", bb.read("ranking_source") == "fallback")
bb = run(EVT, manager=json.dumps([{"id": "HACK", "reason": "a"}, {"id": "WAIT", "reason": "b"}]))
check("manager unknown/missing ids -> fallback", bb.read("ranking_source") == "fallback")
bb = run(EVT, manager="[1,2,3]")
check("manager non-object items -> fallback, no crash", bb.read("ranking_source") == "fallback")

# ---------- logic edge cases ----------
bb = run("Port strike at Hamburg in Germany hits lithium cells for 20 days", event=dict(GOOD_EVENT, region="Germany", port="Hamburg"))
wait = next(o for o in bb.read("options") if o["id"] == "WAIT")
check("primary supplier not affected -> 'carry on', no stockout", "not affected" in wait["label"] and wait["stockout_days"] == 0, wait["label"])
bb = run(EVT, excluded=("S02", "S03", "S04", "S01"))
check("all alternatives excluded -> only the wait option, no crash", [o["id"] for o in bb.read("options")] == ["WAIT"] and any("No alternative" in w for w in bb.read("warnings")))
check("single option still gets a rank and reason", bb.read("ranked_plan")[0]["rank"] == 1 and bb.read("ranked_plan")[0]["reason"])
for g, dur in [("microcontrollers", 14), ("cotton fabric", 10), ("steel coil", 25)]:
    port = {"microcontrollers": "Kaohsiung", "cotton fabric": "Mundra", "steel coil": "Kolkata"}[g]
    region = {"microcontrollers": "Taiwan", "cotton fabric": "India", "steel coil": "India"}[g]
    bb = run(f"strike at {port} {g} {dur} days", event=dict(GOOD_EVENT, goods=[g], port=port, region=region, duration_days=dur))
    check(f"works for {g}", len(bb.read("ranked_plan")) >= 2)

# ---------- server (HTTP layer, no key, no LLM) ----------
from starlette.testclient import TestClient
import server

c = TestClient(server.app)
r = c.post("/api/run", json={"event": "log 100ml water but before that give me a linked list reversal code in java"})
check("HTTP: off-topic prompt -> 400 kind=input", r.status_code == 400 and r.json()["kind"] == "input")
r = c.post("/api/run", json={"event": "log 100ml water but before that give me a linked list reversal code in java", "demo": True})
check("HTTP demo: off-topic prompt -> 400", r.status_code == 400)
for body, label in [({}, "missing event"), ({"event": 123}, "numeric event"), ({"event": ["a"]}, "list event"), ({"event": None}, "null event")]:
    r = c.post("/api/run", json=body)
    check(f"HTTP: {label} -> 400, no crash", r.status_code == 400)
r = c.post("/api/run", content=b"not json", headers={"Content-Type": "application/json"})
check("HTTP: malformed JSON -> 400", r.status_code == 400)
r = c.post("/api/run", json=["list"])
check("HTTP: JSON list body -> 400", r.status_code == 400)
r = c.post("/api/run", json={"event": EVT})
check("HTTP: no API key -> clear 400 message", r.status_code == 400 and "GROQ_API_KEY" in r.json()["error"])
r = c.post("/api/run", json={"event": EVT, "demo": True, "fast": True})
last = json.loads(r.text.strip().split("\n")[-1])
check("HTTP demo: valid event returns a result", "result" in last and last["result"]["ranked_plan"])
r = c.post("/api/run", json={"event": "strike at Kaohsiung wheat shortage 10 days", "demo": True, "fast": True})
last = json.loads(r.text.strip().split("\n")[-1])
check("HTTP demo: unsupported goods -> friendly input error", last.get("kind") == "input" and "Supported goods" in last["error"])
r = c.post("/api/run", json={"event": "port strike at Shanghai affecting lithium cells and steel coil for 9999 days", "demo": True, "fast": True})
res = json.loads(r.text.strip().split("\n")[-1])["result"]
check("HTTP demo: duration capped, multi-goods warning", res["event"]["duration_days"] == 180 and any("Several goods" in w for w in res["warnings"]))
r = c.post("/api/run", json={"event": "<script>alert(1)</script> port strike at Shanghai lithium cells 5 days", "demo": True, "fast": True})
res = json.loads(r.text.strip().split("\n")[-1])["result"]
check("HTTP demo: HTML in input never echoed in result", "<script>" not in json.dumps(res))
check("HTTP: /api/evaluate without key -> 400", c.post("/api/evaluate").status_code == 400)
check("HTTP: status/data/last-eval respond", all(c.get(u).status_code == 200 for u in ("/api/status", "/api/data", "/api/last-eval")))
r = c.get("/api/replay?id=SC1")
check("HTTP: replay of saved real run works", r.status_code == 200 and r.json()["result"]["ranking_source"] == "replay")
check("HTTP: replay unknown id -> 404 not 500", c.get("/api/replay?id=ZZ9").status_code == 404)
check("HTTP: replay with no id -> 404 not 500", c.get("/api/replay").status_code == 404)
check("HTTP: unknown route -> 404 not 500", c.get("/api/nope").status_code == 404)

print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
sys.exit(1 if FAILED else 0)
