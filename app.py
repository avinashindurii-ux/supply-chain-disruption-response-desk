"""Streamlit dashboard:  set GROQ_API_KEY in your cmd, then  streamlit run app.py"""
import asyncio
import json
import os
import pathlib
import time

import altair as alt
import pandas as pd
import streamlit as st

from agents.tools import baseline_rank, compute_options, inventory_tool, load_suppliers

ROOT = pathlib.Path(__file__).parent
EVAL_FILE = ROOT / "data" / "last_eval.json"
NAVY, BLUE, GREEN, RED, GREY = "#12315c", "#1f6feb", "#1a9a4a", "#c0392b", "#9aa7b8"

st.set_page_config(page_title="Disruption Response Desk", page_icon="🚢", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#12315c,#1f6feb); color:#fff; padding:24px 30px; border-radius:16px; margin-bottom:14px;}
.hero h1 {color:#fff; margin:0; font-size:1.8rem;} .hero p {color:#d6e4ff; margin:6px 0 0 0;}
.pill {display:inline-block; background:#eaf1ff; color:#1f4fa8; border-radius:999px; padding:3px 12px; font-size:.8rem; margin:0 6px 4px 0;}
.rec {background:#e8f7ee; border:1px solid #b5e3c5; border-left:6px solid #1a9a4a; border-radius:12px; padding:16px 20px; margin:10px 0 16px 0;}
.rec h3 {margin:0 0 4px 0; color:#12622f;}
.card {background:#fff; border:1px solid #e1e6ee; border-radius:14px; padding:14px 20px; margin-bottom:10px;}
.card.top {border:2px solid #1a9a4a;}
.badge {display:inline-block; width:32px; height:32px; line-height:32px; text-align:center; border-radius:50%; background:#1f6feb; color:#fff; font-weight:700; margin-right:10px;}
.card.top .badge {background:#1a9a4a;}
.title {font-size:1.05rem; font-weight:600;}
.stat {display:inline-block; min-width:140px; margin:8px 16px 2px 0;}
.stat .k {font-size:.7rem; text-transform:uppercase; color:#6b7686; letter-spacing:.04em;}
.stat .v {font-size:1.2rem; font-weight:600;} .stat .v.bad {color:#c0392b;} .stat .v.good {color:#1a9a4a;}
.reason {color:#3a4656; margin-top:6px; line-height:1.5;}
.step {display:inline-block; background:#12315c; color:#fff; border-radius:8px; padding:6px 12px; margin:2px 4px 2px 0; font-size:.85rem;}
.step.w {background:#1a9a4a;}
.kpi {background:#fff; border:1px solid #e1e6ee; border-radius:14px; padding:14px 18px;}
.kpi .k {font-size:.72rem; text-transform:uppercase; color:#6b7686; letter-spacing:.04em;} .kpi .v {font-size:1.7rem; font-weight:700; color:#12315c;}
</style>
<div class="hero"><h1>🚢 Supply Chain Disruption Response Desk</h1>
<p>Multi-agent system · Semantic Kernel agents on Groq · blackboard with manager arbiter · RAG over the supplier catalogue</p></div>
""", unsafe_allow_html=True)

HAS_KEY = bool(os.environ.get("GROQ_API_KEY"))
SUPPLIERS = load_suppliers()
SCENARIOS = json.load(open(ROOT / "data" / "scenarios.json"))
NAME = {s["id"]: s["name"] for s in SUPPLIERS}
NAME["WAIT"] = "Wait it out"


def kpi(col, label, value):
    col.markdown(f'<div class="kpi"><div class="k">{label}</div><div class="v">{value}</div></div>', unsafe_allow_html=True)


def rank_with(options, w_cost):
    return baseline_rank([dict(o) for o in options], w_cost=w_cost, w_delay=1 - w_cost)


def pareto(opts):
    """Options not beaten on both cost and delay by another option."""
    out = []
    for o in opts:
        if not any(p["total_cost"] <= o["total_cost"] and p["delay_days"] <= o["delay_days"] and
                   (p["total_cost"] < o["total_cost"] or p["delay_days"] < o["delay_days"]) for p in opts):
            out.append(o)
    return sorted(out, key=lambda o: o["delay_days"])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    if HAS_KEY:
        st.success("Groq key found in environment")
    else:
        st.error("GROQ_API_KEY not set. Agent runs are disabled; the data and analysis tabs still work.")
    st.text_input("Model", value=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"), key="model")
    os.environ["GROQ_MODEL"] = st.session_state["model"]
    st.divider()
    st.subheader("Ranking weights")
    st.caption("Used by the Analysis and What-if tabs (the manager LLM ranks on its own judgement).")
    w_cost = st.slider("Weight on cost (rest on delay)", 0.0, 1.0, 0.6, 0.05)

tab_run, tab_an, tab_wi, tab_ev, tab_data, tab_bb = st.tabs(
    ["▶ Run desk", "📊 Analysis", "🔁 What-if replay", "✅ Evaluation", "📦 Catalogue & stock", "🧠 Blackboard"])

# ---------------------------------------------------------------- RUN
with tab_run:
    choice = st.selectbox("Scenario", ["Custom"] + [f"{s['id']}: {s['event']}" for s in SCENARIOS])
    default = "Port strike at Shanghai port in China hits lithium cells for 21 days" if choice == "Custom" \
        else choice.split(": ", 1)[1]
    event = st.text_area("Disruption event", value=default, height=80, key=f"ev_{choice}")
    excluded = st.multiselect("Exclude suppliers before the agents run (optional)", [s["id"] for s in SUPPLIERS],
                              format_func=lambda i: f"{i} · {NAME[i]}")
    if st.button("▶  Run response desk", type="primary", disabled=not HAS_KEY):
        from pipeline import run_desk
        t0 = time.time()
        with st.spinner("Agents are working on the blackboard..."):
            try:
                st.session_state["bb"] = asyncio.run(run_desk(event, excluded=tuple(excluded)))
                st.session_state["secs"] = time.time() - t0
                st.session_state["excluded"] = list(excluded)
            except Exception as e:
                st.error(f"Agent run failed: {str(e)[:500]}")

    bb = st.session_state.get("bb")
    if not bb:
        st.info("Run a scenario to see the ranked plan. The other tabs use the latest run.")
    else:
        ev, plan = bb.read("event"), bb.read("ranked_plan")
        best = plan[0]
        st.markdown(
            f'<span class="pill">Type: {ev["type"]}</span><span class="pill">Region: {ev["region"]}</span>'
            f'<span class="pill">Port: {ev["port"]}</span><span class="pill">Goods: {", ".join(ev["goods"])}</span>'
            f'<span class="pill">Duration: {ev["duration_days"]} days</span>'
            f'<span class="pill">Run time: {st.session_state.get("secs", 0):.0f}s</span>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="rec"><h3>Recommended: {best["label"]}</h3>Total cost <b>{best["total_cost"]:,}</b> · '
            f'delay <b>{best["delay_days"]} days</b> · '
            f'{"no stockout" if best["stockout_days"] == 0 else str(best["stockout_days"]) + " days of stockout"}</div>',
            unsafe_allow_html=True)
        if bb.read("ranking_source") == "fallback":
            st.warning("The manager's LLM ranking was invalid, so the weighted-score fallback was used.")
        st.markdown("**Agent trace**")
        st.markdown("".join(f'<span class="step {"w" if k == "ranked_plan" else ""}">{a} → {k}</span>' for a, k in bb.log),
                    unsafe_allow_html=True)
        st.subheader("Ranked action options")
        for o in plan:
            cls = "good" if o["stockout_days"] == 0 else "bad"
            txt = "None" if o["stockout_days"] == 0 else f'{o["stockout_days"]} days'
            st.markdown(
                f'<div class="card {"top" if o["rank"] == 1 else ""}"><span class="badge">{o["rank"]}</span>'
                f'<span class="title">{o["label"]}</span><div>'
                f'<span class="stat"><div class="k">Total cost</div><div class="v">{o["total_cost"]:,}</div></span>'
                f'<span class="stat"><div class="k">Delay</div><div class="v">{o["delay_days"]} days</div></span>'
                f'<span class="stat"><div class="k">Stockout</div><div class="v {cls}">{txt}</div></span>'
                f'<span class="stat"><div class="k">Order qty</div><div class="v">{o["qty"]:,}</div></span></div>'
                f'<div class="reason"><b>Why this rank:</b> {o["reason"]}</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- ANALYSIS
with tab_an:
    bb = st.session_state.get("bb")
    if not bb:
        st.info("Run a scenario first.")
    else:
        plan, ev, st_ = bb.read("ranked_plan"), bb.read("event"), bb.read("stock")[bb.read("event")["goods"][0]]
        opts = bb.read("options")
        df = pd.DataFrame(plan)
        df["Option"] = df["id"].map(lambda i: f"{i} · {NAME.get(i, i)}")
        cheapest, fastest = min(plan, key=lambda o: o["total_cost"]), min(plan, key=lambda o: o["delay_days"])
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "Days of stock cover", st_["days_of_cover"])
        kpi(k2, "Disruption length", f'{ev["duration_days"]} days')
        kpi(k3, "Cheapest option", f'{NAME.get(cheapest["id"], cheapest["id"])}')
        kpi(k4, "Fastest option", f'{NAME.get(fastest["id"], fastest["id"])}')
        st.write("")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cost breakdown: purchase vs stockout loss**")
            long = df.melt(id_vars=["Option", "rank"], value_vars=["purchase_cost", "stockout_cost"],
                           var_name="Part", value_name="Cost")
            long["Part"] = long["Part"].map({"purchase_cost": "Purchase", "stockout_cost": "Stockout loss"})
            st.altair_chart(alt.Chart(long).mark_bar().encode(
                y=alt.Y("Option:N", sort=alt.EncodingSortField("rank"), title=None),
                x=alt.X("Cost:Q", title="Cost"),
                color=alt.Color("Part:N", scale=alt.Scale(domain=["Purchase", "Stockout loss"], range=[BLUE, RED])),
                tooltip=["Option", "Part", alt.Tooltip("Cost:Q", format=",")]).properties(height=280),
                width="stretch")
        with c2:
            st.markdown("**Delay vs days of stock cover**")
            d2 = df[["Option", "rank", "delay_days"]].copy()
            bars = alt.Chart(d2).mark_bar(color=BLUE).encode(
                y=alt.Y("Option:N", sort=alt.EncodingSortField("rank"), title=None),
                x=alt.X("delay_days:Q", title="Delay (days)"), tooltip=["Option", "delay_days"])
            rule = alt.Chart(pd.DataFrame({"c": [st_["days_of_cover"]]})).mark_rule(color=RED, strokeDash=[6, 4], size=2).encode(x="c:Q")
            st.altair_chart((bars + rule).properties(height=280), width="stretch")
            st.caption("Red line = days of stock on hand. Bars beyond it mean a stockout.")

        st.markdown("**Cost vs delay with the Pareto frontier**")
        front = pd.DataFrame(pareto(opts))
        pts = alt.Chart(df).mark_circle(size=220).encode(
            x=alt.X("delay_days:Q", title="Delay (days)"), y=alt.Y("total_cost:Q", title="Total cost"),
            color=alt.Color("rank:O", scale=alt.Scale(scheme="blues", reverse=True), title="Manager rank"),
            tooltip=["Option", alt.Tooltip("total_cost:Q", format=","), "delay_days", "stockout_days", "rank"])
        lbl = alt.Chart(df).mark_text(dy=-16, fontSize=12).encode(x="delay_days:Q", y="total_cost:Q", text="id:N")
        line = alt.Chart(front).mark_line(color=GREEN, strokeDash=[4, 3]).encode(x="delay_days:Q", y="total_cost:Q")
        st.altair_chart((line + pts + lbl).properties(height=340), width="stretch")
        st.caption("Green dashed line joins the options that no other option beats on both cost and delay.")

        st.markdown("**Weight sensitivity: which option wins as the cost weight changes?**")
        sweep = []
        for w in [i / 20 for i in range(21)]:
            win = rank_with(opts, w)[0]["id"]
            sweep.append({"Cost weight": w, "Winner": f"{win} · {NAME.get(win, win)}"})
        sw = pd.DataFrame(sweep)
        st.altair_chart(alt.Chart(sw).mark_rect(height=40).encode(
            x=alt.X("Cost weight:Q", bin=alt.Bin(step=0.05), title="Weight on cost  (0 = only delay, 1 = only cost)"),
            color=alt.Color("Winner:N", legend=alt.Legend(title="Top-ranked option")),
            tooltip=["Cost weight", "Winner"]).properties(height=90), width="stretch")
        flips = sw["Winner"].ne(sw["Winner"].shift()).sum() - 1
        stable = flips == 0
        st.info(f"The top option {'never changes' if stable else f'changes {flips} time(s)'} across the full range of weights. "
                + ("The recommendation is robust to how the business trades cost against speed." if stable else
                   "The decision depends on how much the business values speed over cost; check the slider in the sidebar."))

        st.markdown(f"**Ranking at cost weight {w_cost:.2f} (weighted score, no LLM)**")
        rk = pd.DataFrame(rank_with(opts, w_cost))
        rk["Option"] = rk["id"].map(lambda i: f"{i} · {NAME.get(i, i)}")
        rk.insert(0, "Rank", range(1, len(rk) + 1))
        st.dataframe(rk[["Rank", "Option", "total_cost", "delay_days", "stockout_days", "score"]].rename(
            columns={"total_cost": "Total cost", "delay_days": "Delay (d)", "stockout_days": "Stockout (d)", "score": "Score"}),
            hide_index=True, width="stretch")
        agree = rk.iloc[0]["id"] == plan[0]["id"]
        st.caption("LLM manager and weighted score agree on the top option." if agree else
                   f"LLM manager picked {plan[0]['id']} but the weighted score picks {rk.iloc[0]['id']}: the LLM weighed stockout risk differently.")

# ---------------------------------------------------------------- WHAT-IF
with tab_wi:
    bb = st.session_state.get("bb")
    if not bb:
        st.info("Run a scenario first.")
    else:
        ev, short = bb.read("event"), bb.read("shortlist")
        good = ev["goods"][0]
        st.write("Remove one or more shortlisted suppliers and replay the plan (cost and delay tools, no LLM call).")
        top = bb.read("ranked_plan")[0]["id"]
        gone = st.multiselect("Suppliers to remove", short, default=[top] if top in short else [],
                              format_func=lambda i: f"{i} · {NAME[i]}")
        before = rank_with(compute_options(good, short, ev["duration_days"]), w_cost)
        after = rank_with(compute_options(good, [i for i in short if i not in gone], ev["duration_days"]), w_cost)
        b0, a0 = before[0], after[0]
        c1, c2, c3 = st.columns(3)
        kpi(c1, "Best before", f'{b0["id"]} · {NAME.get(b0["id"], b0["id"])}')
        kpi(c2, "Best after", f'{a0["id"]} · {NAME.get(a0["id"], a0["id"])}')
        kpi(c3, "Change in cost / delay", f'{a0["total_cost"] - b0["total_cost"]:+,} / {a0["delay_days"] - b0["delay_days"]:+.1f}d')
        st.write("")
        l, r = st.columns(2)
        for col, name, data in ((l, "Before", before), (r, "After", after)):
            with col:
                st.markdown(f"**{name}**")
                d = pd.DataFrame(data)
                d["Option"] = d["id"].map(lambda i: f"{i} · {NAME.get(i, i)}")
                d.insert(0, "Rank", range(1, len(d) + 1))
                st.dataframe(d[["Rank", "Option", "total_cost", "delay_days", "stockout_days"]].rename(
                    columns={"total_cost": "Total cost", "delay_days": "Delay (d)", "stockout_days": "Stockout (d)"}),
                    hide_index=True, width="stretch")
        cmp_rows = []
        for name, data in (("Before", before), ("After", after)):
            for o in data:
                cmp_rows.append({"Plan": name, "Option": f"{o['id']} · {NAME.get(o['id'], o['id'])}", "Total cost": o["total_cost"]})
        st.altair_chart(alt.Chart(pd.DataFrame(cmp_rows)).mark_bar().encode(
            x=alt.X("Option:N", title=None), y=alt.Y("Total cost:Q"), xOffset="Plan:N",
            color=alt.Color("Plan:N", scale=alt.Scale(domain=["Before", "After"], range=[GREY, BLUE]))).properties(height=300),
            width="stretch")
        if gone:
            st.success(f"Removing {', '.join(gone)} changes the best plan from **{b0['label']}** to **{a0['label']}** "
                       f"({a0['total_cost'] - b0['total_cost']:+,} cost, {a0['delay_days'] - b0['delay_days']:+.1f} days).")

# ---------------------------------------------------------------- EVALUATION
with tab_ev:
    st.write("Run all 8 scenarios and compare the top-ranked option with the known best.")
    results = st.session_state.get("eval")
    if results is None and EVAL_FILE.exists():
        results = json.load(open(EVAL_FILE))
        st.caption("Showing the last saved evaluation.")
    if st.button("▶ Run evaluation (8 scenarios)", disabled=not HAS_KEY):
        from pipeline import run_desk

        async def run_all():
            sem, out, prog = asyncio.Semaphore(2), [], st.progress(0.0, "Running scenarios...")

            async def one(s):
                async with sem:
                    t0 = time.time()
                    try:
                        bb = await run_desk(s["event"])
                        plan = bb.read("ranked_plan")
                        rec = {"id": s["id"], "event": s["event"], "expected": s["expected_best"], "llm_top": plan[0]["id"],
                               "rules_top": bb.read("baseline_ranking")[0], "source": bb.read("ranking_source"),
                               "plan": [{k: o[k] for k in ("rank", "id", "label", "total_cost", "delay_days", "stockout_days", "reason")} for o in plan]}
                    except Exception as e:
                        rec = {"id": s["id"], "event": s["event"], "expected": s["expected_best"], "error": str(e)[:200]}
                    rec["secs"] = round(time.time() - t0, 1)
                    out.append(rec)
                    prog.progress(len(out) / len(SCENARIOS), f"{len(out)}/{len(SCENARIOS)} done")
                    return rec
            res = await asyncio.gather(*(one(s) for s in SCENARIOS))
            prog.empty()
            return res
        results = asyncio.run(run_all())
        st.session_state["eval"] = results
        json.dump(results, open(EVAL_FILE, "w"), indent=1)

    if results:
        ok = [r for r in results if "error" not in r]
        llm_hit = sum(r["llm_top"] == r["expected"] for r in ok)
        rule_hit = sum(r["rules_top"] == r["expected"] for r in ok)
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "LLM manager top-1", f"{llm_hit}/{len(results)} ({llm_hit / len(results):.0%})")
        kpi(c2, "Weighted-score baseline", f"{rule_hit}/{len(results)} ({rule_hit / len(results):.0%})")
        kpi(c3, "Errored runs", len(results) - len(ok))
        kpi(c4, "Avg time / scenario", f'{sum(r["secs"] for r in results) / len(results):.0f}s')
        st.write("")
        rows = []
        for r in results:
            if "error" in r:
                rows.append({"ID": r["id"], "Event": r["event"], "Expected": r["expected"], "LLM top": "ERROR",
                             "Rules top": "-", "Result": "⚠ error", "Seconds": r["secs"]})
            else:
                rows.append({"ID": r["id"], "Event": r["event"], "Expected": r["expected"], "LLM top": r["llm_top"],
                             "Rules top": r["rules_top"], "Result": "✅ match" if r["llm_top"] == r["expected"] else "❌ miss",
                             "Seconds": r["secs"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        chart = pd.DataFrame({"Method": ["LLM manager", "Weighted-score baseline"], "Accuracy": [llm_hit / len(results), rule_hit / len(results)]})
        st.altair_chart(alt.Chart(chart).mark_bar().encode(
            x=alt.X("Method:N", title=None), y=alt.Y("Accuracy:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Method:N", legend=None, scale=alt.Scale(range=[BLUE, GREY]))).properties(height=240, width=420))
        for r in results:
            with st.expander(f"{r['id']} · {r['event']}"):
                if "error" in r:
                    st.error(r["error"])
                else:
                    st.write(f"Expected **{r['expected']}**, LLM chose **{r['llm_top']}**, ranking source: {r['source']}")
                    st.dataframe(pd.DataFrame(r["plan"]), hide_index=True, width="stretch")
    st.caption("Known-best labels were set by the team: the cheapest option with no stockout, or the least stockout if none avoids it.")

# ---------------------------------------------------------------- CATALOGUE
with tab_data:
    st.subheader("Supplier catalogue")
    sdf = pd.DataFrame(SUPPLIERS)
    sdf["goods"] = sdf["goods"].map(", ".join)
    st.dataframe(sdf.drop(columns=["profile"]), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Lead time vs unit cost (size = capacity)**")
        st.altair_chart(alt.Chart(sdf).mark_circle(opacity=.8).encode(
            x=alt.X("lead_days:Q", title="Lead time (days)"), y=alt.Y("unit_cost:Q", title="Unit cost", scale=alt.Scale(type="log")),
            size=alt.Size("capacity:Q", legend=None), color=alt.Color("goods:N", title="Goods"),
            tooltip=["name", "region", "port", "goods", "lead_days", "unit_cost", "reliability", "capacity"]).properties(height=320),
            width="stretch")
    with c2:
        st.markdown("**Days of stock cover by good**")
        inv = [{"Good": g, "Days of cover": inventory_tool(g)["days_of_cover"], "On hand": inventory_tool(g)["on_hand"],
                "Daily demand": inventory_tool(g)["daily_demand"]} for g in sorted({g for s in SUPPLIERS for g in s["goods"]})]
        idf = pd.DataFrame(inv)
        st.altair_chart(alt.Chart(idf).mark_bar(color=BLUE).encode(
            x=alt.X("Good:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Days of cover:Q"),
            tooltip=["Good", "Days of cover", "On hand", "Daily demand"]).properties(height=320), width="stretch")
    st.dataframe(idf, hide_index=True, width="stretch")

# ---------------------------------------------------------------- BLACKBOARD
with tab_bb:
    bb = st.session_state.get("bb")
    if not bb:
        st.info("Run a scenario first.")
    else:
        st.markdown("**Write trail**")
        st.markdown("".join(f'<span class="step">{i + 1}. {a} → {k}</span>' for i, (a, k) in enumerate(bb.log)), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Structured event** (Event Interpreter)")
            st.json(bb.read("event"))
            st.markdown("**Stock** (Inventory tool)")
            st.json(bb.read("stock"))
            st.markdown("**Inventory agent note**")
            st.write(bb.read("inventory_note"))
        with c2:
            st.markdown("**Shortlist** (Supplier Alternatives)")
            st.write(bb.read("shortlist"), "—", bb.read("shortlist_why"))
            st.markdown("**Retrieved from catalogue (RAG)**")
            st.dataframe(pd.DataFrame(bb.read("retrieved", {}).values()).drop(columns=["goods"], errors="ignore"),
                         hide_index=True, width="stretch")
            st.markdown("**Trade-off agent note**")
            st.write(bb.read("tradeoff_note"))
        with st.expander("Options computed by the cost-delay tool"):
            st.dataframe(pd.DataFrame(bb.read("options")), hide_index=True, width="stretch")
