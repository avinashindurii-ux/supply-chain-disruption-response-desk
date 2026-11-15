"""Deterministic tools the LLM agents call (inventory lookup, catalogue RAG, cost/delay calculator)."""
import json
import math
import pathlib
import re
from collections import Counter

from .errors import UserInputError

DATA = pathlib.Path(__file__).parent.parent / "data"
MAX_EVENT_CHARS = 500
EVENT_TYPES = ["port strike", "weather", "shortage", "congestion", "blockade/customs", "fire/outage", "other"]
DISRUPTION_WORDS = ["strike", "shortage", "delay", "delayed", "flood", "typhoon", "storm", "hurricane", "blockade", "customs",
                    "closure", "closed", "shutdown", "congestion", "fire", "earthquake", "embargo", "sanction", "disruption",
                    "outage", "halt", "halted", "shipment", "shipments", "supplier", "port", "stock", "supply", "blocked", "stuck"]
BATCH_DAYS = 30        # replenishment batch = 30 days of demand
IN_TRANSIT_DAYS = 3    # WAIT: pipeline orders arrive shortly after the disruption clears


def load_suppliers(excluded=()):
    return [s for s in json.load(open(DATA / "suppliers.json")) if s["id"] not in excluded]


def supported_goods():
    return sorted({g for s in load_suppliers() for g in s["goods"]})


def clean_event_text(text):
    """Gate 1 (no LLM): length, emptiness, and 'does this even look like a supply-chain event?'."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or "")).strip()
    if not text:
        raise UserInputError("Please describe a disruption, for example: 'Port strike at Shanghai port in China hits lithium cells for 21 days'.")
    if len(text) > MAX_EVENT_CHARS:
        raise UserInputError(f"That message is too long ({len(text)} characters). Please keep the event under {MAX_EVENT_CHARS} characters.")
    low = text.lower()
    sups = load_suppliers()
    terms = set(supported_goods()) | {s["port"] for s in sups} | {s["region"] for s in sups}
    if not (any(t.lower() in low for t in terms) or any(re.search(rf"\b{w}\b", low) for w in DISRUPTION_WORDS)):
        raise UserInputError("This desk only plans responses to supply-chain disruptions (port strikes, shortages, floods and similar). "
                             "It cannot help with that request. Try: 'Typhoon closes Kaohsiung in Taiwan; microcontrollers delayed for 14 days'.")
    return text


def primary_affected(primary, event):
    """Is the primary supplier inside the disrupted port/region? Unknown location => assume yes."""
    if not event or (not event.get("port") and not event.get("region")):
        return True
    return primary["port"] == event.get("port") or primary["region"] == event.get("region")


def inventory_tool(good):
    rec = json.load(open(DATA / "inventory.json")).get(good)
    return dict(rec, days_of_cover=round(rec["on_hand"] / rec["daily_demand"], 1)) if rec else None


# ---------- RAG over the supplier catalogue (TF-IDF cosine) ----------
def _tok(s):
    return re.findall(r"[a-z]+", s.lower())


class CatalogueIndex:
    def __init__(self, suppliers):
        self.docs = suppliers
        self.toks = [_tok(f"{s['name']} {s['region']} {' '.join(s['goods'])} {s['profile']}") for s in suppliers]
        n = len(self.toks)
        df = Counter(t for d in self.toks for t in set(d))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vecs = [self._vec(d) for d in self.toks]

    def _vec(self, toks):
        v = {t: f * self.idf.get(t, 0) for t, f in Counter(toks).items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1
        return {t: x / norm for t, x in v.items()}

    def search(self, query, k=6):
        q = self._vec(_tok(query))
        scored = [(sum(w * v.get(t, 0) for t, w in q.items()), s) for v, s in zip(self.vecs, self.docs)]
        return sorted(scored, key=lambda x: -x[0])[:k]


# ---------- cost / delay calculator ----------
def compute_options(good, supplier_ids, duration_days, excluded=(), event=None):
    st = inventory_tool(good)
    demand, cover, margin = st["daily_demand"], st["days_of_cover"], st["unit_margin"]
    batch = demand * BATCH_DAYS
    sups = {s["id"]: s for s in load_suppliers()}
    options = []

    def build(label, sid, s, delay):
        qty = min(batch, s["capacity"])
        delay_eff = delay + (1 - s["reliability"]) * 5            # reliability-adjusted
        gap_days = max(0, delay_eff - cover)
        missing = demand * gap_days + max(0, batch - qty)         # lost + uncovered units
        stockout = missing * (margin + s["unit_cost"])
        purchase = qty * s["unit_cost"]
        options.append({"id": sid, "label": label, "qty": qty, "delay_days": round(delay_eff, 1),
                        "purchase_cost": round(purchase), "stockout_cost": round(stockout),
                        "total_cost": round(purchase + stockout), "stockout_days": round(gap_days, 1)})

    prim = sups[st["primary_supplier"]]
    if primary_affected(prim, event):
        build(f"WAIT out disruption ({prim['name']})", "WAIT", prim, duration_days + IN_TRANSIT_DAYS)
    else:                                   # disruption elsewhere: the usual supplier keeps delivering
        build(f"Carry on with {prim['name']} (not affected)", "WAIT", prim, IN_TRANSIT_DAYS)
    for sid in supplier_ids:
        if sid in sups and sid not in excluded:
            s = sups[sid]
            build(f"Reorder from {s['name']} ({s['region']})", sid, s, s["lead_days"])
    return options


def baseline_rank(options, w_cost=0.6, w_delay=0.4):
    """Weighted cost/delay score; used as the manager's fallback and as a sanity baseline."""
    cmax = max(o["total_cost"] for o in options) or 1
    dmax = max(o["delay_days"] for o in options) or 1
    for o in options:
        o["score"] = round(w_cost * o["total_cost"] / cmax + w_delay * o["delay_days"] / dmax, 4)
    return sorted(options, key=lambda o: o["score"])
