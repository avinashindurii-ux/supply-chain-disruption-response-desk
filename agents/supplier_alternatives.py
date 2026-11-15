"""Supplier Alternatives Agent (LLM + RAG tool): retrieves and shortlists replacement suppliers."""
from semantic_kernel.functions import kernel_function

from .llm import ask, parse_json
from .tools import CatalogueIndex, load_suppliers


class CataloguePlugin:
    def __init__(self, bb, excluded):
        self.bb = bb
        self.index = CatalogueIndex(load_suppliers(excluded))

    @kernel_function(name="search_catalogue",
                     description="RAG search over the supplier catalogue. Returns top matching suppliers with "
                                 "region, port, lead_days, unit_cost, reliability, capacity.")
    def search_catalogue(self, query: str) -> str:
        ev = self.bb.read("event")
        hits = [{"id": s["id"], "name": s["name"], "region": s["region"], "port": s["port"],
                 "goods": s["goods"], "lead_days": s["lead_days"], "unit_cost": s["unit_cost"],
                 "reliability": s["reliability"], "capacity": s["capacity"], "rag_score": round(sc, 3)}
                for sc, s in self.index.search(query, k=8)]
        # a supplier located in/shipping via the disrupted area is not an alternative
        hits = [h for h in hits if h["port"] != ev.get("port") and h["region"] != ev.get("region")]
        self.bb.state.setdefault("retrieved", {}).update({h["id"]: h for h in hits})
        return str(hits)


class SupplierAlternativesAgent:
    name = "SupplierAlternatives"

    def __init__(self, excluded=()):
        self.excluded = excluded

    async def run(self, bb):
        ev, stock = bb.read("event"), bb.read("stock")
        instr = ("You are the Supplier Alternatives Agent. Use search_catalogue (RAG) to find suppliers of the "
                 "affected good that are NOT in the disrupted region/port. You may search more than once. "
                 'Reply with ONLY JSON: {"shortlist": [supplier ids], "why": str}.')
        prompt = f"Event: {ev}\nStock: {stock}"
        plugin = CataloguePlugin(bb, self.excluded)
        try:
            out = parse_json(await ask(self.name, instr, prompt, plugins=[plugin])) or {}
        except (ValueError, TypeError):
            out = {}
        if not isinstance(out, dict):
            out = {}
        retrieved = bb.read("retrieved", {})
        good = ev["goods"][0]
        shortlist = []
        for i in out.get("shortlist", []) if isinstance(out.get("shortlist"), list) else []:
            if i in retrieved and good in retrieved[i]["goods"] and i not in shortlist:
                shortlist.append(i)
        why = str(out.get("why", ""))[:400]
        if not shortlist:                          # guarantee: never leave the plan without alternatives if any exist
            plugin.search_catalogue(f"{good} supplier")
            retrieved = bb.read("retrieved", {})
            shortlist = [i for i, h in retrieved.items() if good in h["goods"]]
            why = "The agent's shortlist was empty or invalid, so every retrieved supplier of this good was used."
            bb.state["warnings"] = bb.read("warnings", []) + ["Supplier shortlist was filled by fallback retrieval."]
        bb.write(self.name, "shortlist", shortlist)
        bb.write(self.name, "shortlist_why", why)
