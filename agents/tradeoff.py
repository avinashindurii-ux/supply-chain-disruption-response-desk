"""Cost-Delay Trade-off Agent (LLM + calculator tool): computes cost and delay for each option."""
from semantic_kernel.functions import kernel_function

from .llm import ask
from .tools import compute_options


class TradeoffPlugin:
    def __init__(self, bb):
        self.bb = bb

    @kernel_function(name="evaluate_options",
                     description="Compute cost and delay for waiting it out plus reordering from each supplier id "
                                 "(comma separated, e.g. 'S02,S03').")
    def evaluate_options(self, supplier_ids: str) -> str:
        ev = self.bb.read("event")
        ids = [i.strip() for i in supplier_ids.split(",") if i.strip()]
        opts = compute_options(ev["goods"][0], ids, ev["duration_days"], event=ev)
        self.bb.state["options"] = opts                              # shared-memory write
        return str(opts)


class TradeoffAgent:
    name = "CostDelay"

    async def run(self, bb):
        ids = ",".join(bb.read("shortlist"))
        instr = ("You are the Cost-Delay Trade-off Agent. Call evaluate_options ONCE with the shortlisted supplier "
                 "ids, then summarise in two sentences which options are cheapest and which are fastest.")
        plugin = TradeoffPlugin(bb)
        note = await ask(self.name, instr, f"Shortlisted supplier ids: {ids}", plugins=[plugin])
        want = {"WAIT", *bb.read("shortlist")}
        have = {o["id"] for o in bb.read("options", [])}
        if have != want:                           # guarantee: LLM skipped the tool or passed wrong ids -> compute directly
            plugin.evaluate_options(ids)
            bb.state["warnings"] = bb.read("warnings", []) + ["Cost-Delay agent did not evaluate every option; options were computed directly."]
        if not bb.read("shortlist"):
            bb.state["warnings"] = bb.read("warnings", []) + ["No alternative suppliers exist outside the disrupted area, so only the wait option is available."]
        bb.write(self.name, "tradeoff_note", str(note)[:500])
