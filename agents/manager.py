"""Operations Manager (LLM arbiter): reads the whole blackboard, ranks options and justifies the ranking."""
import json
import re

from .llm import ask, parse_json
from .tools import baseline_rank


class OperationsManager:
    name = "OperationsManager"

    async def run(self, bb):
        opts = bb.read("options")
        baseline = baseline_rank([dict(o) for o in opts])
        bb.write(self.name, "baseline_ranking", [o["id"] for o in baseline])
        instr = ("You are the Operations Manager, the arbiter of a supply-chain disruption desk. Rank ALL options "
                 "from best to worst. Stockouts (lost sales) are very costly, so weigh stockout_days and "
                 "total_cost together with delay_days. Reply with ONLY JSON: "
                 '[{"id": str, "reason": str}] best first, every id exactly once. Each reason must quote the '
                 "option's cost and delay numbers and explain why it sits at that rank. Write 1-2 plain-English sentences "
                 "for a business reader: say 'total cost', 'delay' and 'stockout days' in words, never field names "
                 "with underscores such as total_cost or stockout_days.")
        ctx = {"event": bb.read("event"), "stock": bb.read("stock"),
               "tradeoff_note": bb.read("tradeoff_note"), "options": opts}
        by_id = {o["id"]: o for o in opts}
        try:
            ranking = parse_json(await ask(self.name, instr, json.dumps(ctx)))
            ids = [r["id"] for r in ranking]
            assert sorted(ids) == sorted(by_id), "ranking must contain every option once"
            source = "llm"
        except Exception:                                          # invalid LLM output -> deterministic fallback
            ranking = [{"id": o["id"], "reason": "fallback weighted score (LLM ranking invalid)"} for o in baseline]
            source = "fallback"
        plan = []
        for i, r in enumerate(ranking, 1):
            reason = re.sub(r"```.*?```", "", str(r.get("reason", "")), flags=re.S).strip()[:400]
            plan.append(dict(by_id[r["id"]], rank=i, reason=reason or "No reason given."))
        bb.write(self.name, "ranked_plan", plan)
        bb.write(self.name, "ranking_source", source)
