import inspect

from agents.blackboard import Blackboard
from agents.event_interpreter import EventInterpreterAgent
from agents.inventory_agent import InventoryAgent
from agents.supplier_alternatives import SupplierAlternativesAgent
from agents.tradeoff import TradeoffAgent
from agents.manager import OperationsManager


async def run_desk(event_text, excluded=(), on_step=None):
    """Blackboard loop: the manager schedules the specialists; each reads/writes shared state.
    on_step(agent_name, "start"|"done") is called around every agent (used by the web UI)."""
    bb = Blackboard()
    bb.write("input", "event_text", event_text)
    for agent in (EventInterpreterAgent(), InventoryAgent(), SupplierAlternativesAgent(excluded),
                  TradeoffAgent(), OperationsManager()):
        if on_step:
            r = on_step(agent.name, "start")
            if inspect.isawaitable(r):
                await r
        await agent.run(bb)
        if on_step:
            r = on_step(agent.name, "done")
            if inspect.isawaitable(r):
                await r
    return bb
