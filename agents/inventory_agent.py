"""Inventory Agent (LLM + tool): reads stock levels by calling the inventory tool."""
from semantic_kernel.functions import kernel_function

from .llm import ask
from .tools import inventory_tool


class InventoryPlugin:
    def __init__(self, bb):
        self.bb = bb

    @kernel_function(name="get_stock", description="Get on-hand stock, daily demand and days of cover for a good.")
    def get_stock(self, good: str) -> str:
        rec = inventory_tool(good)
        self.bb.state.setdefault("stock", {})[good] = rec          # shared-memory write
        return str(rec)


class InventoryAgent:
    name = "Inventory"

    async def run(self, bb):
        ev = bb.read("event")
        instr = ("You are the Inventory Agent. Call get_stock once for EACH affected good, then summarise "
                 "the days of cover in one sentence.")
        note = await ask(self.name, instr, f"Affected goods: {ev['goods']}", plugins=[InventoryPlugin(bb)])
        stock = bb.read("stock", {})
        for g in ev["goods"]:                      # guarantee: if the LLM skipped the tool, read the stock ourselves
            if g not in stock:
                stock[g] = inventory_tool(g)
                bb.state["warnings"] = bb.read("warnings", []) + ["Inventory agent did not call its tool; stock was read directly."]
        bb.state["stock"] = stock
        bb.log.append((self.name, "stock"))
        note = str(note)[:500]
        bb.write(self.name, "inventory_note", note)
