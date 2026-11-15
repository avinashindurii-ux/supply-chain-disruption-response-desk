import asyncio
import sys

from pipeline import run_desk


async def main():
    text = " ".join(sys.argv[1:]) or "Port strike at Shanghai port in China hits lithium cells for 21 days"
    bb = await run_desk(text)
    print("Event:", bb.read("event"))
    print("Blackboard trail:", " -> ".join(f"{a}:{k}" for a, k in bb.log))
    print(f"Ranking by: {bb.read('ranking_source')}\n")
    for o in bb.read("ranked_plan"):
        print(f"#{o['rank']} {o['label']}\n    cost={o['total_cost']:,}  delay={o['delay_days']}d\n    {o['reason']}")

asyncio.run(main())
