import asyncio
import json
import time

from pipeline import run_desk

CONCURRENCY = 2          # scenarios run in parallel; lower this if Groq returns rate-limit errors


async def main():
    sc = json.load(open("data/scenarios.json"))
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(s):
        async with sem:
            return await run_desk(s["event"])

    t0 = time.time()
    results = await asyncio.gather(*(one(s) for s in sc), return_exceptions=True)
    hit = base_hit = 0
    print(f"{'ID':4} {'expected':9} {'LLM top':8} {'rules top':9} result")
    for s, bb in zip(sc, results):
        if isinstance(bb, Exception):
            print(f"{s['id']:4} {s['expected_best']:9} ERROR: {str(bb)[:200]}")
            continue
        top = bb.read("ranked_plan")[0]["id"]
        base = bb.read("baseline_ranking")[0]
        hit += top == s["expected_best"]
        base_hit += base == s["expected_best"]
        print(f"{s['id']:4} {s['expected_best']:9} {top:8} {base:9} "
              f"{'OK' if top == s['expected_best'] else 'MISS'} ({bb.read('ranking_source')})")
    print(f"\nLLM manager top-1: {hit}/{len(sc)} = {hit/len(sc):.0%}")
    print(f"Weighted-score baseline top-1: {base_hit}/{len(sc)} = {base_hit/len(sc):.0%}")
    print(f"Elapsed: {time.time()-t0:.0f}s")

asyncio.run(main())
