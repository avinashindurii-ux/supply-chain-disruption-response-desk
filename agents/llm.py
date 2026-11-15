"""Groq (OpenAI-compatible) chat service for Semantic Kernel + a helper to run one agent turn."""
import asyncio
import json
import os
import re

import httpx
from openai import AsyncOpenAI
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# gpt-oss emits bare tool names ("get_stock"), but SK advertises "Plugin-get_stock" and Groq then rejects the
# call. Advertise bare names instead (SK resolves bare names; all our tool names are unique).
from semantic_kernel.functions.kernel_function_metadata import KernelFunctionMetadata
KernelFunctionMetadata.fully_qualified_name = property(lambda self: self.name)

MODEL =os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def make_service():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Set GROQ_API_KEY first (cmd: set GROQ_API_KEY=your_key)")
    client = AsyncOpenAI(api_key=key, base_url="https://api.groq.com/openai/v1",
                         timeout=httpx.Timeout(90.0, connect=30.0), max_retries=3)
    return OpenAIChatCompletion(ai_model_id=MODEL, async_client=client)


_services = {}


def service():
    """One service per event loop (Streamlit calls asyncio.run on every click)."""
    loop = id(asyncio.get_running_loop())
    if loop not in _services:
        _services.clear()
        _services[loop] = make_service()
    return _services[loop]


def root_cause(e):
    while e.__cause__ is not None:
        e = e.__cause__
    return f"{type(e).__name__}: {str(e)[:300]}"


async def ask(name, instructions, prompt, plugins=None, retries=5):
    """Run one Semantic Kernel ChatCompletionAgent turn (auto tool-calling), retrying transient failures."""
    for attempt in range(retries):
        try:
            agent = ChatCompletionAgent(
                service=service(), name=name, instructions=instructions, plugins=plugins or [],
                function_choice_behavior=FunctionChoiceBehavior.Auto())
            resp = await agent.get_response(messages=prompt)
            return str(resp.content)
        except Exception as e:
            cause = root_cause(e)
            if attempt == retries - 1:
                raise RuntimeError(f"{name} failed after {retries} tries: {cause}") from e
            # 429 rate limit: wait longer (Groq says how long in the message); tool_use_failed: retry quickly
            m = re.search(r"try again in ([\d.]+)(ms|s|m)", cause)
            wait = 3 + attempt * 3
            if "429" in cause or "rate" in cause.lower():
                wait = 10 + attempt * 10
                if m:
                    v = float(m.group(1)); wait = max(wait, v / 1000 if m.group(2) == "ms" else v * (60 if m.group(2) == "m" else 1) + 1)
            await asyncio.sleep(wait)


def parse_json(text):
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    return json.loads(m.group(1)) if m else None
