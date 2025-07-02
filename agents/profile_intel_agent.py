from __future__ import annotations

import json
from typing import Dict, Any

from src.vertex_llm import deepseek_chat
from .tools import TOOL_MAP

SYSTEM_PROMPT = """You are \"ProfileIntel-LLM\", an expert code analyst helping recruiters understand a candidate's GitHub profile.
You can call MCP tools by returning a SINGLE JSON object—nothing else.

Allowed tools
 1. search_code
    { \"tool\": \"search_code\", \"args\": { \"pattern\": \"<string>\", \"file_pattern\": \"*.ts\" } }
 2. get_file_summary
    { \"tool\": \"get_file_summary\", \"args\": { \"file_path\": \"<string>\" } }
 3. none
    { \"tool\": \"none\", \"answer\": \"<text>\" }

ReACT rules
• Think silently, but only output the final JSON for each step.
• After a tool Observation is provided, continue reasoning if needed.
"""


def run_profile_intel(question: str, job_desc: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for _ in range(4):  # max 4 tool calls
        response = deepseek_chat(json.dumps(messages)) if False else deepseek_chat(
            "".join([f"{m['role'].capitalize()}: {m['content']}\n" for m in messages]),
            temperature=0.2,
            max_tokens=256,
        )
        try:
            tool_call = json.loads(response)
        except Exception:
            return response  # not JSON tool call
        tool_name = tool_call.get("tool")
        if tool_name == "none":
            return tool_call.get("answer", "")
        func = TOOL_MAP.get(tool_name)
        if not func:
            return f"Unknown tool {tool_name}"
        observation = func(**tool_call.get("args", {}))
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Observation: {observation}"})
    return "Max steps exceeded" 