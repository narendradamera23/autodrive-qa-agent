import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import re
from agent.tools import rag_search, fault_code_decoder, general_automotive_answer

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ── Tool Router ───────────────────────────────────────────────────────────
def route_query(query: str) -> str:
    """Use LLM to decide which tool to use, then call it."""

    routing_prompt = f"""You are a routing assistant. Given a user question, decide which tool to use.

Tools available:
1. fault_code - Use when the question contains an OBD-II fault code like P0420, P0300, C0001, B0001, U0001 etc.
2. rag_search - Use when the question is about a specific vehicle feature, maintenance procedure, warning light, or operating instruction from a car manual.
3. general - Use for all other general automotive questions.

Respond with ONLY one word: fault_code, rag_search, or general.

Question: {query}
Tool:"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "llama3.2",
            "prompt": routing_prompt,
            "stream": False
        }
    )
    tool_choice = response.json()["response"].strip().lower()

    if "fault_code" in tool_choice:
        tool_choice = "fault_code"
    elif "rag" in tool_choice:
        tool_choice = "rag_search"
    else:
        tool_choice = "general"

    return tool_choice


# ── Agent ─────────────────────────────────────────────────────────────────
class AutoDriveAgent:
    def __init__(self):
        self.chat_history = []

    def invoke(self, query: str) -> str:
        history_context = ""
        if self.chat_history:
            history_context = "Previous conversation:\n"
            for turn in self.chat_history[-3:]:
                history_context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
            history_context += "\n"

        full_query = history_context + query

        tool_choice = route_query(query)
        print(f"[Router] Selected tool: {tool_choice}")

        if tool_choice == "fault_code":
            codes = re.findall(r'[PBCU][0-9]{4}', query.upper())
            if codes:
                raw_result = fault_code_decoder(codes[0])
                explain_prompt = f"""Given this fault code data:
{raw_result}

Explain this to a car owner in a clear, helpful way. Include what it means, 
how serious it is, and what they should do next.

Answer:"""
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": "llama3.2",
                        "prompt": explain_prompt,
                        "stream": False
                    }
                )
                result = response.json()["response"]
            else:
                result = fault_code_decoder(query)

        elif tool_choice == "rag_search":
            result = rag_search(full_query)

        else:
            result = general_automotive_answer(full_query)

        self.chat_history.append({
            "user": query,
            "assistant": result
        })

        return result


def get_agent():
    return AutoDriveAgent()