from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from typing import Annotated
import operator

# ── 1. STATE ──────────────────────────────────────────────────────────────
class AgentState(BaseModel):
    messages: Annotated[list, operator.add] = []
    escalate: bool = False

# ── 2. KNOWLEDGE BASE ─────────────────────────────────────────────────────
with open("knowledge.txt", "r") as f:
    KNOWLEDGE = f.read()

# ── 3. LLM ────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile")

# ── 4. NODES ──────────────────────────────────────────────────────────────
def support_node(state: AgentState) -> dict:
    system = f"""You are a helpful customer support agent for TaskFlow.
Answer questions using ONLY the knowledge base below.
If the answer is not in the knowledge base, say you don't know and offer to escalate.
If the user seems very angry, mentions lawyers, data loss, or refund disputes over $100 — set escalation needed.

KNOWLEDGE BASE:
{KNOWLEDGE}"""

    response = llm.invoke(
        [{"role": "system", "content": system}] +
        [{"role": "human" if isinstance(m, HumanMessage) else "assistant", 
          "content": m.content} for m in state.messages]
    )
    return {"messages": [AIMessage(content=response.content)]}


def escalation_check(state: AgentState) -> str:
    last_human = ""
    for m in reversed(state.messages):
        if isinstance(m, HumanMessage):
            last_human = m.content.lower()
            break

    triggers = ["lawyer", "lost all data", "data breach", "refund refused", "furious", "sue"]
    if any(t in last_human for t in triggers):
        return "escalate"
    return "continue"


def escalate_node(state: AgentState) -> dict:
    msg = "I'm escalating this to a senior support agent right away. Someone will contact you within 2 hours."
    return {"messages": [AIMessage(content=msg)], "escalate": True}

# ── 5. GRAPH ──────────────────────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node("support", support_node)
graph.add_node("escalate", escalate_node)

graph.set_entry_point("support")

graph.add_conditional_edges(
    "support",
    escalation_check,
    {"escalate": "escalate", "continue": END}
)

graph.add_edge("escalate", END)

agent = graph.compile()
