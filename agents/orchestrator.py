"""LangGraph orchestrator — classifies user intent and routes to the correct agent."""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain.schema import HumanMessage, SystemMessage
import json
from dotenv import load_dotenv

load_dotenv()

from agents.specialist_agents import (
    qa_agent,
    bug_agent,
    diagram_agent,
    docs_agent,
    impact_agent,
    onboarding_agent
)

router_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)

INTENT_OPTIONS = ["qa", "bug", "diagram", "docs", "impact", "onboarding"]


#State schema
class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    diagram_type: Optional[str]
    chunks: List[Dict]
    file_tree: List[str]
    result: Optional[str]
    diagram_result: Optional[Dict]
    error: Optional[str]


#Router node
def router_node(state: AgentState) -> AgentState:
    query = state["query"]
    messages = [
        SystemMessage(content=f"""You are an intent classifier for a codebase assistant.
Classify the user's query into exactly ONE of these intents:
- qa: general question about the code
- bug: debugging, error, or root-cause analysis
- diagram: wants a visual diagram (architecture, ER, DFD, class diagram)
- docs: wants documentation generated
- impact: impact or risk analysis of a change
- onboarding: wants to understand/overview the whole repo

Also if intent is 'diagram', classify diagram_type as one of: architecture, er, dfd, class

Respond ONLY with valid JSON: {{"intent": "...", "diagram_type": "..."}}"""),
        HumanMessage(content=query)
    ]
    response = router_llm.invoke(messages).content.strip()
    try:
        parsed = json.loads(response)
        intent = parsed.get("intent", "qa")
        diagram_type = parsed.get("diagram_type", "architecture")
    except Exception:
        intent = "qa"
        diagram_type = "architecture"

    return {**state, "intent": intent, "diagram_type": diagram_type}


#Agent nodes
def qa_node(state: AgentState) -> AgentState:
    try:
        result = qa_agent(state["query"], state["chunks"])
        return {**state, "result": result}
    except Exception as e:
        return {**state, "error": str(e)}


def bug_node(state: AgentState) -> AgentState:
    try:
        result = bug_agent(state["query"], state["chunks"])
        return {**state, "result": result}
    except Exception as e:
        return {**state, "error": str(e)}


def diagram_node(state: AgentState) -> AgentState:
    try:
        result = diagram_agent(
            state["query"],
            state["chunks"],
            state.get("diagram_type", "architecture")
        )
        return {**state, "diagram_result": result, "result": result["mermaid_code"]}
    except Exception as e:
        return {**state, "error": str(e)}


def docs_node(state: AgentState) -> AgentState:
    try:
        result = docs_agent(state["query"], state["chunks"])
        return {**state, "result": result}
    except Exception as e:
        return {**state, "error": str(e)}


def impact_node(state: AgentState) -> AgentState:
    try:
        result = impact_agent(state["query"], state["chunks"])
        return {**state, "result": result}
    except Exception as e:
        return {**state, "error": str(e)}


def onboarding_node(state: AgentState) -> AgentState:
    try:
        result = onboarding_agent(state["chunks"], state.get("file_tree", []))
        return {**state, "result": result}
    except Exception as e:
        return {**state, "error": str(e)}


#Routing function
def route_intent(state: AgentState) -> str:
    return state.get("intent", "qa")


#Build the graph
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("qa", qa_node)
    graph.add_node("bug", bug_node)
    graph.add_node("diagram", diagram_node)
    graph.add_node("docs", docs_node)
    graph.add_node("impact", impact_node)
    graph.add_node("onboarding", onboarding_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_intent,
        {
            "qa": "qa",
            "bug": "bug",
            "diagram": "diagram",
            "docs": "docs",
            "impact": "impact",
            "onboarding": "onboarding"
        }
    )

    for node in ["qa", "bug", "diagram", "docs", "impact", "onboarding"]:
        graph.add_edge(node, END)

    return graph.compile()


#Main entry point
def run_agent(query: str, chunks: List[Dict], file_tree: List[str] = None) -> Dict:
    """Run the full multi-agent pipeline for a given query."""
    app = build_graph()
    state = AgentState(
        query=query,
        intent=None,
        diagram_type=None,
        chunks=chunks,
        file_tree=file_tree or [],
        result=None,
        diagram_result=None,
        error=None
    )
    final_state = app.invoke(state)
    return {
        "intent": final_state.get("intent"),
        "result": final_state.get("result"),
        "diagram_result": final_state.get("diagram_result"),
        "error": final_state.get("error")
    }
