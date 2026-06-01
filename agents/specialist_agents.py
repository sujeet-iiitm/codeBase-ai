"""Specialist agents using Google Gemini"""

import os
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

def _get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )

def _format_context(chunks: List[Dict]) -> str:
    parts = []
    for c in chunks:
        path = c["metadata"].get("file_path", "unknown")
        parts.append(f"### File: {path}\n```\n{c['text']}\n```")
    return "\n\n".join(parts)


def qa_agent(query: str, chunks: List[Dict]) -> str:
    llm = _get_llm()
    context = _format_context(chunks)
    messages = [
        SystemMessage(content="""You are a senior software engineer assistant.
Answer the user's question using ONLY the provided codebase context.
Always cite the file path when referring to specific code.
If the answer isn't in the context, say so clearly — do not hallucinate."""),
        HumanMessage(content=f"Codebase context:\n{context}\n\nQuestion: {query}")
    ]
    return llm.invoke(messages).content


def bug_agent(bug_description: str, chunks: List[Dict]) -> str:
    llm = _get_llm()
    context = _format_context(chunks)
    messages = [
        SystemMessage(content="""You are an expert debugger and code analyst.
Given a bug description and relevant code context, identify:
1. The most likely root cause
2. The exact file(s) and line(s) involved
3. Why this bug occurs
4. A concrete fix with code snippet"""),
        HumanMessage(content=f"Codebase context:\n{context}\n\nBug report: {bug_description}\n\nPerform root-cause analysis and suggest a fix.")
    ]
    return llm.invoke(messages).content


def diagram_agent(query: str, chunks: List[Dict], diagram_type: str = "architecture") -> Dict:
    llm = _get_llm()
    context = _format_context(chunks)
    type_instructions = {
        "architecture": "Generate a Mermaid flowchart (graph TD) showing system components, relationships, and data flow.",
        "er": "Generate a Mermaid erDiagram showing all entities, attributes, and relationships.",
        "dfd": "Generate a Mermaid flowchart showing data flows between processes, data stores, and external entities.",
        "class": "Generate a Mermaid classDiagram showing classes, attributes, methods, and relationships."
    }
    instruction = type_instructions.get(diagram_type, type_instructions["architecture"])
    messages = [
        SystemMessage(content=f"""You are an expert software architect who generates Mermaid diagrams.
{instruction}
Output ONLY valid Mermaid diagram code — no markdown fences, no explanation, nothing else."""),
        HumanMessage(content=f"Codebase context:\n{context}\n\nGenerate a {diagram_type} diagram for: {query}")
    ]
    mermaid_code = llm.invoke(messages).content.strip()
    if mermaid_code.startswith("```"):
        lines = mermaid_code.split("\n")
        mermaid_code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return {"diagram_type": diagram_type, "mermaid_code": mermaid_code}


def docs_agent(query: str, chunks: List[Dict]) -> str:
    llm = _get_llm()
    context = _format_context(chunks)
    messages = [
        SystemMessage(content="""You are a technical documentation expert.
Generate clear, professional documentation in Markdown format including:
- Overview, module/function descriptions, parameters, return types, usage examples."""),
        HumanMessage(content=f"Codebase context:\n{context}\n\nGenerate documentation for: {query}")
    ]
    return llm.invoke(messages).content


def impact_agent(change_description: str, chunks: List[Dict]) -> str:
    llm = _get_llm()
    context = _format_context(chunks)
    messages = [
        SystemMessage(content="""You are a senior software architect specialising in change impact analysis.
Analyse:
1. Direct impact — files and functions directly affected
2. Indirect impact — downstream dependencies that may break
3. Risk level — High / Medium / Low with justification
4. Testing recommendations
5. Suggested rollout strategy"""),
        HumanMessage(content=f"Codebase context:\n{context}\n\nProposed change: {change_description}\n\nAnalyse the full impact.")
    ]
    return llm.invoke(messages).content


def onboarding_agent(chunks: List[Dict], file_tree: List[str]) -> str:
    llm = _get_llm()
    context = _format_context(chunks[:12])
    tree_text = "\n".join(file_tree[:60])
    messages = [
        SystemMessage(content="""You are a developer onboarding assistant.
Generate a comprehensive onboarding guide in Markdown including:
1. Project purpose and overview
2. Repository structure explanation
3. Tech stack and dependencies
4. Local setup steps
5. Key entry points
6. Architecture summary
7. Where to start reading the code"""),
        HumanMessage(content=f"File tree:\n{tree_text}\n\nKey code context:\n{context}\n\nGenerate a complete onboarding guide.")
    ]
    return llm.invoke(messages).content