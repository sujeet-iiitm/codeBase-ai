import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st

if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

import streamlit as st
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

#Page config
st.set_page_config(
    page_title="CodeMind-AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; background: linear-gradient(90deg, #6366f1, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .feature-badge { background: #f0f0ff; color: #5b4fcf; padding: 4px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 500; margin: 2px; display: inline-block; }
    .intent-badge { padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .intent-qa      { background:#dbeafe; color:#1e40af; }
    .intent-bug     { background:#fee2e2; color:#991b1b; }
    .intent-diagram { background:#d1fae5; color:#065f46; }
    .intent-docs    { background:#fef3c7; color:#92400e; }
    .intent-impact  { background:#fce7f3; color:#9d174d; }
    .intent-onboarding { background:#ede9fe; color:#5b21b6; }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

#Lazy imports (only when API key set)
def load_pipeline():
    from ingestion.ingestor import ingest_repo, get_retriever
    from agents.orchestrator import run_agent
    return ingest_repo, get_retriever, run_agent


#Session state init
for key, default in {
    "ingested": False,
    "repo_meta": None,
    "chat_history": [],
    "retriever": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


#Sidebar
with st.sidebar:
    st.markdown("🧠 CodeMind-AI")
    st.markdown("*AI Codebase Intelligence Assistant*")
    st.divider()

    # API Key input
    api_key = st.text_input("Gemini API Key", type="password", placeholder="sk-...")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()
    st.markdown("### Repository")
    repo_url = st.text_input(
        "GitHub Repo URL",
        placeholder="https://github.com/user/repo",
        help="Public GitHub repositories only"
    )

    if st.button("🚀 Ingest Repository", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key first.")
        elif not repo_url:
            st.error("Please enter a GitHub repo URL.")
        else:
            with st.spinner("Cloning, parsing and embedding the codebase... this takes ~1-2 min"):
                try:
                    ingest_repo, get_retriever, _ = load_pipeline()
                    meta = ingest_repo(repo_url)
                    st.session_state.repo_meta = meta
                    st.session_state.retriever = get_retriever()
                    st.session_state.ingested = True
                    st.session_state.chat_history = []
                    st.success(f"✅ Ingested! {meta['total_files']} files, {meta['total_chunks']} chunks")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    if st.session_state.ingested and st.session_state.repo_meta:
        meta = st.session_state.repo_meta
        st.divider()
        st.markdown("### Repo Stats")
        col1, col2 = st.columns(2)
        col1.metric("Files", meta["total_files"])
        col2.metric("Chunks", meta["total_chunks"])

        with st.expander("File tree", expanded=False):
            tree = "\n".join(meta.get("file_tree", [])[:50])
            st.code(tree, language=None)

    st.divider()
    st.markdown("### Capabilities")
    features = [
        "Codebase Q&A",
        "Bug root-cause",
        "ER/DFD/Architecture",
        "Auto docs",
        "Impact analysis",
        "Repo onboarding"
    ]
    for f in features:
        st.markdown(f'<span class="feature-badge">{f}</span>', unsafe_allow_html=True)


#Main panel
st.markdown('<p class="main-header">🧠 CodeMind-AI</p>', unsafe_allow_html=True)
st.markdown("*Ask anything about your codebase — Q&A, bugs, diagrams, docs, impact analysis*")

if not st.session_state.ingested:
    st.info("👈 Enter your OpenAI API key and a GitHub repo URL in the sidebar to get started.")
    st.markdown("#### Example queries you can ask after ingestion:")
    examples = [
        "How does authentication work in this repo?",
        "Debug: I'm getting a NullPointerException in UserService.java",
        "Generate an architecture diagram of this project",
        "Generate ER diagram for the database models",
        "What's the impact of removing the caching layer?",
        "Give me an onboarding guide for this repo"
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")

else:
    #Chat history
    for msg in st.session_state.chat_history:
        role = msg["role"]
        with st.chat_message(role):
            if msg.get("intent"):
                intent = msg["intent"]
                st.markdown(
                    f'<span class="intent-badge intent-{intent}">🎯 {intent.upper()} agent</span>',
                    unsafe_allow_html=True
                )
            if msg.get("is_diagram") and msg.get("mermaid"):
                st.markdown(f"```mermaid\n{msg['mermaid']}\n```")
            else:
                st.markdown(msg["content"])

    #Chat input
    user_query = st.chat_input("Ask about the codebase...")

    if user_query:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Run pipeline
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    _, _, run_agent = load_pipeline()
                    retriever = st.session_state.retriever
                    chunks = retriever(user_query)
                    output = run_agent(
                        user_query,
                        chunks,
                        st.session_state.repo_meta.get("file_tree", [])
                    )

                    intent = output.get("intent", "qa")
                    st.markdown(
                        f'<span class="intent-badge intent-{intent}">🎯 {intent.upper()} agent</span>',
                        unsafe_allow_html=True
                    )

                    if output.get("error"):
                        st.error(f"Error: {output['error']}")
                        result_content = f"Error: {output['error']}"
                        is_diagram = False
                        mermaid_code = None

                    elif intent == "diagram" and output.get("diagram_result"):
                        mermaid_code = output["diagram_result"]["mermaid_code"]
                        st.markdown(f"```mermaid\n{mermaid_code}\n```")
                        is_diagram = True
                        result_content = mermaid_code

                    else:
                        result_content = output.get("result", "No result returned.")
                        st.markdown(result_content)
                        is_diagram = False
                        mermaid_code = None

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "intent": intent,
                        "content": result_content,
                        "is_diagram": is_diagram,
                        "mermaid": mermaid_code
                    })

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    #Quick-action buttons
    st.divider()
    st.markdown("##### Quick actions")
    cols = st.columns(3)
    quick_actions = [
        ("🗺️ Onboard me", "Give me a complete onboarding guide for this repository"),
        ("📊 Architecture diagram", "Generate an architecture diagram of this project"),
        ("📝 Generate README", "Generate comprehensive README documentation for this project")
    ]
    for col, (label, query) in zip(cols, quick_actions):
        if col.button(label, use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.rerun()
