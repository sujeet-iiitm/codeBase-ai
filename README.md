# CodeMind AI — Codebase (Github-Repo) Intelligence Assistant

An AI-powered multi-agent system that lets you chat with any GitHub codebase.
Ask questions, find bugs, generate diagrams, auto-create docs, and analyse change impact.

## Features

| Feature            | What it does.
| Codebase Q&A       | Ask natural language questions about any code.
| Bug root-cause     | Describe a bug and get a root-cause analysis + fix.
| Diagram generation | Auto-generate Architecture / ER / DFD / Class diagrams (Mermaid).
| Auto documentation | Generate README-quality docs for any module or function.
| Impact analysis    | Describe a change and get risk + affected files.
| Repo onboarding    | Get a full onboarding guide for any new repo.


## Tech Stack

- **LLM**: OpenAI GPT-4o
- **Orchestration**: LangChain + LangGraph (multi-agent routing)
- **Vector DB**: ChromaDB (persistent, local)
- **Embeddings**: OpenAI text-embedding-ada-002
- **Code parsing**: tree-sitter, GitPython
- **Diagrams**: Mermaid.js
- **UI**: Streamlit
- **Backend**: FastAPI (optional REST layer)


## Architecture

GitHub Repo URL
     │
     ▼
Ingestion Pipeline
(clone → parse → chunk → embed)
     │
     ▼
ChromaDB Vector Store
     │
     ▼
LangGraph Orchestrator
(intent routing)
     │
  ┌──┼──────────────┬──────────────┬──────────┐
  ▼  ▼              ▼              ▼          ▼
Q&A  Bug RCA  Diagram Gen    Docs Gen  Impact Analysis
                                        Onboarding
     │
     ▼
Streamlit UI

## Setup

### 1. Clone this repo
bash :
git clone https://github.com/yourusername/codemind-ai
cd codemind-ai

### 2. Create a virtual environment
bash :
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

### 3. Install dependencies
bash :
pip install -r requirements.txt

### 4. Add your API key
bash :
cp .env.example .env

# Edit .env and add your OPENAI_API_KEY

### 5. Run the app
bash :
cd ui
streamlit run app.py

Open http://localhost:8501 in your browser.


## Usage

1. Enter your OpenAI API key in the sidebar
2. Paste any public GitHub repo URL
3. Click **Ingest Repository** (takes ~1-2 min for large repos)
4. Start asking questions in the chat!

### Example queries

How does authentication work in this codebase?
I'm getting a KeyError in the user service, help me debug it
Generate an architecture diagram of this project
Generate the ER diagram for all database models
What's the impact of removing the Redis cache layer?
Give me a complete onboarding guide for a new developer


## Project Structure

codebase-ai/
├── ingestion/
│   └── ingestor.py           # Clone, parse, chunk, embed pipeline
├── agents/
│   ├── specialist_agents.py  # Q&A, bug, diagram, docs, impact, onboarding agents
│   └── orchestrator.py       # LangGraph multi-agent router
├── ui/
│   └── app.py                # Streamlit frontend
├── requirements.txt
├── .env.example
└── README.md


## Resume Talking Points

- Built a **multi-agent RAG system** using LangGraph for intent-based routing across 6 specialist agents
- Implemented **semantic code search** using OpenAI embeddings + ChromaDB vector store with cosine similarity
- Designed an **ingestion pipeline** that clones, parses, and chunks code across 15+ file types
- Auto-generates **Mermaid diagrams** (ER, DFD, architecture, class) directly from source code
- **Zero hallucinations** — all answers are grounded in retrieved code context with file citations


## Contributing

PRs welcome! See open issues for ideas.

## License
MIT
