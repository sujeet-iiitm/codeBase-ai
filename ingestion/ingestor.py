"""Ingestion pipeline: clone repo → parse files → chunk → embed → store in ChromaDB : Uses Google Gemini embeddings"""

import os
import shutil
from pathlib import Path
from typing import List, Dict
import git
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go",
    ".cpp", ".c", ".cs", ".rb", ".rs", ".php", ".swift",
    ".md", ".txt", ".yml", ".yaml", ".json", ".toml"
}

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", "target", ".idea", ".vscode"
}


def clone_repo(repo_url: str, clone_dir: str = None) -> str:
    if clone_dir is None:
        clone_dir = os.getenv("CLONE_DIR", "./cloned_repos")
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = os.path.join(clone_dir, repo_name)
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    print(f"[Ingestion] Cloning {repo_url} ...")
    git.Repo.clone_from(repo_url, local_path, depth=1)
    print(f"[Ingestion] Cloned to {local_path}")
    return local_path


def collect_files(repo_path: str) -> List[Dict]:
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, repo_path)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if content.strip():
                        files.append({
                            "path": rel_path,
                            "content": content,
                            "extension": ext,
                            "filename": filename
                        })
                except Exception as e:
                    print(f"[Ingestion] Skipping {rel_path}: {e}")
    print(f"[Ingestion] Found {len(files)} source files")
    return files


def chunk_files(files: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
    )
    chunks = []
    for file in files:
        raw_chunks = splitter.split_text(file["content"])
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "file_path": file["path"],
                    "filename": file["filename"],
                    "extension": file["extension"],
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks)
                }
            })
    print(f"[Ingestion] Created {len(chunks)} chunks from {len(files)} files")
    return chunks


def embed_and_store(chunks: List[Dict], collection_name: str = "codebase") -> chromadb.Collection:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Gemini embeddings
    # embeddings_model = GoogleGenerativeAIEmbeddings(
    #     model="models/embedding-001",
    #     google_api_key=os.getenv("GEMINI_API_KEY")
    # )
    embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
    )

    BATCH_SIZE = 50
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        metas = [c["metadata"] for c in batch]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        vectors = embeddings_model.embed_documents(texts)
        collection.add(
            embeddings=vectors,
            documents=texts,
            metadatas=metas,
            ids=ids
        )
        print(f"[Ingestion] Embedded batch {i // BATCH_SIZE + 1}/{(len(chunks) - 1) // BATCH_SIZE + 1}")

    print(f"[Ingestion] Stored {len(chunks)} chunks in ChromaDB")
    return collection


def ingest_repo(repo_url: str, collection_name: str = "codebase") -> dict:
    local_path = clone_repo(repo_url)
    files = collect_files(local_path)
    chunks = chunk_files(files)
    collection = embed_and_store(chunks, collection_name)
    return {
        "repo_url": repo_url,
        "local_path": local_path,
        "total_files": len(files),
        "total_chunks": len(chunks),
        "collection_name": collection_name,
        "file_tree": [f["path"] for f in files]
    }


def get_retriever(collection_name: str = "codebase", top_k: int = 8):
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(collection_name)
    embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
    )

    def retrieve(query: str) -> List[Dict]:
        query_vec = embeddings_model.embed_query(query)
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        chunks_out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks_out.append({"text": doc, "metadata": meta, "score": 1 - dist})
        return chunks_out

    return retrieve
