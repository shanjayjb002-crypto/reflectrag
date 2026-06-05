# ReflectRAG — Self-Healing RAG Pipeline

A production-pattern Retrieval-Augmented Generation system built with **LangGraph** that automatically detects low-quality answers and retries with a refined search query — without any human intervention.

---

## How It Works

```
User Query
    │
    ▼
┌─────────┐     ┌──────────┐     ┌────────┐
│ Retrieve│────▶│ Generate │────▶│ Critic │
└─────────┘     └──────────┘     └────────┘
     ▲                               │
     │        FAIL + retries < 2     │  PASS
     └───────────────────────────────┤────▶ Final Answer
                                     │
                            FAIL + retries ≥ 2
                                     │
                                     ▼
                               ┌──────────┐
                               │ Fallback │────▶ Safe Message
                               └──────────┘
```

1. **Retrieve** — Encodes the query with `all-MiniLM-L6-v2` and performs semantic search over a FAISS index built from your documents
2. **Generate** — Sends retrieved chunks + question to Groq LLaMA 3.3 70B; the LLM answers using *only* the provided context
3. **Critic** — A second LLM call evaluates the answer on faithfulness (grounded in context?) and relevance (answers the question?), returning a PASS/FAIL verdict with scores out of 10
4. **Retry** — On FAIL, the critic rewrites the search query and the pipeline loops back to Retrieve (max 2 retries)
5. **Fallback** — After 2 failed attempts, returns a safe, transparent message explaining why

---

## Features

- **Self-healing loop** — automatically retries with smarter queries on low-quality answers
- **Dual LLM evaluation** — separate generate and critic calls prevent self-grading bias
- **Evaluation scores** — faithfulness and relevance metrics on every response
- **Web UI** — Streamlit interface showing each pipeline step in real time
- **PDF + TXT support** — upload any combination of document types
- **Local vector search** — FAISS + sentence-transformers, no external vector DB needed
- **Free to run** — powered by Groq's free API tier

---

## Tech Stack

| Component | Technology |
|---|---|
| Pipeline orchestration | LangGraph (StateGraph) |
| LLM | Groq — LLaMA 3.3 70B |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS (IndexFlatL2) |
| Text chunking | LangChain RecursiveCharacterTextSplitter |
| Web UI | Streamlit |
| PDF parsing | pypdf |

---

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/YOUR_USERNAME/self-healing-rag.git
cd self-healing-rag
pip install -r requirements.txt
```

**2. Get a free Groq API key**

Sign up at [console.groq.com](https://console.groq.com) — no credit card required.

**3. Set your API key**
```bash
# macOS / Linux
export GROQ_API_KEY="gsk_..."

# Windows PowerShell
$env:GROQ_API_KEY = "gsk_..."
```

---

## Usage

### Web Interface
```bash
streamlit run app.py
```
Opens in your browser at `http://localhost:8501`. Upload documents via the sidebar, type a question, and watch each pipeline step execute in real time.

### Command Line
```bash
python main.py "What is retrieval-augmented generation?"
```

### Add Your Own Documents
Drop `.txt` or `.pdf` files into the `docs/` folder (CLI) or upload them through the web UI sidebar.

---

## Project Structure

```
self-healing-rag/
├── app.py             # Streamlit web interface
├── main.py            # CLI entry point
├── rag_graph.py       # LangGraph StateGraph — all 4 nodes + routing logic
├── vector_store.py    # FAISS index build + semantic search
├── requirements.txt   # Python dependencies
└── docs/
    └── sample.txt     # Sample document for testing
```

---

## Example Output

```
Query: "What is a vector database?"

Step 1 — Retrieved 4 context chunks
Step 1 — Generated answer using context
Step 1 — Critic: PASS | Faithfulness: 9/10 | Relevance: 10/10

Final Answer:
Vector databases store data as dense numerical vectors and enable fast
similarity search. Unlike keyword-based search, vector search finds
semantically similar content even when exact words differ...
```
