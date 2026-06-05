import sys
import os

import vector_store
from rag_graph import build_graph, RAGState

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
DEFAULT_QUERY = "What is retrieval-augmented generation and how does it reduce hallucination?"


def main() -> None:
    query: str = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUERY

    print(f"[INFO] Building vector store from: {DOCS_DIR}")
    vector_store.build_vector_store(DOCS_DIR)
    print(f"[INFO] Vector store ready — {len(vector_store._chunks)} chunks indexed")

    graph = build_graph()

    initial_state: RAGState = {
        "query": query,
        "refined_query": query,
        "context": [],
        "answer": "",
        "retry_count": 0,
        "critique_reason": "",
        "faithfulness_score": 0,
        "relevance_score": 0,
    }

    print(f"[INFO] Query: {query!r}\n")
    final_state = graph.invoke(initial_state)

    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_state["answer"])
    print("=" * 60)
    print(f"[INFO] Retries used:      {final_state['retry_count']}")
    print(f"[INFO] Critique:          {final_state['critique_reason']}")
    print(f"[INFO] Faithfulness:      {final_state['faithfulness_score']}/10")
    print(f"[INFO] Relevance:         {final_state['relevance_score']}/10")


if __name__ == "__main__":
    main()
