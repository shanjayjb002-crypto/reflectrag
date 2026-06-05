from groq import Groq
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import vector_store

MAX_RETRIES = 2
MODEL = "llama-3.3-70b-versatile"

# Lazy client — created on first use so importing this module never fails,
# even if the API key is not yet present in the environment.
_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()
    return _client


class RAGState(TypedDict):
    query: str              # Original question — never mutated
    refined_query: str      # Current retrieval string; critic rewrites on FAIL
    context: list[str]      # Chunks from the latest retrieve call
    answer: str             # Current generated or fallback answer
    retry_count: int        # Number of FAIL evaluation cycles so far
    critique_reason: str    # "PASS: <reason>" or "FAIL: <reason>"
    faithfulness_score: int # 1-10: how grounded the answer is in context
    relevance_score: int    # 1-10: how well the answer addresses the question


def retrieve(state: RAGState) -> dict:
    chunks = vector_store.search(state["refined_query"], k=4)
    return {"context": chunks}


def generate(state: RAGState) -> dict:
    if not state["context"]:
        return {"answer": "No relevant context was retrieved for this query."}

    context_text = "\n---\n".join(state["context"])
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context does not contain enough information to answer, say so explicitly rather than guessing."
    )
    user_message = f"Context:\n{context_text}\n\nQuestion: {state['query']}"

    response = _get_client().chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return {"answer": response.choices[0].message.content}


def critic(state: RAGState) -> dict:
    context_text = "\n---\n".join(state["context"]) if state["context"] else "[No context]"
    prompt = (
        f"You are a strict quality evaluator for a RAG system.\n\n"
        f"Original question: {state['query']}\n"
        f"Context used:\n{context_text}\n\n"
        f"Generated answer: {state['answer']}\n\n"
        f"Evaluate whether the answer is accurate, complete, and grounded in the provided context.\n"
        f"Respond ONLY in this exact 5-line format with no extra text:\n"
        f"VERDICT: PASS or FAIL\n"
        f"REASON: <one concise sentence>\n"
        f"REFINED_QUERY: <improved search query if FAIL, or the original query if PASS>\n"
        f"FAITHFULNESS: <integer 1-10, how well the answer is grounded in the context>\n"
        f"RELEVANCE: <integer 1-10, how well the answer addresses the original question>"
    )

    response = _get_client().chat.completions.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = response.choices[0].message.content.strip()

    verdict = ""
    reason = ""
    refined_query = state["query"]
    faithfulness_score = 0
    relevance_score = 0

    for line in response_text.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip()
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
        elif line.startswith("REFINED_QUERY:"):
            refined_query = line.split(":", 1)[1].strip()
        elif line.startswith("FAITHFULNESS:"):
            try:
                faithfulness_score = int(line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                faithfulness_score = 0
        elif line.startswith("RELEVANCE:"):
            try:
                relevance_score = int(line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                relevance_score = 0

    if verdict not in ("PASS", "FAIL"):
        verdict = "FAIL"
        reason = "Could not parse critic response."

    if verdict == "PASS":
        return {
            "critique_reason": f"PASS: {reason}",
            "faithfulness_score": faithfulness_score,
            "relevance_score": relevance_score,
        }
    else:
        return {
            "retry_count": state["retry_count"] + 1,
            "critique_reason": f"FAIL: {reason}",
            "refined_query": refined_query,
            "faithfulness_score": faithfulness_score,
            "relevance_score": relevance_score,
        }


def fallback(state: RAGState) -> dict:
    msg = (
        f"I was unable to produce a confident answer after {state['retry_count']} attempt(s). "
        f"Last known issue: {state['critique_reason']}. "
        f"Please try rephrasing your query or expanding the document collection."
    )
    return {"answer": msg}


def route_after_critic(state: RAGState) -> str:
    if state["critique_reason"].startswith("PASS"):
        return "end"
    elif state["retry_count"] < MAX_RETRIES:
        return "retrieve"
    else:
        return "fallback"


def build_graph():
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("critic", critic)
    builder.add_node("fallback", fallback)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"end": END, "retrieve": "retrieve", "fallback": "fallback"},
    )
    builder.add_edge("fallback", END)

    return builder.compile()
