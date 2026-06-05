import os
import tempfile
import streamlit as st
import vector_store
from rag_graph import build_graph, RAGState

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
AUTHOR = "SHANJAY"
EMAIL = "shanjayravikumar02@gmail.com"

st.set_page_config(
    page_title="ReflectRAG",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN SYSTEM ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg:      #0A0A0A;
    --surface: #0E0E0E;
    --line:    #2A2A2A;
    --line-hi: #FAFAFA;
    --ink:     #FAFAFA;
    --muted:   #8A8A8A;
    --accent:  #FF3D00;
}

/* ── Base ── */
html, body, p, div, span, li, td, th, label, a, input, textarea, caption {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, .serif {
    font-family: 'Instrument Serif', serif !important;
    font-weight: 400;
    letter-spacing: -0.01em;
}
.mono, .mono * {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Preserve material icon glyphs */
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded,
.material-symbols-outlined, i[class*="material"] {
    font-family: 'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
}

/* Kill chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ── Zero radius / zero shadow everywhere ── */
*, *::before, *::after {
    border-radius: 0 !important;
    box-shadow: none !important;
}

/* ── Canvas ── */
[data-testid="stAppViewContainer"] { background: var(--bg); }
.main .block-container { padding-top: 0 !important; max-width: 1280px; }

/* ── Film grain overlay ── */
.grain {
    position: fixed; inset: 0; z-index: 9998; pointer-events: none; opacity: 0.05;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

.sb-brand {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 800; font-size: 1.05rem; letter-spacing: 0.04em; color: var(--ink);
}
.sb-tag {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem; letter-spacing: 0.18em; color: var(--muted);
    text-transform: uppercase; margin-top: 0.35rem;
}
.sb-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem; letter-spacing: 0.22em; color: var(--muted);
    text-transform: uppercase; margin: 1.6rem 0 0.7rem 0;
    padding-bottom: 0.4rem; border-bottom: 1px solid var(--line);
}
.sb-step {
    display: flex; gap: 0.85rem; padding: 0.7rem 0;
    border-bottom: 1px solid var(--line);
}
.sb-step-idx {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem; color: var(--accent); font-weight: 700; min-width: 1.4rem;
}
.sb-step-name {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--ink); font-weight: 700;
}
.sb-step-desc { font-size: 0.74rem; color: var(--muted); margin-top: 0.15rem; line-height: 1.4; }
.sb-chip {
    font-family: 'JetBrains Mono', monospace !important;
    display: inline-block; border: 1px solid var(--line); color: var(--muted);
    font-size: 0.62rem; letter-spacing: 0.08em; padding: 0.25rem 0.55rem;
    margin: 0 0.3rem 0.3rem 0; text-transform: uppercase;
}
.sb-status {
    font-family: 'JetBrains Mono', monospace !important;
    border: 1px solid var(--accent); color: var(--accent);
    font-size: 0.68rem; letter-spacing: 0.08em; padding: 0.5rem 0.7rem;
    text-transform: uppercase;
}
.sb-credit {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.64rem; color: var(--muted); line-height: 1.7;
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--line);
}
.sb-credit a { color: var(--muted); text-decoration: none; }
.sb-credit a:hover { color: var(--accent); }

/* ── Top nav ── */
.topnav {
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid var(--line); padding: 1rem 0; margin-bottom: 0;
}
.topnav-brand {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 800; letter-spacing: 0.06em; font-size: 0.95rem; color: var(--ink);
}
.topnav-brand b { color: var(--accent); }
.topnav-meta {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem; letter-spacing: 0.14em; color: var(--muted); text-transform: uppercase;
}

/* ── Hero ── */
.hero {
    display: grid; grid-template-columns: 1fr 320px;
    border-bottom: 1px solid var(--line);
}
.hero-main { padding: 3rem 2.5rem 3rem 0; }
.hero-side {
    border-left: 1px solid var(--line); padding: 3rem 0 3rem 2.5rem;
    display: flex; flex-direction: column; justify-content: flex-end;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.66rem; letter-spacing: 0.28em; color: var(--accent);
    text-transform: uppercase; margin-bottom: 1.4rem;
}
.hero-title {
    font-family: 'Instrument Serif', serif !important;
    font-size: 6rem; line-height: 0.92; color: var(--ink);
    letter-spacing: -0.02em; margin: 0;
}
.hero-title em { font-style: italic; color: var(--accent); }
.hero-desc {
    color: var(--muted); font-size: 1rem; line-height: 1.7;
    max-width: 540px; margin-top: 1.6rem;
}
.hero-desc b { color: var(--ink); font-weight: 600; }
.hero-meta-row {
    font-family: 'JetBrains Mono', monospace !important;
    display: flex; justify-content: space-between;
    font-size: 0.66rem; letter-spacing: 0.1em; color: var(--muted);
    text-transform: uppercase; padding: 0.55rem 0; border-top: 1px solid var(--line);
}
.hero-meta-row span:last-child { color: var(--ink); }

/* ── Section heading ── */
.sec-head {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.66rem; letter-spacing: 0.26em; color: var(--muted);
    text-transform: uppercase; margin: 2.4rem 0 1rem 0;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--line-hi);
    display: flex; justify-content: space-between;
}
.sec-head b { color: var(--accent); }

/* ── Trace block ── */
.node {
    border: 1px solid var(--line); border-top: none;
    padding: 1.5rem 1.7rem;
}
.node:first-of-type { border-top: 1px solid var(--line); }
.node-head {
    font-family: 'JetBrains Mono', monospace !important;
    display: flex; align-items: baseline; gap: 1rem;
    font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--ink); margin-bottom: 1rem;
}
.node-idx { color: var(--accent); font-weight: 800; }
.node-sub { color: var(--muted); letter-spacing: 0.1em; }

/* ── Chunk row ── */
.chunk {
    border: 1px solid var(--line); padding: 0.8rem 1rem; margin-top: 0.6rem;
    color: var(--muted); font-size: 0.84rem; line-height: 1.65;
}
.chunk-tag {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.58rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--accent); display: block; margin-bottom: 0.4rem;
}

/* ── Answer ── */
.answer {
    border-left: 2px solid var(--accent); background: var(--surface);
    padding: 1.3rem 1.6rem; color: var(--ink); font-size: 1rem; line-height: 1.75;
}

/* ── Verdict ── */
.verdict-pass {
    font-family: 'JetBrains Mono', monospace !important;
    display: inline-block; border: 1px solid var(--line-hi); color: var(--ink);
    font-weight: 700; font-size: 0.74rem; letter-spacing: 0.2em;
    padding: 0.4rem 1.1rem; text-transform: uppercase;
}
.verdict-fail {
    font-family: 'JetBrains Mono', monospace !important;
    display: inline-block; background: var(--accent); color: #000;
    font-weight: 800; font-size: 0.74rem; letter-spacing: 0.2em;
    padding: 0.4rem 1.1rem; text-transform: uppercase;
}

/* ── Score table (collapsed borders) ── */
.scores { display: flex; border: 1px solid var(--line); margin-top: 1.1rem; }
.score-cell {
    flex: 1; padding: 1.2rem 1.4rem; border-right: 1px solid var(--line);
}
.score-cell:last-child { border-right: none; }
.score-cap {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem; letter-spacing: 0.18em; color: var(--muted);
    text-transform: uppercase;
}
.score-val {
    font-family: 'Instrument Serif', serif !important;
    font-size: 3rem; line-height: 1; color: var(--ink); margin-top: 0.3rem;
}
.score-val u { text-decoration: none; color: var(--muted); font-size: 1.1rem; }

/* ── Refine notice ── */
.refine {
    font-family: 'JetBrains Mono', monospace !important;
    border: 1px solid var(--accent); color: var(--accent);
    font-size: 0.72rem; letter-spacing: 0.04em; padding: 0.6rem 0.9rem;
    margin-top: 0.9rem; text-transform: uppercase;
}
.refine i { color: var(--ink); font-style: normal; }

/* ── Reason line ── */
.reason {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--muted); font-size: 0.74rem; line-height: 1.55; margin-top: 0.9rem;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input {
    background: var(--surface) !important; border: 1px solid var(--line) !important;
    color: var(--ink) !important; font-size: 0.95rem !important;
    padding: 0.85rem 1.1rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important; outline: none !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: var(--muted) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important;
}

/* ── Buttons ── */
[data-testid="baseButton-primary"], [data-testid="baseButton-secondary"] {
    font-family: 'JetBrains Mono', monospace !important;
    background: transparent !important; border: 1px solid var(--line-hi) !important;
    color: var(--ink) !important; font-weight: 700 !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
    font-size: 0.74rem !important; transition: background 0.12s, color 0.12s !important;
}
[data-testid="baseButton-primary"]:hover {
    background: var(--accent) !important; border-color: var(--accent) !important; color: #000 !important;
}
[data-testid="baseButton-secondary"]:hover {
    background: var(--ink) !important; color: #000 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploadDropzone"] {
    background: var(--surface) !important; border: 1px solid var(--line) !important; padding: 0.7rem !important;
}
[data-testid="stFileUploadDropzone"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploadDropzone"] p, [data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small {
    color: var(--muted) !important; font-size: 0.72rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stFileUploadDropzone"] button {
    background: transparent !important; border: 1px solid var(--line-hi) !important;
    color: var(--ink) !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important;
}

/* ── Expander ── */
[data-testid="stExpander"] { background: transparent !important; border: 1px solid var(--line) !important; }
[data-testid="stExpander"] summary { font-family: 'JetBrains Mono', monospace !important; }

/* ── Streamlit caption ── */
[data-testid="stCaptionContainer"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--muted) !important; font-size: 0.66rem !important; letter-spacing: 0.06em !important;
}

/* ── Footer ── */
.foot {
    font-family: 'JetBrains Mono', monospace !important;
    border-top: 1px solid var(--line); margin-top: 3.5rem; padding: 1.4rem 0;
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.6rem;
    font-size: 0.64rem; letter-spacing: 0.12em; color: var(--muted); text-transform: uppercase;
}
.foot a { color: var(--ink); text-decoration: none; }
.foot a:hover { color: var(--accent); }

/* ── Responsive ── */
@media (max-width: 768px) {
    .hero { grid-template-columns: 1fr; }
    .hero-side { border-left: none; border-top: 1px solid var(--line); padding: 2rem 0; }
    .hero-main { padding: 2rem 0; }
    .hero-title { font-size: 3.4rem; }
    .scores { flex-direction: column; }
    .score-cell { border-right: none; border-bottom: 1px solid var(--line); }
    .score-cell:last-child { border-bottom: none; }
}
</style>
<div class="grain"></div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("ready", False), ("chunk_count", 0)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">REFLECT<span style="color:#FF3D00">RAG</span></div>
    <div class="sb-tag">Self-Healing Retrieval</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "docs", type=["txt", "pdf"], accept_multiple_files=True, label_visibility="hidden",
    )
    st.caption("TXT / PDF — empty uses sample corpus")

    if st.button("Build Index", type="primary", use_container_width=True):
        if uploaded_files:
            tmp = tempfile.mkdtemp()
            for f in uploaded_files:
                with open(os.path.join(tmp, f.name), "wb") as o:
                    o.write(f.getbuffer())
            target = tmp
        else:
            target = DOCS_DIR
        with st.spinner("Indexing..."):
            try:
                vector_store.build_vector_store(target)
                st.session_state.ready = True
                st.session_state.chunk_count = len(vector_store._chunks)
            except Exception as e:
                st.error(str(e))

    if not st.session_state.ready:
        with st.spinner("Loading sample corpus..."):
            try:
                vector_store.build_vector_store(DOCS_DIR)
                st.session_state.ready = True
                st.session_state.chunk_count = len(vector_store._chunks)
            except Exception:
                pass

    if st.session_state.ready:
        st.markdown(
            f'<div class="sb-status">INDEX READY — {st.session_state.chunk_count} CHUNKS</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sb-label">Pipeline</div>', unsafe_allow_html=True)
    for idx, name, desc in [
        ("01", "Retrieve", "Semantic vector search across corpus"),
        ("02", "Generate", "Answer constrained to retrieved context"),
        ("03", "Critic", "Scores faithfulness and relevance"),
        ("04", "Retry", "Refines query when scores fall short"),
        ("05", "Fallback", "Safe response after two failures"),
    ]:
        st.markdown(f"""
        <div class="sb-step">
            <div class="sb-step-idx">{idx}</div>
            <div>
                <div class="sb-step-name">{name}</div>
                <div class="sb-step-desc">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <span class="sb-chip">LangGraph</span>
        <span class="sb-chip">FAISS</span>
        <span class="sb-chip">Groq</span>
        <span class="sb-chip">LLaMA 3.3 70B</span>
        <span class="sb-chip">Streamlit</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sb-credit">
        BUILT BY {AUTHOR}<br>
        <a href="mailto:{EMAIL}">{EMAIL}</a>
    </div>""", unsafe_allow_html=True)

# ── Top nav ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topnav">
    <div class="topnav-brand">REFLECT<b>RAG</b></div>
    <div class="topnav-meta">Self-Healing RAG &nbsp;/&nbsp; v1.0 &nbsp;/&nbsp; SHANJAY</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-main">
        <div class="hero-eyebrow">[ AI / RETRIEVAL / SELF-HEALING ]</div>
        <h1 class="hero-title">Reflect<em>RAG</em></h1>
        <p class="hero-desc">
            An engineered retrieval pipeline that answers from your documents, then
            <b>critiques its own output</b> and retries with a sharper query when the
            answer fails to meet a faithfulness threshold.
        </p>
    </div>
    <div class="hero-side">
        <div class="hero-meta-row"><span>Orchestration</span><span>LangGraph</span></div>
        <div class="hero-meta-row"><span>Retrieval</span><span>FAISS</span></div>
        <div class="hero-meta-row"><span>Model</span><span>LLaMA 3.3 70B</span></div>
        <div class="hero-meta-row"><span>Max Retries</span><span>02</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Query ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head"><span>Query</span><b>// INPUT</b></div>', unsafe_allow_html=True)
c1, c2 = st.columns([5, 1])
with c1:
    query = st.text_input(
        "q", label_visibility="collapsed",
        placeholder="INTERROGATE THE CORPUS...",
        disabled=not st.session_state.ready,
    )
with c2:
    run = st.button("Execute", type="primary", use_container_width=True,
                    disabled=not st.session_state.ready or not query)

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run and query:
    st.session_state.last_result = None
    graph = build_graph()
    init: RAGState = {
        "query": query, "refined_query": query, "context": [], "answer": "",
        "retry_count": 0, "critique_reason": "", "faithfulness_score": 0, "relevance_score": 0,
    }

    st.markdown('<div class="sec-head"><span>Trace</span><b>// EXECUTION</b></div>', unsafe_allow_html=True)

    attempt = 0
    final_answer = ""
    final_faith = final_relev = final_retries = 0

    for event in graph.stream(init):
        for node, out in event.items():

            if node == "retrieve":
                attempt += 1
                chunks = out.get("context", [])
                rows = "".join(
                    f'<div class="chunk"><span class="chunk-tag">Chunk {i:02d}</span>{c}</div>'
                    for i, c in enumerate(chunks, 1)
                ) if chunks else '<div class="reason">No chunks retrieved.</div>'
                st.markdown(f"""
                <div class="node">
                    <div class="node-head"><span class="node-idx">01</span> / RETRIEVE
                        <span class="node-sub">ATTEMPT {attempt:02d} — {len(chunks)} CHUNKS</span></div>
                    {rows}
                </div>""", unsafe_allow_html=True)

            elif node == "generate":
                ans = out.get("answer", "")
                final_answer = ans
                st.markdown(f"""
                <div class="node">
                    <div class="node-head"><span class="node-idx">02</span> / GENERATE
                        <span class="node-sub">ATTEMPT {attempt:02d}</span></div>
                    <div class="answer">{ans}</div>
                </div>""", unsafe_allow_html=True)

            elif node == "critic":
                reason = out.get("critique_reason", "")
                is_pass = reason.startswith("PASS")
                faith = out.get("faithfulness_score", 0)
                relev = out.get("relevance_score", 0)
                badge = '<span class="verdict-pass">PASS</span>' if is_pass \
                    else '<span class="verdict-fail">FAIL</span>'
                refine = ""
                if not is_pass:
                    rq = out.get("refined_query", "")
                    if rq:
                        refine = f'<div class="refine">REFINED QUERY &rarr; <i>{rq}</i></div>'
                st.markdown(f"""
                <div class="node">
                    <div class="node-head"><span class="node-idx">03</span> / CRITIC
                        <span class="node-sub">ATTEMPT {attempt:02d}</span></div>
                    <div>{badge}</div>
                    <div class="scores">
                        <div class="score-cell"><div class="score-cap">Faithfulness</div>
                            <div class="score-val">{faith}<u>/10</u></div></div>
                        <div class="score-cell"><div class="score-cap">Relevance</div>
                            <div class="score-val">{relev}<u>/10</u></div></div>
                    </div>
                    <div class="reason">{reason}</div>
                    {refine}
                </div>""", unsafe_allow_html=True)
                final_faith, final_relev, final_retries = faith, relev, attempt - 1

            elif node == "fallback":
                final_answer = out.get("answer", "")
                st.markdown(f"""
                <div class="node">
                    <div class="node-head"><span class="node-idx">05</span> / FALLBACK
                        <span class="node-sub">SAFE RESPONSE</span></div>
                    <div class="reason" style="color:var(--accent)">{final_answer}</div>
                </div>""", unsafe_allow_html=True)

    # ── Result ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head"><span>Result</span><b>// FINAL</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer" style="font-size:1.05rem">{final_answer}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="scores">
        <div class="score-cell"><div class="score-cap">Retries Used</div>
            <div class="score-val">{final_retries}<u>/2</u></div></div>
        <div class="score-cell"><div class="score-cap">Faithfulness</div>
            <div class="score-val">{final_faith}<u>/10</u></div></div>
        <div class="score-cell"><div class="score-cap">Relevance</div>
            <div class="score-val">{final_relev}<u>/10</u></div></div>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="foot">
    <span>REFLECTRAG &nbsp;/&nbsp; SELF-HEALING RAG PIPELINE</span>
    <span>BUILT BY {AUTHOR} &nbsp;/&nbsp; <a href="mailto:{EMAIL}">{EMAIL}</a></span>
    <span>LANGGRAPH / FAISS / GROQ / STREAMLIT</span>
</div>
""", unsafe_allow_html=True)
