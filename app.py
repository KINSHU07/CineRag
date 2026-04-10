"""
Streamlit UI for Movie RAG — supports huggingface | mistral | claude | openai
"""

import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="🎬 CineRAG – Movie Q&A",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');
    :root {
        --bg:       #0a0a0f;
        --surface:  #13131a;
        --card:     #1c1c28;
        --gold:     #e8b84b;
        --gold-dim: #9a7a2e;
        --text:     #e8e6df;
        --muted:    #7a7880;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid #222230;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4.2rem;
        letter-spacing: 3px;
        background: linear-gradient(135deg, var(--gold) 0%, #fff8e7 60%, var(--gold-dim) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin-bottom: 0.15rem;
    }
    .hero-sub {
        color: var(--muted);
        font-size: 0.95rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }
    .gold-divider {
        height: 2px;
        background: linear-gradient(90deg, var(--gold), transparent);
        border: none;
        margin: 1.5rem 0;
    }
    .answer-card {
        background: var(--card);
        border: 1px solid #2a2a3a;
        border-left: 4px solid var(--gold);
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
        font-size: 1.05rem;
        line-height: 1.75;
        color: var(--text);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        animation: fadeSlide 0.5s ease;
    }
    @keyframes fadeSlide {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .source-card {
        background: var(--card);
        border: 1px solid #252535;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }
    .source-card:hover { border-color: var(--gold-dim); }
    .source-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.15rem;
        letter-spacing: 1px;
        color: var(--gold);
    }
    .source-meta  { font-size: 0.78rem; color: var(--muted); margin: 0.15rem 0 0.5rem; }
    .source-plot  { font-size: 0.88rem; color: #b0adb8; line-height: 1.6; }
    .score-badge {
        display: inline-block;
        background: #1f1f2e;
        border: 1px solid var(--gold-dim);
        color: var(--gold);
        font-size: 0.72rem;
        padding: 0.1rem 0.55rem;
        border-radius: 20px;
        float: right;
    }
    .api-badge {
        display: inline-block;
        background: #1f1f2e;
        border: 1px solid var(--gold-dim);
        color: var(--gold);
        font-size: 0.72rem;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .chat-user {
        background: #1e1e2f;
        border-radius: 12px 12px 4px 12px;
        padding: 0.75rem 1.1rem;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        color: var(--text);
        border: 1px solid #2a2a3e;
        max-width: 85%;
        margin-left: auto;
        text-align: right;
    }
    .chat-bot {
        background: var(--card);
        border-radius: 12px 12px 12px 4px;
        padding: 0.75rem 1.1rem;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        color: var(--text);
        border-left: 3px solid var(--gold);
        max-width: 85%;
    }
    .sb-movie { padding: 0.5rem 0; border-bottom: 1px solid #1e1e28; font-size: 0.85rem; }
    .sb-movie-title { color: var(--gold); font-weight: 500; }
    .sb-movie-meta  { color: var(--muted); font-size: 0.75rem; }
    textarea, input {
        background: var(--card) !important;
        color: var(--text) !important;
        border-color: #2a2a3a !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--gold), #c9922a) !important;
        color: #0a0a0f !important;
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 1.05rem !important;
        letter-spacing: 1.5px !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }
    .stSpinner > div { border-top-color: var(--gold) !important; }
    details summary { color: var(--muted) !important; font-size: 0.85rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_top_movies():
    try:
        r = requests.get(f"{API_URL}/movies/top", params={"limit": 12}, timeout=8)
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def fetch_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=4)
        if r.ok:
            return r.json()   # {"status": "ok", "active_api": "huggingface"}
    except Exception:
        pass
    return None


def ask_api(question: str, top_k: int) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "top_k": top_k},
            timeout=180,
        )
        if r.ok:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the FastAPI backend.\n```\nuvicorn main:app --reload\n```")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
health_data = fetch_health()
active_api  = health_data.get("active_api", "unknown") if health_data else "offline"
alive       = health_data is not None

API_LABELS = {
    "huggingface": "🤗 HuggingFace",
    "mistral":     "⚡ Mistral API",
    "claude":      "🟣 Claude API",
    "openai":      "🟢 OpenAI API",
    "offline":     "⚠️ Offline",
}

with st.sidebar:
    st.markdown(
        "<div style='font-family:Bebas Neue,sans-serif;font-size:1.6rem;"
        "letter-spacing:2px;color:#e8b84b;margin-bottom:0.3rem'>🎬 CineRAG</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='color:#7a7880;font-size:0.75rem;margin-bottom:0.5rem'>"
        f"MongoDB Atlas Vector Search</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span class='api-badge'>{API_LABELS.get(active_api, active_api)}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

    if alive:
        st.success("✅ Backend connected")
    else:
        st.error("⚠️ Backend offline")

    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

    st.markdown("**⚙️ Settings**")
    top_k = st.slider("Movies to retrieve", min_value=1, max_value=10, value=5)

    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

    st.markdown("**🏆 Top-Rated Movies**")
    top_movies = fetch_top_movies()
    if top_movies:
        for m in top_movies:
            rating = ""
            if isinstance(m.get("imdb"), dict):
                rating = m["imdb"].get("rating", "")
            genres = ", ".join((m.get("genres") or [])[:2])
            st.markdown(
                f"""<div class='sb-movie'>
                    <div class='sb-movie-title'>{m.get('title','?')}</div>
                    <div class='sb-movie-meta'>{m.get('year','')} &nbsp;|&nbsp; ⭐ {rating} &nbsp;|&nbsp; {genres}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No movie data available.")

    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.history = []
        st.rerun()

# ─────────────────────────────────────────────
# Main — Hero
# ─────────────────────────────────────────────
st.markdown("<div class='hero-title'>CineRAG</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Movie Intelligence · Ask Anything</div>", unsafe_allow_html=True)
st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Suggested questions
# ─────────────────────────────────────────────
SUGGESTIONS = [
    "What are the best horror movies to watch?",
    "Recommend a sci-fi movie with a twist ending.",
    "Which movies are similar to Inception?",
    "What is a good romantic comedy from the 90s?",
    "Tell me about movies with time travel plots.",
]

st.markdown("**💡 Try asking:**")
cols = st.columns(len(SUGGESTIONS))
clicked_suggestion = None
for col, sug in zip(cols, SUGGESTIONS):
    with col:
        if st.button(sug, key=f"sug_{sug[:20]}", use_container_width=True):
            clicked_suggestion = sug

st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────
if st.session_state.history:
    st.markdown("**🗨️ Conversation**")
    for turn in st.session_state.history:
        st.markdown(f"<div class='chat-user'>🎬 {turn['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bot'>🤖 {turn['answer']}</div>", unsafe_allow_html=True)
        if turn.get("sources"):
            with st.expander(f"📽️ {len(turn['sources'])} source movies used", expanded=False):
                for src in turn["sources"]:
                    genres_str = ", ".join(src["genres"]) if src.get("genres") else "N/A"
                    st.markdown(
                        f"""<div class='source-card'>
                            <span class='score-badge'>score {src['score']:.3f}</span>
                            <div class='source-title'>{src['title']}</div>
                            <div class='source-meta'>{genres_str}</div>
                            <div class='source-plot'>{src['plot']} …</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────
default_question = clicked_suggestion or ""
question = st.text_area(
    "Ask a movie question:",
    value=default_question,
    placeholder="e.g. What are good thriller movies with plot twists?",
    height=100,
    key="question_input",
    label_visibility="collapsed",
)

col_btn, col_tip = st.columns([1, 5])
with col_btn:
    submit = st.button("ASK ▶", use_container_width=True)
with col_tip:
    st.markdown(
        f"<div style='color:#7a7880;font-size:0.78rem;padding-top:0.6rem'>"
        f"Vector search via MongoDB Atlas · Answer generated by "
        f"<b style='color:#e8b84b'>{API_LABELS.get(active_api, active_api)}</b></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Answer
# ─────────────────────────────────────────────
if submit and question.strip():
    if not alive:
        st.error("Backend is not running. Please start the FastAPI server first.")
    else:
        with st.spinner("🎬 Searching movies and generating answer …"):
            t0      = time.time()
            result  = ask_api(question.strip(), top_k)
            elapsed = time.time() - t0

        if result:
            st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
            st.markdown("**🤖 Answer**")
            st.markdown(f"<div class='answer-card'>{result['answer']}</div>", unsafe_allow_html=True)
            st.caption(
                f"⏱ {elapsed:.1f}s · "
                f"{len(result.get('sources', []))} movies retrieved · "
                f"API: {result.get('api_used', active_api)}"
            )

            if result.get("sources"):
                st.markdown("<br>**📽️ Source Movies Used**", unsafe_allow_html=True)
                src_cols = st.columns(min(len(result["sources"]), 3))
                for i, src in enumerate(result["sources"]):
                    with src_cols[i % 3]:
                        genres_str = ", ".join(src["genres"]) if src.get("genres") else "N/A"
                        st.markdown(
                            f"""<div class='source-card'>
                                <span class='score-badge'>score {src['score']:.3f}</span>
                                <div class='source-title'>{src['title']}</div>
                                <div class='source-meta'>{genres_str}</div>
                                <div class='source-plot'>{src['plot']} …</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            st.session_state.history.append({
                "question": question.strip(),
                "answer":   result["answer"],
                "sources":  result.get("sources", []),
            })

elif submit and not question.strip():
    st.warning("Please enter a question before submitting.")