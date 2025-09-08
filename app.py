import os, time, json
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from rag.retriever import HybridRetriever
from rag.rerank import Reranker
from rag.query_rewrite import rule_based_parse, llm_rewrite
from rag.generator import generate_explanations
from dotenv import load_dotenv
load_dotenv()

LOG_DB = os.getenv("LOG_DB","./logs/app.db")
engine = create_engine(f"sqlite:///{LOG_DB}", echo=False)
os.makedirs(os.path.dirname(LOG_DB), exist_ok=True)

st.set_page_config(page_title="Rock Recommender RAG", layout="wide")
st.title("🎸 Rock Recommender (RAG)")

@st.cache_resource
def get_components():
    return HybridRetriever(), Reranker()

retr, reranker = get_components()

with st.sidebar:
    st.header("Search")
    q = st.text_input("Describe what you want", placeholder="Energetic 90s alternative with gritty guitars")
    method = st.selectbox("Retrieval method", ["hybrid","bm25","vector","hybrid+rerank"])
    rock_only = st.checkbox("Rock only", value=True)
    tempo = st.slider("Tempo (BPM)", 60, 200, (80, 180))
    popularity = st.slider("Popularity (song)", 0, 100, (0, 100))
    do_rewrite = st.checkbox("Smart query rewrite", value=True)
    topk = st.number_input("Results", min_value=3, max_value=20, value=7, step=1)

if st.button("Recommend") and q.strip():
    # app.py (inside the Recommend button handler)
    t0 = time.time()
    plan = llm_rewrite(q) if do_rewrite else rule_based_parse(q)

    # Normalize keywords to a string
    kw = plan.get("keywords", q)
    if isinstance(kw, (list, tuple)):
        kw = " ".join(map(str, kw))
    elif not isinstance(kw, str):
        kw = str(kw)

    base_method = "hybrid" if "hybrid" in method else method
    hits = retr.search(kw, method=base_method, topk=50).copy()

    # Coerce numerics
    for col in ["tempo", "popularity_song", "is_rock"]:
        if col in hits.columns:
            hits[col] = pd.to_numeric(hits[col], errors="coerce")

    # Filters (lenient with NaNs)
    tempo_low, tempo_high = tempo
    pop_low, pop_high = popularity
    tempo_mask = hits["tempo"].between(tempo_low, tempo_high, inclusive="both") | hits["tempo"].isna()
    pop_mask = hits["popularity_song"].between(pop_low, pop_high, inclusive="both") | hits["popularity_song"].isna()
    mask = tempo_mask & pop_mask
    if rock_only and "is_rock" in hits.columns:
        rock_flag = pd.to_numeric(hits["is_rock"], errors="coerce").fillna(0).astype(int)
        mask &= (rock_flag == 1)
    hits = hits[mask]

    # If empty after filters, retry with BM25 and no filters to avoid dead-ends
    if hits.empty:
        fallback = retr.search(kw, method="bm25", topk=10)
        if not fallback.empty:
            hits = fallback

    # Rerank or truncate
    if "rerank" in method and len(hits) > 0:
        hits = reranker.rerank(q, hits, topk=topk)
    else:
        hits = hits.head(topk)

    # Sanitize columns and ensure required fields (no song_id needed)
    hits = hits.copy()
    hits.columns = [str(c).strip().strip('"').strip("'") for c in hits.columns]

    required_cols = ["name","artists","song_genre","release_date","doc_text"]
    missing = [c for c in required_cols if c not in hits.columns]
    if missing:
        retr_meta = retr.meta.copy()
        retr_meta.columns = [str(c).strip().strip('"').strip("'") for c in retr_meta.columns]
        to_join = [c for c in required_cols if c in retr_meta.columns and c not in hits.columns]
        if to_join:
            hits = hits.join(retr_meta[to_join], how="left")

    # If still empty, bail out nicely
    if hits.empty:
        st.warning("No candidates available (or required columns missing). Try a broader query or relax filters.")
        with st.expander("Debug: retrieved columns"):
            st.write(list(hits.columns))
        st.stop()

    parsed, raw = generate_explanations(q, hits, k_ctx=min(5, len(hits)))
    latency = time.time() - t0

    # ---------- Retrieved candidates (ranked, no index, Unknowns, no commas) ----------
    with st.expander("Retrieved candidates", expanded=True):
        # Ensure we have a copy to format for display
        _hits = hits.copy()

        # Make sure a rank column exists and is first
        if "rank" not in _hits.columns:
            _hits.insert(0, "rank", range(1, len(_hits) + 1))
        else:
            # Move existing 'rank' to first column if needed
            cols = ["rank"] + [c for c in _hits.columns if c != "rank"]
            _hits = _hits[cols]

        # Select columns to show if you already had a curated list; otherwise fallback to some likely columns
        preferred_cols = [
            "rank", "name", "artists", "album", "release_date",
            "year", "tempo", "popularity_song", "is_rock"
        ]
        show_cols = [c for c in preferred_cols if c in _hits.columns]
        if not show_cols:
            show_cols = list(_hits.columns)  # fallback to everything

        # Cell formatter:
        # - NaN/None/empty -> "Unknown"
        # - Ints/floats -> plain strings (no commas); float that is an integer shows as int
        # - Lists/tuples -> comma-joined
        import numpy as np
        def _fmt_cell(x):
            if x is None:
                return "Unknown"
            # pandas NA / numpy nan
            try:
                if pd.isna(x):
                    return "Unknown"
            except Exception:
                pass

            if isinstance(x, (list, tuple)):
                return ", ".join(map(str, x)) if x else "Unknown"

            if isinstance(x, (int, np.integer)):
                return f"{int(x)}"

            if isinstance(x, (float, np.floating)):
                if np.isfinite(x):
                    return f"{int(x)}" if float(x).is_integer() else f"{x}"
                return "Unknown"

            # Everything else as string
            s = str(x).strip()
            return s if s else "Unknown"

        display_df = _hits[show_cols].applymap(_fmt_cell)

        # Hide the side index while keeping our explicit 'rank' column
        try:
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        except TypeError:
            # Streamlit < 1.25 compatibility: fake-hide index by blanking it
            display_df.index = [""] * len(display_df)
            st.dataframe(display_df, use_container_width=True)

    # ---------- Recommendations & Explanations UI ----------
    st.subheader("Explanations")

    if isinstance(parsed, list) and len(parsed) > 0:
        for i, item in enumerate(parsed, start=1):
            name = (item.get("name") or "Unknown")
            artists = (item.get("artists") or "Unknown")
            if isinstance(artists, (list, tuple)):
                artists = ", ".join(map(str, artists)) if artists else "Unknown"
            reason = (item.get("reason") or "No explanation provided.")

            st.markdown(f"**{i} - {name}, {artists}**")
            st.markdown(reason)
            st.markdown("")  # spacing between entries
    else:
        st.warning("No valid explanations were returned.")
        with st.expander("Raw model output (debug)"):
            st.code(raw)


