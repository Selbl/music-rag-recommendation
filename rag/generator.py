# rag/generator.py
import os
import pandas as pd
from .utils import extract_json_payload  # keep your robust JSON extractor

TEMPLATE = """You are a music recommender that MUST ground your explanations in the provided CONTEXT.
Return ONLY a JSON array where each item has keys: name (string), artists (string), reason (string).
- Do not include any other keys.
- The "reason" must cite grounded facts in square brackets like [field: value] or a short lyric excerpt.
- Keep each reason under 2 sentences.

USER_QUERY:
{query}

CONTEXT (top-{k} candidates):
{context}
"""

def _safe_val(row, key, default=""):
    try:
        v = row.get(key, default)
    except Exception:
        v = default
    return "" if v is None else v

def _make_context(df: pd.DataFrame, k=5) -> str:
    # No reliance on song_id, only human-readable fields
    rows = []
    use = df.head(k)
    for _, r in use.iterrows():
        name  = _safe_val(r, "name", "?")
        arts  = _safe_val(r, "artists", "?")
        genre = _safe_val(r, "song_genre", "")
        rel   = _safe_val(r, "release_date", "")
        text  = _safe_val(r, "doc_text", "")[:600]
        rows.append(f"{name} — {arts} | {genre} | {rel}\n{text}")
    return "\n---\n".join(rows)

def generate_explanations(query: str, df_hits: pd.DataFrame, k_ctx=5):
    prompt = TEMPLATE.format(query=query, k=k_ctx, context=_make_context(df_hits, k_ctx))
    backend = os.getenv("LLM_BACKEND", "openai")

    if backend == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        out = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
            messages=[
                {"role":"system","content":"You are a helpful, precise recommender that responds in pure JSON."},
                {"role":"user","content":prompt}
            ],
            temperature=0.3
        )
        text = out.choices[0].message.content
    else:
        import subprocess
        model = os.getenv("OLLAMA_MODEL","llama3")
        text = subprocess.check_output(["ollama","run",model,prompt]).decode("utf-8")

    try:
        parsed = extract_json_payload(text)  # returns list/dict or raises
    except Exception:
        parsed = None

    return parsed, text
