# rag/query_rewrite.py
import re, json, os
from typing import Dict
from dotenv import load_dotenv
load_dotenv()

def rule_based_parse(query: str) -> Dict:
    q = query if isinstance(query, str) else str(query)
    plan = {"keywords": q, "filters": {}}
    ql = q.lower()
    m = re.search(r"(19\d{2}|20\d{2})s", ql)
    if m: plan["filters"]["era"] = m.group(1)
    m = re.search(r"(\d{2,3})\s?bpm", ql)
    if m: plan["filters"]["tempo"] = int(m.group(1))
    if any(w in ql for w in ["melancholy","sad"]): plan["filters"]["valence_max"] = 0.4
    if any(w in ql for w in ["energetic","high energy"]): plan["filters"]["energy_min"] = 0.7
    if "rock only" in ql: plan["filters"]["is_rock"] = 1
    return plan

def _normalize_plan(obj, original_query: str) -> Dict:
    # Accept dict or list; ensure string keywords
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return {"keywords": original_query, "filters": {}}
    kw = obj.get("keywords", original_query)
    if isinstance(kw, (list, tuple)):
        kw = " ".join(map(str, kw))
    elif not isinstance(kw, str):
        kw = str(kw)
    filters = obj.get("filters", {})
    return {"keywords": kw, "filters": filters if isinstance(filters, dict) else {}}

def llm_rewrite(query: str) -> Dict:
    backend = os.getenv("LLM_BACKEND", "openai")
    try:
        if backend == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            sys = "Return only JSON with keys: keywords (string) and filters (object)."
            out = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                messages=[{"role":"system","content":sys},
                          {"role":"user","content":f"Rewrite this music search request into concise keywords and optional filters (era,bpm,energy/valence,is_rock):\n\n{query}"}],
                temperature=0
            )
            txt = out.choices[0].message.content
            obj = json.loads(txt)
            return _normalize_plan(obj, query)
        else:
            # Fallback: no LLM/ollama here; use rules
            return rule_based_parse(query)
    except Exception:
        return rule_based_parse(query)
