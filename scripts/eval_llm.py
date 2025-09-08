# Simple LLM-as-a-judge for helpfulness & groundedness
import os, json, argparse, pandas as pd
from rag.retriever import HybridRetriever
from rag.rerank import Reranker
from rag.generator import generate_explanations
from openai import OpenAI

JUDGE_PROMPT = """You are a strict evaluator.
Given USER_QUERY, CONTEXT (retrieved snippets), and ASSISTANT_ANSWER (JSON list), score:
- helpfulness (0..5): how well it addresses the user's intent
- groundedness (0..5): are claims traceable to CONTEXT facts?
Return JSON: {"helpfulness": x, "groundedness": y, "notes": "..."}"""

def main(args):
    client = OpenAI()
    test = pd.read_csv(args.eval_queries)  # columns: query
    retr = HybridRetriever()
    reranker = Reranker()

    rows = []
    for _, r in test.iterrows():
        q = r["query"]
        hits = retr.search(q, method="hybrid", topk=50)
        hits = reranker.rerank(q, hits, topk=5)
        ans = generate_explanations(q, hits, k_ctx=5)

        context = "\n".join(hits["doc_text"].tolist())
        messages = [
            {"role":"system","content":JUDGE_PROMPT},
            {"role":"user","content": f"USER_QUERY:\n{q}\n\nCONTEXT:\n{context[:4000]}\n\nASSISTANT_ANSWER:\n{ans[:4000]}"},
        ]
        out = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)
        try:
            js = json.loads(out.choices[0].message.content)
        except:
            js = {"helpfulness": None, "groundedness": None, "notes":"parse_error"}
        rows.append({"query": q, **js})
    df = pd.DataFrame(rows)
    print(df.describe())
    df.to_csv("llm_eval.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_queries", required=True)
    args = ap.parse_args()
    main(args)
