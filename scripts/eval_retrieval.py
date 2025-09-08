import os, json, time, argparse
import numpy as np
import pandas as pd
from rag.retriever import HybridRetriever
from rag.rerank import Reranker

def metrics(ranks, k_list=(1,3,5,10)):
    out = {}
    for k in k_list:
        out[f"Recall@{k}"] = np.mean([1.0 if r <= k else 0.0 for r in ranks])
    # MRR@10
    out["MRR@10"] = np.mean([1.0/r if r<=10 else 0.0 for r in ranks])
    return out

def main(args):
    gt = pd.read_csv(args.ground_truth)  # columns: query, song_id
    retr = HybridRetriever()
    reranker = Reranker() if args.use_rerank else None

    results = []
    for method in ["bm25","vector","hybrid"]:
        ranks = []
        for _, row in gt.iterrows():
            q, target = row["query"], row["song_id"]
            hits = retr.search(q, method=method, topk=50)
            if reranker and method in ("hybrid","vector"):
                hits = reranker.rerank(q, hits, topk=10)
            rank = hits["song_id"].tolist()
            # 1-indexed rank of target (inf if not present)
            r = rank.index(target)+1 if target in rank else float("inf")
            ranks.append(r)
        m = metrics(ranks)
        results.append({"method": method + ("+ce" if reranker and method in ("hybrid","vector") else ""), **m})
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(os.path.dirname(args.ground_truth), "retrieval_eval.csv"), index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground_truth", required=True, help="CSV with columns: query,song_id")
    ap.add_argument("--use_rerank", action="store_true")
    args = ap.parse_args()
    main(args)
