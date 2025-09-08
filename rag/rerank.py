from sentence_transformers import CrossEncoder
import pandas as pd

# Lightweight cross-encoder; downloads at runtime
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    def __init__(self, model_name=MODEL):
        self.ce = CrossEncoder(model_name)

    def rerank(self, query: str, df_hits: pd.DataFrame, topk=10):
        pairs = [(query, t) for t in df_hits["doc_text"].tolist()]
        scores = self.ce.predict(pairs)
        df = df_hits.copy()
        df["ce_score"] = scores
        df = df.sort_values("ce_score", ascending=False).head(topk)
        df["rank"] = range(1, len(df)+1)
        return df
