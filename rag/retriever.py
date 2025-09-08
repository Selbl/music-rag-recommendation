# rag/retriever.py
import os, re, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RRF_K = 60

class HybridRetriever:
    def __init__(self):
        self.meta = pd.read_parquet(DATA_DIR / "meta.parquet")
        self.texts = self.meta["doc_text"].tolist()
        self.emb_model = SentenceTransformer(EMB_MODEL_NAME)
        self.index = faiss.read_index(str(DATA_DIR / "index.faiss"))
        with open(DATA_DIR / "bm25.pkl", "rb") as f:
            obj = pickle.load(f)
        self.bm25 = obj["bm25"]
        self.corpus_tokens = obj["corpus_tokens"]

    @staticmethod
    def _tok(s): 
        return re.findall(r"[a-z0-9]+", str(s).lower())

    @staticmethod
    def _normalize_query(q):
        """Always return a string query."""
        if q is None:
            return ""
        if isinstance(q, str):
            return q
        # If list/tuple (including a 1-tuple), flatten to space-joined string
        if isinstance(q, (list, tuple)):
            return " ".join(map(str, q))
        if isinstance(q, dict):
            # try common key; otherwise join all values
            return str(q.get("keywords") or " ".join(map(str, q.values())))
        return str(q)

    def vector_search(self, query, topk=200):
        qtext = self._normalize_query(query)
        if not qtext.strip():
            return np.array([], dtype=int), np.array([], dtype=float)
        q = self.emb_model.encode([qtext], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        D, I = self.index.search(q, topk)
        return I[0], D[0]

    def bm25_search(self, query, topk=200):
        qtext = self._normalize_query(query)
        if not qtext.strip():
            return np.array([], dtype=int), np.array([], dtype=float)
        scores = self.bm25.get_scores(self._tok(qtext))
        idx = np.argsort(scores)[::-1][:topk]
        return idx, scores[idx]

    def rrf_fuse(self, list_a, list_b, k=RRF_K, topk=50):
        ranks = {}
        for rank, idx in enumerate(list_a):
            ranks[idx] = ranks.get(idx, 0) + 1.0/(k+rank+1)
        for rank, idx in enumerate(list_b):
            ranks[idx] = ranks.get(idx, 0) + 1.0/(k+rank+1)
        fused = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:topk]
        return [i for i,_ in fused], [s for _,s in fused]

    def search(self, query, method="hybrid", topk=10):
        if method == "bm25":
            idx,_ = self.bm25_search(query, topk=topk)
        elif method == "vector":
            idx,_ = self.vector_search(query, topk=topk)
        else:
            i_v,_ = self.vector_search(query, topk=200)
            i_b,_ = self.bm25_search(query, topk=200)
            idx,_ = self.rrf_fuse(i_v, i_b, topk=max(topk,50))
        if len(idx) == 0:
            return self.meta.iloc[:0].copy()
        res = self.meta.iloc[idx[:topk]].copy()
        res["rank"] = range(1, len(res)+1)
        return res
