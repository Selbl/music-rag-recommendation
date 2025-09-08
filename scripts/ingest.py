import os, re, json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

DATA_CSV = os.getenv("DATA_CSV", "./rock_songs_full.csv")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMB_BATCH = int(os.getenv("EMB_BATCH", "128"))

TEXT_COLS = ["name", "artists", "song_genre", "lyrics"]

def build_doc(row):
    name = str(row.get("name", ""))[:200]
    artists = str(row.get("artists", ""))[:200]
    genre = str(row.get("song_genre", ""))
    year = str(row.get("release_date", ""))[:10]
    lyr = str(row.get("lyrics", "")) or ""
    lyr = re.sub(r"\s+", " ", lyr).strip()
    # keep a compact snippet for retrieval (avoid entire lyric walls)
    lyr_snip = lyr[:1200]
    return f"Title: {name}\nArtists: {artists}\nGenre: {genre}\nRelease: {year}\nLyrics: {lyr_snip}"

def main():

    df = pd.read_csv(DATA_CSV)

    # Normalize dtypes early
    numeric_cols = [
        "tempo","popularity_song","popularity_artist","is_rock","key","mode",
        "acousticness","danceability","energy","instrumentalness","liveness",
        "loudness","speechiness","valence"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If is_rock is boolean-ish, make 0/1
    if "is_rock" in df.columns:
        df["is_rock"] = df["is_rock"].fillna(0)
        # If values look like True/False strings, this will coerce to NaN first; fillna, then cast
        df["is_rock"] = pd.to_numeric(df["is_rock"], errors="coerce").fillna(0).astype(int)

    df = pd.read_csv(DATA_CSV)
    df["doc_text"] = df.apply(build_doc, axis=1)
    # keep minimal metadata alongside id
    if "song_id" not in df.columns:
        df["song_id"] = np.arange(len(df))
    meta = df[["song_id","name","artists","song_genre","release_date","tempo","popularity_song","is_rock","doc_text"]].copy()
    meta.to_parquet(DATA_DIR / "meta.parquet", index=False)

    # ----- Vector index (FAISS) -----
    print("Loading embedding model:", EMB_MODEL_NAME)
    emb_model = SentenceTransformer(EMB_MODEL_NAME)
    texts = meta["doc_text"].tolist()

    embs = []
    for i in tqdm(range(0, len(texts), EMB_BATCH), desc="Embedding"):
        batch = texts[i:i+EMB_BATCH]
        embs.append(emb_model.encode(batch, normalize_embeddings=True, convert_to_numpy=True))
    embs = np.vstack(embs).astype("float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)        # cosine if vectors are normalized
    index.add(embs)
    faiss.write_index(index, str(DATA_DIR / "index.faiss"))
    np.save(DATA_DIR / "embeddings.npy", embs)

    # ----- Lexical index (BM25) -----
    def tok(s): return re.findall(r"[a-z0-9]+", str(s).lower())
    corpus_tokens = [tok(t) for t in meta["doc_text"].tolist()]
    bm25 = BM25Okapi(corpus_tokens)
    # persist bm25 in a light way
    import pickle
    with open(DATA_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "corpus_tokens": corpus_tokens}, f)

    print("Saved:")
    print(" - data/index.faiss")
    print(" - data/embeddings.npy")
    print(" - data/meta.parquet")
    print(" - data/bm25.pkl")

if __name__ == "__main__":
    main()
