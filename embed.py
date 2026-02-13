import os
import json
import argparse
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from db import get_conn


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def load_posts(limit: int):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT id, clean_text
        FROM posts
        WHERE clean_text IS NOT NULL AND clean_text != ''
          AND (is_ad IS NULL OR is_ad = 0)
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def upsert_embedding(rows, vectors, method: str, model_version: str):
    conn = get_conn()
    cur = conn.cursor()

    sql = """
    INSERT INTO embeddings (post_row_id, method, dim, vector_json, model_version)
    VALUES (%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      method=VALUES(method),
      dim=VALUES(dim),
      vector_json=VALUES(vector_json),
      model_version=VALUES(model_version),
      created_at=CURRENT_TIMESTAMP
    """

    dim = int(vectors.shape[1])
    for r, v in zip(rows, vectors):
        v_list = v.astype(float).tolist()
        cur.execute(sql, (int(r["id"]), method, dim, json.dumps(v_list), model_version))

    conn.commit()
    cur.close()
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5000, help="max docs to embed")
    ap.add_argument("--dim", type=int, default=128, help="embedding dimension after SVD")
    ap.add_argument("--max_features", type=int, default=5000, help="TF-IDF max vocab size")
    ap.add_argument("--model_version", type=str, default="tfidf_svd_v1", help="tag for DB/model files")
    args = ap.parse_args()

    rows = load_posts(args.limit)
    if len(rows) < 2:
        print(f"Not enough documents to embed: {len(rows)}")
        return

    texts = [r["clean_text"] for r in rows]
    print(f"Loaded {len(texts)} documents for embedding.")


    vectorizer = TfidfVectorizer(stop_words="english", max_features=args.max_features, min_df=5, max_df=0.7)
    X = vectorizer.fit_transform(texts)

    max_possible = min(X.shape[0] - 1, X.shape[1] - 1)
    dim = min(args.dim, max_possible)
    if dim < 2:
        print(f"Cannot compute SVD with dim={args.dim} for X shape {X.shape}. Try more data.")
        return

    svd = TruncatedSVD(n_components=dim, random_state=42)
    Z = svd.fit_transform(X)  # (n_docs, dim)

    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    dump(vectorizer, MODEL_DIR / f"{args.model_version}_vectorizer.joblib")
    dump(svd, MODEL_DIR / f"{args.model_version}_svd.joblib")


    upsert_embedding(rows, Z, method="tfidf+svd", model_version=args.model_version)

    print(f"Saved embeddings to DB. dim={dim}, model_version={args.model_version}")
    print(f"Model files saved under: {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    main()