import argparse
import json
import numpy as np
from joblib import load
from sklearn.metrics.pairwise import cosine_distances
from db import get_conn

def embed_query(text: str, model_version: str):
    vectorizer = load(f"models/{model_version}_vectorizer.joblib")
    svd = load(f"models/{model_version}_svd.joblib")

    X = vectorizer.transform([text])
    z = svd.transform(X)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return z[0]  # (dim,)

def load_centroids(model_version: str, k: int):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT e.vector_json, p.cluster_id
        FROM embeddings e
        JOIN posts p ON p.id = e.post_row_id
        WHERE e.model_version=%s AND p.cluster_id IS NOT NULL
    """, (model_version,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    dim = len(json.loads(rows[0]["vector_json"]))
    sums = np.zeros((k, dim), dtype=float)
    cnt = np.zeros(k, dtype=int)

    for r in rows:
        c = int(r["cluster_id"])
        if 0 <= c < k:
            v = np.array(json.loads(r["vector_json"]), dtype=float)
            sums[c] += v
            cnt[c] += 1

    centroids = np.zeros_like(sums)
    for c in range(k):
        if cnt[c] > 0:
            centroids[c] = sums[c] / cnt[c]
            centroids[c] = centroids[c] / (np.linalg.norm(centroids[c]) + 1e-12)
    return centroids, cnt

def load_cluster_keywords(model_version: str, k: int, cluster_id: int):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT top_terms
        FROM cluster_topics
        WHERE model_version=%s AND k=%s AND cluster_id=%s
    """, (model_version, k, cluster_id))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row["top_terms"] if row else ""

def load_representative_posts(cluster_id: int, n: int = 5):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT title, post_url
        FROM posts
        WHERE cluster_id=%s
        ORDER BY created_at DESC
        LIMIT %s
    """, (cluster_id, n))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", type=str, help="query text")
    ap.add_argument("--model_version", type=str, default="tfidf_svd_v2")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    q = embed_query(args.text, args.model_version)
    centroids, cnt = load_centroids(args.model_version, args.k)

    d = cosine_distances([q], centroids)[0]
    best = int(np.argmin(d))

    print(f"\nBest cluster: {best}  (size={cnt[best]})")
    print("Top terms:", load_cluster_keywords(args.model_version, args.k, best))

    reps = load_representative_posts(best, n=5)
    print("\nRecent posts in this cluster:")
    for i, r in enumerate(reps, 1):
        print(f"{i}. {r['title']}")
        print(f"   {r['post_url']}")

if __name__ == "__main__":
    main()