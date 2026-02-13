# cluster_from_embeddings.py
import json
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from db import get_conn


def load_embeddings(limit: int, model_version: str):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT e.post_row_id, e.vector_json, p.title
        FROM embeddings e
        JOIN posts p ON e.post_row_id = p.id
        WHERE e.model_version = %s
        ORDER BY e.post_row_id DESC
        LIMIT %s
    """, (model_version, limit))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    ids, vectors, titles = [], [], []
    for r in rows:
        ids.append(r["post_row_id"])
        vectors.append(json.loads(r["vector_json"]))
        titles.append(r.get("title", "") or "")

    # 强制变成规则矩阵（若混维度会在这里立刻报错，便于定位）
    X = np.array(vectors, dtype=float)
    return ids, X, titles


def update_cluster_ids(ids, labels):
    conn = get_conn()
    cur = conn.cursor()
    for pid, lab in zip(ids, labels):
        cur.execute("UPDATE posts SET cluster_id=%s WHERE id=%s", (int(lab), int(pid)))
    conn.commit()
    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--model_version", type=str, default="tfidf_svd_v2")
    parser.add_argument("--topn", type=int, default=3)
    args = parser.parse_args()

    ids, X, titles = load_embeddings(args.limit, args.model_version)
    print(f"Loaded {len(ids)} embeddings. Dim={X.shape[1]} (model_version={args.model_version})")

    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init="auto")
    kmeans.fit(X)

    labels = kmeans.labels_
    update_cluster_ids(ids, labels)
    print("Cluster IDs updated.")

    # 找 centroid 最近的帖子
    D = pairwise_distances(X, kmeans.cluster_centers_)
    for c in range(args.k):
        idx = np.argsort(D[:, c])[:args.topn]
        print(f"\nCluster {c}")
        for i in idx:
            print(titles[i])
            print("----")


if __name__ == "__main__":
    main()