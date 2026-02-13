# cluster_from_embeddings.py
import json
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from db import get_conn


def load_embeddings(limit):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT e.post_row_id, e.vector_json, p.title
        FROM embeddings e
        JOIN posts p ON e.post_row_id = p.id
        ORDER BY e.post_row_id DESC
        LIMIT %s
    """, (limit,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    ids = []
    vectors = []
    titles = []

    for r in rows:
        ids.append(r["post_row_id"])
        vectors.append(json.loads(r["vector_json"]))
        titles.append(r["title"])

    return ids, np.array(vectors), titles


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
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    ids, X, titles = load_embeddings(args.limit)

    print(f"Loaded {len(ids)} embeddings. Dim={X.shape[1]}")

    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init="auto")
    kmeans.fit(X)

    labels = kmeans.labels_
    update_cluster_ids(ids, labels)

    print("Cluster IDs updated.")

    # 找 centroid 最近
    D = pairwise_distances(X, kmeans.cluster_centers_)
    for c in range(args.k):
        idx = np.argsort(D[:, c])[:3]
        print(f"\nCluster {c}")
        for i in idx:
            print(titles[i])
            print("----")


if __name__ == "__main__":
    main()