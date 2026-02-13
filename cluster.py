# cluster.py
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from db import get_conn


def load_posts(limit: int):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT id, clean_text, title
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


def update_cluster_ids(ids, labels):
    conn = get_conn()
    cur = conn.cursor()
    for pid, lab in zip(ids, labels):
        cur.execute("UPDATE posts SET cluster_id=%s WHERE id=%s", (int(lab), int(pid)))
    conn.commit()
    cur.close()
    conn.close()


def print_cluster_representatives(texts, titles, labels, centers, X, topn=3):
    # 对每个 cluster：找离 centroid 最近的 topn 条
    D = pairwise_distances(X, centers)  # shape: (n_docs, k)
    k = centers.shape[0]
    for c in range(k):
        idx = np.argsort(D[:, c])[:topn]
        print(f"\nCluster {c} (top {topn} closest to centroid)")
        for j, i in enumerate(idx, 1):
            print(f"{j}. {titles[i][:120]}")
            print(f"   {texts[i][:220]}")
            print("   ---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="number of clusters")
    ap.add_argument("--limit", type=int, default=5000, help="max docs to cluster")
    args = ap.parse_args()

    rows = load_posts(args.limit)
    if not rows:
        print("No rows found. (clean_text may be empty?)")
        return

    ids = [r["id"] for r in rows]
    texts = [r["clean_text"] for r in rows]
    titles = [r.get("title", "") or "" for r in rows]

    print(f"Loaded {len(texts)} documents.")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init="auto")
    kmeans.fit(X)

    labels = kmeans.labels_
    update_cluster_ids(ids, labels)

    print("Clusters assigned and saved to DB.")

    print_cluster_representatives(
        texts=texts,
        titles=titles,
        labels=labels,
        centers=kmeans.cluster_centers_,
        X=X,
        topn=3,
    )


if __name__ == "__main__":
    main()