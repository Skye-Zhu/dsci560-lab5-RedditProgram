import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from db import get_conn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_version", type=str, default="tfidf_svd_v2")
    ap.add_argument("--out", type=str, default="cluster_pca.png")
    args = ap.parse_args()

    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT e.vector_json, p.cluster_id
        FROM embeddings e
        JOIN posts p ON p.id = e.post_row_id
        WHERE e.model_version = %s AND p.cluster_id IS NOT NULL
    """, (args.model_version,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    X = np.array([json.loads(r["vector_json"]) for r in rows], dtype=float)
    y = np.array([int(r["cluster_id"]) for r in rows], dtype=int)

    Z = PCA(n_components=2, random_state=42).fit_transform(X)

    plt.figure()
    plt.scatter(Z[:,0], Z[:,1], c=y, s=10)
    plt.title(f"PCA of Embeddings (model={args.model_version})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()