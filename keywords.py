import argparse
import numpy as np
from joblib import load
from db import get_conn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_version", type=str, default="tfidf_svd_v2")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    vec_path = f"models/{args.model_version}_vectorizer.joblib"
    vectorizer = load(vec_path)
    terms = np.array(vectorizer.get_feature_names_out())

    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT p.id, p.clean_text, p.cluster_id
        FROM posts p
        JOIN embeddings e ON e.post_row_id = p.id
        WHERE e.model_version = %s AND p.cluster_id IS NOT NULL
    """, (args.model_version,))
    rows = cur.fetchall()
    cur.close()

    texts = [r["clean_text"] for r in rows]
    labels = np.array([int(r["cluster_id"]) for r in rows], dtype=int)

    X = vectorizer.transform(texts)  # sparse TF-IDF


    out = []
    for c in range(args.k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            out.append((c, []))
            continue
        mean_vec = X[idx].mean(axis=0)  # 1 x vocab
        mean_arr = np.asarray(mean_vec).ravel()
        top_idx = np.argsort(mean_arr)[::-1][:args.topn]
        top_terms = [terms[i] for i in top_idx if mean_arr[i] > 0]
        out.append((c, top_terms))

    wcur = conn.cursor()
    for c, top_terms in out:
        wcur.execute("""
            INSERT INTO cluster_topics (model_version, k, cluster_id, top_terms)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              top_terms=VALUES(top_terms),
              created_at=CURRENT_TIMESTAMP
        """, (args.model_version, args.k, c, ", ".join(top_terms)))
    conn.commit()
    wcur.close()
    conn.close()

    for c, top_terms in out:
        print(f"Cluster {c}: {', '.join(top_terms[:10])}")

if __name__ == "__main__":
    main()