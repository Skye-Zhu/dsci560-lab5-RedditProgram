# preprocess.py
import re
from db import get_conn

def clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"<[^>]+>", " ", s)          # HTML tags
    s = re.sub(r"http\S+", " ", s)          # URLs
    s = re.sub(r"[\r\n\t]+", " ", s)        # whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main(limit=2000):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    # 只处理非广告
    cur.execute("""
        SELECT id, title, body
        FROM posts
        WHERE (clean_text IS NULL OR clean_text = '')
          AND (is_ad IS NULL OR is_ad = 0)
        ORDER BY id DESC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()

    upd = conn.cursor()
    for r in rows:
        text = f"{r.get('title','')} {r.get('body','')}"
        ct = clean_text(text)
        upd.execute("UPDATE posts SET clean_text=%s WHERE id=%s", (ct, r["id"]))

    conn.commit()
    upd.close()
    cur.close()
    conn.close()

    print(f"Preprocessed {len(rows)} rows.")

if __name__ == "__main__":
    main()