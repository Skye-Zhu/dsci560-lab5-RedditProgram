import time
import subprocess

def run(cmd: str):
    print(f"\n[RUN] {cmd}")
    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def loop(interval_minutes: int, scrape_n: int = 200, embed_limit: int = 2000):
    interval_sec = interval_minutes * 60
    while True:
        try:
            run(f"python scraper.py {scrape_n} --subs cybersecurity,netsec --sleep 1.5")
            run("python preprocess.py")
            run(f"python embed.py --limit {embed_limit} --dim 128 --model_version tfidf_svd_v2")
            run("python cluster_from_embeddings.py --k 8 --limit 2000 --model_version tfidf_svd_v2")
            run("python keywords.py --model_version tfidf_svd_v2 --k 8 --topn 10")
            run("python visualize.py --model_version tfidf_svd_v2 --out cluster_pca_v2.png")
            print(f"\n[OK] Update complete. Sleeping {interval_minutes} minutes...")
        except Exception as e:
            print(f"\n[ERROR] {e}\nSleeping 60 seconds then retry...")
            time.sleep(60)
            continue

        time.sleep(interval_sec)