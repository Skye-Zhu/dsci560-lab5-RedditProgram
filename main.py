# main.py
import argparse
import threading
import subprocess
import shlex
import time


def run(cmd: str):
    """Run a shell command, print it, and raise if failed."""
    print(f"\n[RUN] {cmd}")
    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {cmd}")


def updater_loop(
    interval_minutes: int,
    scrape_n: int,
    subs: str,
    sleep_sec: float,
    embed_limit: int,
    cluster_limit: int,
    model_version: str,
    k: int,
    topn_terms: int,
    pca_out: str,
):
    interval_sec = interval_minutes * 60

    while True:
        start = time.time()
        try:
            # 1) scrape (incremental)
            run(f"python scraper.py {scrape_n} --subs {subs} --sleep {sleep_sec}")

            # 2) preprocess
            run("python preprocess.py")

            # 3) embed (rebuild embeddings for latest window)
            run(
                f"python embed.py --limit {embed_limit} --dim 128 "
                f"--model_version {model_version}"
            )

            # 4) cluster from embeddings
            run(
                f"python cluster_from_embeddings.py --k {k} --limit {cluster_limit} "
                f"--model_version {model_version}"
            )

            # 5) keywords/topics
            run(
                f"python keywords.py --model_version {model_version} --k {k} --topn {topn_terms}"
            )

            # 6) visualization
            run(f"python visualize.py --model_version {model_version} --out {pca_out}")

            elapsed = time.time() - start
            print(f"\n[OK] Update complete in {elapsed:.1f}s. Next update in {interval_minutes} min.")

        except Exception as e:
            print(f"\n[ERROR] {e}")
            print("[INFO] Sleeping 60s then retry...")
            time.sleep(60)
            continue

        time.sleep(interval_sec)


def main():
    ap = argparse.ArgumentParser()

    # required: interval minutes
    ap.add_argument("interval", type=int, help="update interval in minutes (e.g., 5)")

    # scraping controls
    ap.add_argument("--scrape_n", type=int, default=200, help="how many posts to scrape each cycle")
    ap.add_argument(
        "--subs",
        type=str,
        default="cybersecurity,netsec",
        help="comma-separated subreddits, e.g. cybersecurity,netsec,hacking",
    )
    ap.add_argument("--sleep", type=float, default=1.5, help="sleep seconds between page requests")

    # pipeline controls
    ap.add_argument("--model_version", type=str, default="tfidf_svd_v2")
    ap.add_argument("--k", type=int, default=8, help="number of clusters")
    ap.add_argument("--embed_limit", type=int, default=2000, help="max docs to embed each cycle")
    ap.add_argument("--cluster_limit", type=int, default=2000, help="max docs to cluster each cycle")
    ap.add_argument("--topn_terms", type=int, default=10, help="top keywords per cluster")
    ap.add_argument("--pca_out", type=str, default="cluster_pca_v2.png", help="output PCA image path")

    args = ap.parse_args()

    # start updater thread
    t = threading.Thread(
        target=updater_loop,
        args=(
            args.interval,
            args.scrape_n,
            args.subs,
            args.sleep,
            args.embed_limit,
            args.cluster_limit,
            args.model_version,
            args.k,
            args.topn_terms,
            args.pca_out,
        ),
        daemon=True,
    )
    t.start()

    print("\nAutomation started")
    print(f"- interval: {args.interval} min")
    print(f"- scrape_n: {args.scrape_n}")
    print(f"- subs: {args.subs}")
    print(f"- model_version: {args.model_version}, k={args.k}")
    print("\nType a query to find the closest cluster.")
    print("Commands:")
    print("  :exit        quit")
    print("  :help        show help")
    print("  :pca         print PCA image path\n")

    while True:
        q = input("> ").strip()
        if not q:
            continue

        if q.lower() in (":exit", ":quit", "exit", "quit"):
            break

        if q.lower() == ":help":
            print("Type any text to query the closest cluster.")
            print("Use :pca to see visualization path, :exit to quit.")
            continue

        if q.lower() == ":pca":
            print(f"PCA image: {args.pca_out}")
            continue

        # query mode: call query.py
        safe_q = q.replace('"', '\\"')
        cmd = (
            f'python query.py "{safe_q}" '
            f"--model_version {args.model_version} --k {args.k}"
        )
        subprocess.run(cmd, shell=True)

    print("Bye.")


if __name__ == "__main__":
    main()