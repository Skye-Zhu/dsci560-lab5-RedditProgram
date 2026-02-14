# scraper_old_reddit.py
import re
import time
import argparse
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

from db import get_conn

UA = "DSCI560-Lab5-OldRedditScraper/1.0 (contact: your_email@usc.edu)"
BASE = "https://old.reddit.com"

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def mask_author(author: str) -> str:
    if not author:
        return "user_unknown"
    return "user_" + str(abs(hash(author)) % 1000000).zfill(6)

def is_promoted(thing) -> bool:
    txt = thing.get_text(" ", strip=True).lower()
    return ("promoted" in txt) or ("advertisement" in txt) or ("sponsored" in txt)

def parse_created_at(thing) -> Optional[datetime]:
    time_tag = thing.select_one("time")
    if time_tag and time_tag.has_attr("datetime"):
        try:
            dt = dtparser.isoparse(time_tag["datetime"])
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:
            return None
    return None

def extract_post_id(thing) -> Optional[str]:
    fullname = thing.get("data-fullname")
    if fullname and fullname.startswith("t3_"):
        return fullname[3:]
    a = thing.select_one("a.comments")
    if a and a.has_attr("href"):
        m = re.search(r"/comments/([a-z0-9]+)/", a["href"])
        if m:
            return m.group(1)
    return None

def extract_post_url(thing) -> Optional[str]:
    a = thing.select_one("a.title")
    if a and a.has_attr("href"):
        href = a["href"]
        if href.startswith("/r/"):
            return BASE + href
        return href
    return None

def extract_image_url(post_url: str) -> Optional[str]:

    if not post_url:
        return None
    u = post_url.lower()
    if any(u.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
        return post_url
    if "i.redd.it/" in u or "preview.redd.it/" in u:
        return post_url
    return None

def fetch_page(subreddit: str, after: Optional[str]) -> Tuple[str, Optional[str]]:
    url = f"{BASE}/r/{subreddit}/new/"
    params = {}
    if after:
        params["after"] = after

    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    html = r.text

    soup = BeautifulSoup(html, "html.parser")
    next_a = soup.select_one("span.next-button a")
    next_after = None
    if next_a and next_a.has_attr("href"):
        m = re.search(r"after=([^&]+)", next_a["href"])
        if m:
            next_after = m.group(1)
    return html, next_after

def parse_posts(html: str, subreddit: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    things = soup.select("div.thing")
    posts = []

    for thing in things:
        if thing.get("data-domain") == "self":
            pass

        post_id = extract_post_id(thing)
        if not post_id:
            continue

        title_tag = thing.select_one("a.title")
        title = clean_text(title_tag.get_text()) if title_tag else ""

        author_tag = thing.select_one("a.author")
        author = clean_text(author_tag.get_text()) if author_tag else ""
        author_masked = mask_author(author)

        created_at = parse_created_at(thing)

        post_url = extract_post_url(thing) or ""
        image_url = extract_image_url(post_url)

        ad = is_promoted(thing)

        body = ""

        posts.append({
            "post_id": post_id,
            "subreddit": subreddit,
            "title": title,
            "body": body,
            "clean_text": clean_text(f"{title} {body}"),
            "author_masked": author_masked,
            "created_at": created_at,
            "post_url": post_url,
            "image_url": image_url,
            "is_ad": ad
        })

    return posts

def upsert_posts(rows: List[Dict]) -> int:
    if not rows:
        return 0
    conn = get_conn()
    cur = conn.cursor()

    sql = """
    INSERT INTO posts (post_id, subreddit, title, body, clean_text, author_masked, created_at, post_url, image_url, is_ad)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
      title=VALUES(title),
      body=VALUES(body),
      clean_text=VALUES(clean_text),
      author_masked=VALUES(author_masked),
      created_at=VALUES(created_at),
      post_url=VALUES(post_url),
      image_url=VALUES(image_url),
      is_ad=VALUES(is_ad);
    """

    n = 0
    for r in rows:
        cur.execute(sql, (
            r["post_id"], r["subreddit"], r["title"], r["body"], r["clean_text"],
            r["author_masked"], r["created_at"], r["post_url"], r["image_url"], r["is_ad"]
        ))
        n += 1

    cur.close()
    conn.close()
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("num_posts", type=int, help="total number of posts to fetch (across all subreddits)")
    ap.add_argument("--subs", type=str, default="cybersecurity",
                    help="comma-separated subreddits, e.g. cybersecurity,netsec,hacking")
    ap.add_argument("--sleep", type=float, default=1.2, help="sleep seconds between page requests")
    ap.add_argument("--max_pages_per_sub", type=int, default=200, help="safety cap")
    args = ap.parse_args()

    subs = [s.strip() for s in args.subs.split(",") if s.strip()]
    target = args.num_posts

    total_saved = 0

    for sub in subs:
        after = None
        pages = 0

        while total_saved < target and pages < args.max_pages_per_sub:
            pages += 1

            try:
                html, next_after = fetch_page(sub, after)
            except requests.HTTPError as e:
                print(f"[{sub}] HTTPError: {e} | after={after} -> sleeping 10s")
                time.sleep(10)
                continue
            except Exception as e:
                print(f"[{sub}] Error: {e} | after={after} -> sleeping 10s")
                time.sleep(10)
                continue

            posts = parse_posts(html, sub)
            saved = upsert_posts(posts)

            total_saved += saved
            print(f"[{sub}] page={pages} saved={saved} total_saved={total_saved} next_after={next_after}")

            if not next_after or next_after == after:
                break
            after = next_after

            time.sleep(args.sleep)

        if total_saved >= target:
            break

    print(f"Done. Total saved/updated rows: {total_saved}")

if __name__ == "__main__":
    main()