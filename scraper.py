"""
Full scraper for Camera dei Deputati - XIX Legislature roll-call votes.
Run: python src/scraper.py

Features:
  - Discovers all sessions with votes (probes 1-700), cached after first run
  - Dynamically finds all votes per session
  - Retries on timeout (up to 3 attempts)
  - Saves progress incrementally (resume-safe)
  - Outputs: data/raw/camera_votes_full.csv

Estimated runtime: 8-12 hours for full legislature.
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
import re
import os
import sys
import json

BASE = "https://documenti.camera.it/votazioni/votazionitutte"
LEGISLATURE = 19
DELAY = 1.0
MAX_RETRIES = 3
RETRY_BACKOFF = 5
TIMEOUT = 45
MAX_SESSION = 700

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research project - parliamentary voting analysis)"
}

OUTPUT_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "camera_votes_full.csv")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, ".scraper_progress.txt")
SESSIONS_FILE = os.path.join(OUTPUT_DIR, ".sessions_cache.json")

FIELDNAMES = [
    "legislature", "session", "vote_num", "date",
    "vote_type", "title", "description", "deputy_name", "group", "vote"
]


def fetch_with_retry(url):
    """Fetch a URL with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF * attempt
            if attempt < MAX_RETRIES:
                print(f" TIMEOUT (attempt {attempt}/{MAX_RETRIES}, retrying in {wait}s)")
                time.sleep(wait)
            else:
                print(f" TIMEOUT (gave up after {MAX_RETRIES} attempts)")
                return None
        except requests.exceptions.RequestException as e:
            print(f" ERROR: {e}")
            return None


def parse_vote_page(html, session_num, vote_num):
    """Parse a vote detail page. Returns (deputy_votes_list, has_data)."""
    soup = BeautifulSoup(html, "html.parser")

    meta = {
        "legislature": LEGISLATURE,
        "session": session_num,
        "vote_num": vote_num,
    }

    # Date - extract from h2 text: "del 29/12/2023 seduta n. 220 ..."
    date_match = re.search(r"del\s+(\d{1,2}/\d{1,2}/\d{4})", soup.get_text())
    meta["date"] = date_match.group(1) if date_match else ""

    # Vote type - from h1: "Votazione finale nominale n.165"
    h1_tags = soup.find_all("h1")
    meta["vote_type"] = ""
    for h1 in h1_tags:
        txt = h1.get_text(strip=True)
        if "votazione" in txt.lower():
            meta["vote_type"] = txt
            break

    # Title - from h3: "Progetto di legge n. 1627"
    h3_tags = soup.find_all("h3")
    meta["title"] = ""
    for h3 in h3_tags:
        txt = h3.get_text(strip=True)
        if txt and "formato tabellare" not in txt.lower() and "riepilogo" not in txt.lower():
            meta["title"] = txt
            break

    # Description - from h4: full bill description
    h4_tags = soup.find_all("h4")
    meta["description"] = ""
    for h4 in h4_tags:
        txt = h4.get_text(strip=True)
        if txt:
            meta["description"] = txt
            break

    # Deputy votes table
    deputy_votes = []
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 3:
                name = cols[0].get_text(strip=True)
                group = cols[1].get_text(strip=True)
                vote = cols[2].get_text(strip=True)
                if name.lower() in ("nominativo", ""):
                    continue
                deputy_votes.append({
                    **meta,
                    "deputy_name": name,
                    "group": group,
                    "vote": vote,
                })

    return deputy_votes, len(deputy_votes) > 0


def load_progress():
    """Load set of already-scraped (session, vote_num) pairs."""
    done = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    done.add((int(parts[0]), int(parts[1])))
    return done


def save_progress(session, vote_num):
    """Append a completed (session, vote_num) to progress file."""
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{session},{vote_num}\n")


def append_to_csv(rows):
    """Append rows to CSV, creating file with headers if needed."""
    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def load_sessions_cache():
    """Load cached session list if available."""
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as f:
            data = json.load(f)
            print(f"  Loaded {len(data['sessions'])} sessions from cache")
            print(f"  (cached on {data.get('date', 'unknown')})")
            print(f"  To force re-discovery, delete {SESSIONS_FILE}")
            return data["sessions"]
    return None


def save_sessions_cache(sessions):
    """Save discovered sessions to cache file."""
    from datetime import datetime
    with open(SESSIONS_FILE, "w") as f:
        json.dump({
            "sessions": sessions,
            "date": datetime.now().isoformat(),
            "max_session_probed": MAX_SESSION,
        }, f)


def discover_sessions():
    """Probe sessions 1-MAX_SESSION to find which ones have roll-call votes."""
    # Check cache first
    cached = load_sessions_cache()
    if cached:
        return cached

    print("=" * 60)
    print(f"PHASE 1: Discovering sessions with roll-call votes")
    print(f"  Probing sessions 1-{MAX_SESSION}...")
    print("=" * 60)

    valid = []
    consecutive_misses = 0

    for s in range(1, MAX_SESSION + 1):
        url = (
            f"{BASE}/schedavotazione.asp"
            f"?Legislatura={LEGISLATURE}"
            f"&RifVotazione={s}_01"
            f"&tipo=dettaglio"
        )
        resp = fetch_with_retry(url)

        if resp and "Nominativo" in resp.text:
            print(f"  Session {s:4d}: HAS VOTES")
            valid.append(s)
            consecutive_misses = 0
        else:
            consecutive_misses += 1

        # If 80 consecutive misses past the last found, stop
        if consecutive_misses >= 80 and len(valid) > 0:
            print(f"  (80 consecutive misses after session {s}, stopping discovery)")
            break

        time.sleep(0.5)

    # Save to cache
    save_sessions_cache(valid)
    print(f"\n  Sessions cached to {SESSIONS_FILE}")

    return valid


def scrape_session(session, done):
    """Scrape all votes in a session. Returns total deputy-vote rows added."""
    total_rows = 0
    vote_num = 1
    consecutive_misses = 0

    while consecutive_misses < 3:
        if (session, vote_num) in done:
            vote_num += 1
            consecutive_misses = 0
            continue

        url = (
            f"{BASE}/schedavotazione.asp"
            f"?Legislatura={LEGISLATURE}"
            f"&RifVotazione={session}_{vote_num:02d}"
            f"&tipo=dettaglio"
        )
        print(f"  Session {session:4d} vote {vote_num:3d}:", end="")

        resp = fetch_with_retry(url)

        if resp is None:
            print(" skipping (network error)")
            vote_num += 1
            consecutive_misses += 1
            continue

        deputy_votes, has_data = parse_vote_page(resp.text, session, vote_num)

        if has_data:
            append_to_csv(deputy_votes)
            save_progress(session, vote_num)
            total_rows += len(deputy_votes)
            title_preview = deputy_votes[0].get("title", "")[:50]
            print(f" OK - {len(deputy_votes)} deputies | {title_preview}")
            consecutive_misses = 0
        else:
            print(f" no data")
            consecutive_misses += 1

        vote_num += 1
        time.sleep(DELAY)

    return total_rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("=" * 60)
    print("  Camera dei Deputati - XIX Legislature")
    print("  Full Roll-Call Vote Scraper")
    print("=" * 60)
    print()

    # Check for existing progress
    done = load_progress()
    if done:
        print(f"  Resuming: {len(done)} votes already scraped")
        print()

    # Phase 1: discover sessions (cached after first run)
    sessions = discover_sessions()

    if not sessions:
        print("\nNo sessions found. Check your connection or the URL pattern.")
        sys.exit(1)

    print(f"\nFound {len(sessions)} sessions with votes.")
    print()

    # Phase 2: scrape all votes
    print("=" * 60)
    print("PHASE 2: Scraping all roll-call votes")
    print("=" * 60)

    grand_total = 0
    for i, session in enumerate(sessions):
        print(f"\n--- Session {session} ({i+1}/{len(sessions)}) ---")
        rows = scrape_session(session, done)
        grand_total += rows

    # Final summary
    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f) - 1
        print(f"  Output file: {OUTPUT_FILE}")
        print(f"  Total rows:  {total_lines}")
        print(f"  New rows added this run: {grand_total}")
    else:
        print("  No data was scraped.")

    print()
    print("  Next: run src/clean.py to process the raw data")


if __name__ == "__main__":
    main()
    
