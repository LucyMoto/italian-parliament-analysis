"""
Scrape sample voting data from Camera dei Deputati - XIX Legislature
Run locally: python scrape_camera_votes.py

Outputs:
  - camera_votes_sample.csv  (individual deputy votes)
  - summary printed to console
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
import re
import sys

BASE = "https://documenti.camera.it/votazioni/votazionitutte"
LEGISLATURE = 19  # XIX legislature (started Oct 2022)
DELAY = 1.5  # seconds between requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research project - parliamentary voting analysis)"
}


def get_vote_detail(session_num, vote_num):
    """Scrape a single vote detail page and return list of deputy votes."""
    url = (
        f"{BASE}/schedavotazione.asp"
        f"?Legislatura={LEGISLATURE}"
        f"&RifVotazione={session_num}_{vote_num:02d}"
        f"&tipo=dettaglio"
    )
    print(f"  Fetching: session {session_num}, vote {vote_num} ...", end=" ")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: {e}")
        return None, None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract vote metadata
    meta = {}
    meta["legislature"] = LEGISLATURE
    meta["session"] = session_num
    meta["vote_num"] = vote_num

    # Try to get the vote title/description
    h2 = soup.find("h2")
    meta["title"] = h2.get_text(strip=True) if h2 else ""

    # Try to get the date from the page
    date_match = re.search(r"del\s+(\d{1,2}/\d{1,2}/\d{4})", soup.get_text())
    meta["date"] = date_match.group(1) if date_match else ""

    # Try to get vote type
    h1_tags = soup.find_all("h1")
    for h1 in h1_tags:
        txt = h1.get_text(strip=True)
        if "nominale" in txt.lower() or "votazione" in txt.lower():
            meta["vote_type"] = txt
            break
    else:
        meta["vote_type"] = ""

    # Find the deputy votes table
    tables = soup.find_all("table")
    deputy_votes = []

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 3:
                name = cols[0].get_text(strip=True)
                group = cols[1].get_text(strip=True)
                vote = cols[2].get_text(strip=True)
                # Skip header-like rows
                if name.lower() in ("nominativo", ""):
                    continue
                deputy_votes.append({
                    "deputy_name": name,
                    "group": group,
                    "vote": vote,
                    **meta
                })

    if deputy_votes:
        print(f"OK - {len(deputy_votes)} deputies")
    else:
        print("No deputy data found (page may not exist)")

    return deputy_votes, meta


def discover_sessions():
    """
    Try to find valid session/vote combinations for legislature XIX.
    We'll probe a range of session numbers with vote 01 to find active sessions.
    """
    print("=" * 60)
    print("STEP 1: Discovering available sessions...")
    print("=" * 60)

    valid_sessions = []

    # Legislature XIX started Oct 2022. Sessions are numbered sequentially.
    # Let's probe sessions in steps to find the range.
    probe_sessions = list(range(1, 20)) + list(range(50, 70)) + list(range(100, 120)) + list(range(200, 220))

    for s in probe_sessions:
        url = (
            f"{BASE}/schedavotazione.asp"
            f"?Legislatura={LEGISLATURE}"
            f"&RifVotazione={s}_01"
            f"&tipo=dettaglio"
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            # Check if page has actual vote data
            if resp.status_code == 200 and "Nominativo" in resp.text:
                print(f"  Session {s}: HAS VOTES")
                valid_sessions.append(s)
            else:
                print(f"  Session {s}: no votes")
        except Exception as e:
            print(f"  Session {s}: error ({e})")

        time.sleep(0.5)

        # Stop probing once we have enough to work with
        if len(valid_sessions) >= 8:
            break

    return valid_sessions


def main():
    print("=" * 60)
    print("Camera dei Deputati - XIX Legislature Vote Scraper")
    print("=" * 60)
    print()

    # Step 1: Find valid sessions
    valid_sessions = discover_sessions()

    if not valid_sessions:
        print("\nNo valid sessions found. The URL pattern may have changed.")
        print("Try visiting this URL manually in your browser:")
        print(f"  {BASE}/FormVotazioni.Asp?Legislatura={LEGISLATURE}")
        sys.exit(1)

    print(f"\nFound {len(valid_sessions)} sessions with votes: {valid_sessions}")

    # Step 2: Scrape deputy-level data from a sample of votes
    print()
    print("=" * 60)
    print("STEP 2: Scraping individual vote details...")
    print("=" * 60)

    all_votes = []
    votes_scraped = 0
    target_votes = 10  # Scrape up to 10 votes for the sample

    for session in valid_sessions[:5]:  # Use up to 5 sessions
        for vote_num in range(1, 6):  # Up to 5 votes per session
            if votes_scraped >= target_votes:
                break

            deputy_votes, meta = get_vote_detail(session, vote_num)
            if deputy_votes:
                all_votes.extend(deputy_votes)
                votes_scraped += 1

            time.sleep(DELAY)

        if votes_scraped >= target_votes:
            break

    if not all_votes:
        print("\nNo vote data retrieved. See errors above.")
        sys.exit(1)

    # Step 3: Save to CSV
    output_file = "camera_votes_sample.csv"
    fieldnames = [
        "legislature", "session", "vote_num", "date",
        "vote_type", "title", "deputy_name", "group", "vote"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_votes)

    print(f"\nSaved {len(all_votes)} rows to {output_file}")

    # Step 4: Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    unique_deputies = set(row["deputy_name"] for row in all_votes)
    unique_groups = set(row["group"] for row in all_votes)
    unique_votes_scraped = set(
        (row["session"], row["vote_num"]) for row in all_votes
    )

    print(f"  Roll-call votes scraped:  {len(unique_votes_scraped)}")
    print(f"  Total deputy-vote rows:   {len(all_votes)}")
    print(f"  Unique deputies:          {len(unique_deputies)}")
    print(f"  Parliamentary groups:     {len(unique_groups)}")

    print()
    print("  Groups found:")
    group_counts = {}
    for row in all_votes:
        group_counts[row["group"]] = group_counts.get(row["group"], 0) + 1
    for g, c in sorted(group_counts.items(), key=lambda x: -x[1]):
        print(f"    {g}: {c} vote records")

    print()
    print("  Vote types found:")
    vote_type_counts = {}
    for row in all_votes:
        vote_type_counts[row["vote"]] = vote_type_counts.get(row["vote"], 0) + 1
    for v, c in sorted(vote_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {v}: {c}")

    print()
    print("  Sample rows (first 5):")
    for row in all_votes[:5]:
        print(f"    {row['deputy_name']:30s} | {row['group']:15s} | {row['vote']}")

    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("  1. Inspect camera_votes_sample.csv")
    print("  2. If data looks good, scale up to scrape all sessions")
    print("  3. Cross-reference with party affiliation for cohesion analysis")


if __name__ == "__main__":
    main()
    