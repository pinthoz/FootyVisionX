"""FootyVision Player Photo Scraper CLI.

Downloads player portrait photos across Wikipedia/Wikimedia Commons & TheSportsDB
and stores them into `frontend/web/public/photos/{player_id}.jpg`.

Usage:
    python -m footyvision.scrape_photos
    python -m footyvision.scrape_photos --limit 30
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from sqlalchemy import select

from footyvision.db.base import get_session
from footyvision.db.models import Player, PlayerSeasonStats

PHOTOS_DIR = Path(__file__).resolve().parents[2] / "frontend" / "web" / "public" / "photos"


def get_name_variants(full_name: str) -> list[str]:
    clean = full_name.strip()
    variants = [clean]
    parts = clean.split()
    if len(parts) > 2:
        variants.append(f"{parts[0]} {parts[-1]}")
        variants.append(f"{parts[0]} {parts[1]}")
    return variants


def fetch_url(url: str, is_json: bool = True):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "FootyVisionScout/1.0 (https://github.com/pinthoz/FootyVisionX; football analytics)"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            data = response.read()
            return json.loads(data.decode("utf-8")) if is_json else data
    except Exception:
        return None


def resolve_photo_url(player_name: str) -> str | None:
    variants = get_name_variants(player_name)

    # 1. Wikipedia Summary API
    for v in variants:
        slug = urllib.parse.quote(v.replace(" ", "_"))
        data = fetch_url(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")
        if data and isinstance(data, dict):
            thumb = data.get("thumbnail", {}).get("source")
            if thumb and "Disambig" not in thumb:
                return thumb

    # 2. Wikipedia generator search
    for v in variants[:2]:
        q = urllib.parse.quote(v)
        data = fetch_url(
            f"https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={q}&prop=pageimages&pithumbsize=400&format=json"
        )
        if data and isinstance(data, dict):
            pages = data.get("query", {}).get("pages", {})
            for _, page in sorted(pages.items(), key=lambda x: x[1].get("index", 99)):
                thumb = page.get("thumbnail", {}).get("source")
                if thumb and "logo" not in thumb.lower() and "flag" not in thumb.lower():
                    return thumb

    # 3. TheSportsDB Free Search
    for v in variants[:2]:
        q = urllib.parse.quote(v)
        data = fetch_url(f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={q}")
        if data and isinstance(data, dict):
            player_list = data.get("player") or []
            if player_list:
                p = player_list[0]
                photo = p.get("strCutout") or p.get("strThumb") or p.get("strRender")
                if photo:
                    return photo

    return None


def scrape_all(limit: int = 100):
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    session = next(get_session())

    stmt = (
        select(Player)
        .where(
            select(PlayerSeasonStats.id).where(PlayerSeasonStats.player_id == Player.id).exists()
        )
        .order_by(Player.name)
        .limit(limit)
    )
    players = list(session.scalars(stmt))

    print(f"Scraping photos for {len(players)} players with active season stats...")
    saved_count = 0

    for i, p in enumerate(players, start=1):
        target_file = PHOTOS_DIR / f"{p.id}.jpg"
        if target_file.exists():
            print(f"[{i}/{len(players)}] {p.name}: already cached locally.")
            saved_count += 1
            continue

        url = resolve_photo_url(p.name)
        if url:
            img_data = fetch_url(url, is_json=False)
            if img_data:
                target_file.write_bytes(img_data)
                print(f"[{i}/{len(players)}] {p.name}: [OK] downloaded ({len(img_data)} bytes)")
                saved_count += 1
            else:
                print(f"[{i}/{len(players)}] {p.name}: download failed.")
        else:
            print(f"[{i}/{len(players)}] {p.name}: no public portrait found.")

        time.sleep(0.2)

    print(f"\nFinished: {saved_count}/{len(players)} photos ready in {PHOTOS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape player portraits")
    parser.add_argument("--limit", type=int, default=50, help="Max players to scrape")
    args = parser.parse_args()
    scrape_all(limit=args.limit)
