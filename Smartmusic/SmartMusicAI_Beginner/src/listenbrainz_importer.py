from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

API_ROOT = "https://api.listenbrainz.org/1"
USER_AGENT = "SmartMusicAI/0.1 (educational music recommendation project)"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _make_track_id(recording_mbid: str, artist: str, title: str) -> str:
    if recording_mbid:
        return recording_mbid
    # Stable fallback when a MusicBrainz recording ID is not supplied.
    normalized = f"{artist.lower()}::{title.lower()}"
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def fetch_user_listens(username: str, count: int = 1000) -> list[dict[str, Any]]:
    """Fetch recent public listens for one ListenBrainz username.

    ListenBrainz's listens endpoint supports up to 1,000 records per request.
    For a full history export, use the official export API instead.
    """
    if not username.strip():
        raise ValueError("Το ListenBrainz username δεν μπορεί να είναι κενό.")
    if not 1 <= count <= 1000:
        raise ValueError("Το count πρέπει να είναι από 1 έως 1000.")

    url = f"{API_ROOT}/user/{quote(username.strip(), safe='')}/listens"
    response = requests.get(
        url,
        params={"count": count},
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=30,
    )
    if response.status_code == 404:
        raise ValueError(f"Δεν βρέθηκε δημόσιο προφίλ ListenBrainz: {username}")
    response.raise_for_status()

    payload = response.json().get("payload", {})
    listens = payload.get("listens", [])
    if not isinstance(listens, list):
        raise ValueError(f"Μη αναμενόμενη απάντηση από ListenBrainz για {username}.")
    return listens


def import_users(usernames: list[str], count: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Import several public users, pausing between API calls."""
    track_rows: dict[str, dict[str, str]] = {}
    interaction_rows: list[dict[str, Any]] = []

    clean_usernames = list(dict.fromkeys(u.strip() for u in usernames if u.strip()))
    if not clean_usernames:
        raise ValueError("Δώσε τουλάχιστον ένα ListenBrainz username.")

    for index, username in enumerate(clean_usernames):
        print(f"Λήψη ακροάσεων για {username} ({index + 1}/{len(clean_usernames)})...")
        listens = fetch_user_listens(username, count=count)
        user_track_counts: dict[str, int] = {}

        for listen in listens:
            metadata = listen.get("track_metadata") or {}
            additional = metadata.get("additional_info") or {}
            title = _clean_text(metadata.get("track_name"))
            artist = _clean_text(metadata.get("artist_name"))
            album = _clean_text(metadata.get("release_name"))
            recording_mbid = _clean_text(additional.get("recording_mbid"))

            if not title or not artist:
                continue

            track_id = _make_track_id(recording_mbid, artist, title)
            track_rows.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "title": title,
                    "artist": artist,
                    "album": album,
                    # ListenBrainz listens do not consistently include genre tags.
                    # This field can be enriched later from a separate licensed source.
                    "genres": "",
                },
            )
            user_track_counts[track_id] = user_track_counts.get(track_id, 0) + 1

        for track_id, play_count in user_track_counts.items():
            interaction_rows.append(
                {
                    "user_id": username,
                    "track_id": track_id,
                    "play_count": play_count,
                    "liked": 0,
                }
            )

        # ListenBrainz recommends that clients respect the API's rate limits.
        if index < len(clean_usernames) - 1:
            time.sleep(1.1)

    tracks = pd.DataFrame(
        track_rows.values(), columns=["track_id", "title", "artist", "album", "genres"]
    )
    interactions = pd.DataFrame(
        interaction_rows, columns=["user_id", "track_id", "play_count", "liked"]
    )
    return tracks, interactions
