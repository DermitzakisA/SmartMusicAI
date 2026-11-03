from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.recommender import HybridMusicRecommender

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SmartMusic AI: εξατομικευμένη δημιουργία playlist με ML."
    )
    parser.add_argument("--user-id", default="new_listener", help="User ID ακροατή")
    parser.add_argument("--genres", default="", help="Αγαπημένα είδη, χωρισμένα με κόμμα")
    parser.add_argument("--artists", default="", help="Αγαπημένοι καλλιτέχνες, χωρισμένοι με κόμμα")
    parser.add_argument("--n", type=int, default=20, help="Πλήθος προτάσεων")
    parser.add_argument("--include-listened", action="store_true",
                        help="Να περιλαμβάνονται τραγούδια που έχει ήδη ακούσει")
    args = parser.parse_args()

    tracks_path = BASE_DIR / "data" / "tracks.csv"
    interactions_path = BASE_DIR / "data" / "interactions.csv"
    try:
        tracks = pd.read_csv(tracks_path).fillna("")
        interactions = pd.read_csv(interactions_path).fillna(0)
        recommender = HybridMusicRecommender(tracks, interactions)
        recommendations = recommender.recommend(
            user_id=args.user_id,
            genres=args.genres,
            artists=args.artists,
            n=args.n,
            exclude_listened=not args.include_listened,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Δεν βρέθηκε αρχείο: {exc.filename}") from exc
    except (ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Σφάλμα δεδομένων: {exc}") from exc

    if recommendations.empty:
        print("Δεν βρέθηκαν προτάσεις. Χρειάζονται περισσότερα τραγούδια ή διαφορετικές προτιμήσεις.")
        return

    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    safe_user = "".join(char if char.isalnum() or char in "-_" else "_" for char in args.user_id)
    output_file = output_dir / f"playlist_{safe_user}.csv"
    recommendations.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\nSMARTMUSIC AI — ΠΡΟΤΕΙΝΟΜΕΝΟ PLAYLIST\n")
    print(recommendations.to_string(index=False))
    print(f"\nΑποθηκεύτηκε στο: {output_file}")


if __name__ == "__main__":
    main()
