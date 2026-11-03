from __future__ import annotations

import argparse
from pathlib import Path

from src.listenbrainz_importer import import_users

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Εισαγωγή πρόσφατων δημόσιων ακροάσεων από ListenBrainz."
    )
    parser.add_argument(
        "--usernames", required=True,
        help="ListenBrainz usernames χωρισμένα με κόμμα, π.χ. user1,user2"
    )
    parser.add_argument("--count", type=int, default=500,
                        help="Ακροάσεις ανά χρήστη (1–1000, προεπιλογή 500)")
    args = parser.parse_args()

    usernames = [name.strip() for name in args.usernames.split(",") if name.strip()]
    tracks, interactions = import_users(usernames, count=args.count)
    if tracks.empty or interactions.empty:
        raise SystemExit(
            "Δεν συλλέχθηκαν ακροάσεις. Έλεγξε τα usernames και αν τα προφίλ έχουν δημόσιο ιστορικό."
        )

    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    tracks.to_csv(data_dir / "tracks.csv", index=False, encoding="utf-8-sig")
    interactions.to_csv(data_dir / "interactions.csv", index=False, encoding="utf-8-sig")

    print("\nΗ εισαγωγή ολοκληρώθηκε.")
    print(f"Χρήστες: {interactions['user_id'].nunique()}")
    print(f"Μοναδικά τραγούδια: {len(tracks)}")
    print(f"Αλληλεπιδράσεις χρήστη-τραγουδιού: {len(interactions)}")
    print(f"Αρχεία: {data_dir}")
    print("\nΣημείωση: τα genres μένουν κενά όταν δεν παρέχονται από την πηγή.")


if __name__ == "__main__":
    main()
