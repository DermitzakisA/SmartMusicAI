from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class HybridMusicRecommender:
    """Hybrid recommender using track text + implicit listening interactions."""

    def __init__(self, tracks: pd.DataFrame, interactions: pd.DataFrame) -> None:
        required_tracks = {"track_id", "title", "artist", "genres"}
        required_interactions = {"user_id", "track_id", "play_count", "liked"}
        if missing := required_tracks - set(tracks.columns):
            raise ValueError(f"Λείπουν στήλες από tracks.csv: {sorted(missing)}")
        if missing := required_interactions - set(interactions.columns):
            raise ValueError(f"Λείπουν στήλες από interactions.csv: {sorted(missing)}")
        if tracks.empty:
            raise ValueError("Δεν υπάρχουν τραγούδια στο tracks.csv.")

        self.tracks = tracks.copy().fillna("")
        self.interactions = interactions.copy()
        self.tracks["track_id"] = self.tracks["track_id"].astype(str)
        self.tracks = self.tracks.drop_duplicates("track_id").reset_index(drop=True)
        self.interactions["user_id"] = self.interactions["user_id"].astype(str)
        self.interactions["track_id"] = self.interactions["track_id"].astype(str)
        self.interactions["play_count"] = pd.to_numeric(
            self.interactions["play_count"], errors="coerce"
        ).fillna(0).clip(lower=0)
        self.interactions["liked"] = pd.to_numeric(
            self.interactions["liked"], errors="coerce"
        ).fillna(0).clip(0, 1)

        known = set(self.tracks["track_id"])
        self.interactions = self.interactions[
            self.interactions["track_id"].isin(known)
        ].copy()

        # Content features: artist, title and genre tags (when available).
        text = (
            self.tracks["artist"].astype(str) + " "
            + self.tracks["title"].astype(str) + " "
            + self.tracks["genres"].astype(str).str.replace(";", " ", regex=False)
        ).str.lower()
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode", ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w'-]+\b"
        )
        self.track_features = self.vectorizer.fit_transform(text)
        self.track_index = {
            track_id: index for index, track_id in enumerate(self.tracks["track_id"])
        }
        self._build_user_item_matrix()

    def _build_user_item_matrix(self) -> None:
        grouped = self.interactions.groupby(["user_id", "track_id"], as_index=False).agg(
            play_count=("play_count", "sum"), liked=("liked", "max")
        )
        if grouped.empty:
            self.user_ids = []
            self.user_index = {}
            self.user_item = csr_matrix((0, len(self.tracks)), dtype=np.float32)
            return

        grouped["strength"] = np.log1p(grouped["play_count"]) + 2.0 * grouped["liked"]
        self.user_ids = sorted(grouped["user_id"].unique().tolist())
        self.user_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        rows = grouped["user_id"].map(self.user_index).to_numpy()
        cols = grouped["track_id"].map(self.track_index).to_numpy()
        values = grouped["strength"].to_numpy(dtype=np.float32)
        self.user_item = csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_ids), len(self.tracks)),
            dtype=np.float32,
        )
        self.user_item = normalize(self.user_item, norm="l2", axis=1)

    @staticmethod
    def _split(value: str | None) -> list[str]:
        if not value:
            return []
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]

    def recommend(
        self,
        user_id: str,
        genres: str = "",
        artists: str = "",
        n: int = 20,
        exclude_listened: bool = True,
    ) -> pd.DataFrame:
        if n < 1:
            raise ValueError("Το πλήθος προτάσεων πρέπει να είναι τουλάχιστον 1.")

        user_id = str(user_id)
        favorite_genres = self._split(genres)
        favorite_artists = self._split(artists)
        preference_text = " ".join(favorite_genres + favorite_artists).lower()
        query_vector = None
        if preference_text:
            query_vector = self.vectorizer.transform([preference_text])

        listened_ids: set[str] = set()
        history_vector = None
        if user_id in self.user_index:
            user_rows = self.interactions[self.interactions["user_id"] == user_id]
            if not user_rows.empty:
                grouped = user_rows.groupby("track_id", as_index=False)["play_count"].sum()
                ids = grouped["track_id"].tolist()
                positions = [self.track_index[track_id] for track_id in ids]
                weights = np.log1p(grouped["play_count"].to_numpy(dtype=float)) + 1.0
                weighted = csr_matrix(weights.reshape(1, -1)) @ self.track_features[positions]
                history_vector = normalize(weighted)
                listened_ids = set(ids)

        if history_vector is not None and query_vector is not None:
            profile = normalize(history_vector * 0.7 + query_vector * 0.3)
        elif history_vector is not None:
            profile = history_vector
        elif query_vector is not None:
            profile = query_vector
        else:
            raise ValueError(
                "Δεν υπάρχει ιστορικό για αυτόν τον χρήστη. Δώσε τουλάχιστον ένα αγαπημένο είδος ή καλλιτέχνη."
            )

        content_scores = cosine_similarity(profile, self.track_features).ravel()
        content_scores = np.maximum(content_scores, 0)
        if content_scores.max(initial=0) > 0:
            content_scores = content_scores / content_scores.max()

        collaborative_scores = np.zeros(len(self.tracks), dtype=float)
        has_cf = user_id in self.user_index and len(self.user_ids) > 1
        if has_cf:
            target = self.user_index[user_id]
            user_similarities = cosine_similarity(
                self.user_item[target], self.user_item
            ).ravel()
            user_similarities[target] = 0
            collaborative_scores = np.asarray(
                user_similarities.reshape(1, -1) @ self.user_item
            ).ravel()
            if collaborative_scores.max(initial=0) > 0:
                collaborative_scores /= collaborative_scores.max()

        if has_cf and collaborative_scores.max(initial=0) > 0:
            final_scores = 0.65 * content_scores + 0.35 * collaborative_scores
        else:
            final_scores = content_scores

        result = self.tracks.copy()
        result["content_score"] = content_scores
        result["collaborative_score"] = collaborative_scores
        result["score"] = final_scores
        if exclude_listened and listened_ids:
            result = result[~result["track_id"].isin(listened_ids)]
        result = result.sort_values("score", ascending=False)

        # Light artist diversification, followed by backfilling if needed.
        selected = []
        artist_counts: dict[str, int] = {}
        for idx, row in result.iterrows():
            artist = str(row["artist"])
            if artist_counts.get(artist, 0) >= 2:
                continue
            selected.append(idx)
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            if len(selected) >= n:
                break
        if len(selected) < n:
            selected_set = set(selected)
            for idx in result.index:
                if idx not in selected_set:
                    selected.append(idx)
                    if len(selected) >= n:
                        break

        columns = ["track_id", "title", "artist", "genres", "score", "content_score", "collaborative_score"]
        if "album" in result.columns:
            columns.insert(4, "album")
        return result.loc[selected, columns].reset_index(drop=True)
