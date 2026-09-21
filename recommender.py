from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FILE_KEYS = (
    "user_ids",
    "article_ids",
    "user_factors",
    "item_factors",
    "history_offsets",
    "history_article_ids",
    "candidate_article_ids",
    "popular_article_ids",
)
MAX_CLICKS = 1_000


class ArtifactConfigurationError(RuntimeError):
    pass


def parse_clicks(value: Any) -> list[int]:
    """Validate the optional article IDs clicked by a user, oldest first."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_CLICKS:
        raise ValueError(f"clicks must be a list of at most {MAX_CLICKS} article IDs.")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError("clicks must contain non-negative integer article IDs.")
    return list(value)


@dataclass(frozen=True)
class Recommendation:
    user_id: int
    article_ids: list[int]
    scores: list[float | None]
    strategy: str


class ArtifactStore:
    def __init__(self) -> None:
        local_dir = os.getenv("ARTIFACTS_LOCAL_DIR")
        self.local_dir = Path(local_dir).resolve() if local_dir else None
        self.container = os.getenv("MODEL_CONTAINER", "models")
        self.account_url = os.getenv("STORAGE_ACCOUNT_URL")
        self.cache_dir = Path(tempfile.gettempdir()) / "mycontent-models"

    def materialize(self) -> Path:
        if self.local_dir:
            if not (self.local_dir / "manifest.json").exists():
                raise ArtifactConfigurationError(
                    f"No manifest.json found in {self.local_dir}"
                )
            return self.local_dir

        if not self.account_url:
            raise ArtifactConfigurationError(
                "STORAGE_ACCOUNT_URL is required when ARTIFACTS_LOCAL_DIR is unset"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._download("manifest.json", force=True)
        manifest = json.loads(
            (self.cache_dir / "manifest.json").read_text(encoding="utf-8")
        )
        for key in REQUIRED_FILE_KEYS:
            try:
                artifact = manifest["files"][key]
                filename = artifact["name"]
                checksum = artifact["sha256"]
            except KeyError as error:
                raise ArtifactConfigurationError(
                    f"Missing artifact entry: {key}"
                ) from error
            self._download(filename, expected_sha256=checksum)
        return self.cache_dir

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _download(
        self,
        filename: str,
        expected_sha256: str | None = None,
        force: bool = False,
    ) -> None:
        target = self.cache_dir / filename
        if (
            not force
            and target.exists()
            and target.stat().st_size > 0
            and (
                expected_sha256 is None
                or self._sha256(target) == expected_sha256
            )
        ):
            return

        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobClient

        blob = BlobClient(
            account_url=self.account_url,
            container_name=self.container,
            blob_name=filename,
            credential=DefaultAzureCredential(),
        )
        temporary = target.with_suffix(target.suffix + ".part")
        with temporary.open("wb") as file:
            blob.download_blob(max_concurrency=4).readinto(file)
        if expected_sha256 and self._sha256(temporary) != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise ArtifactConfigurationError(
                f"Checksum mismatch for downloaded artifact: {filename}"
            )
        temporary.replace(target)


class AlsRecommender:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.manifest = json.loads(
            (self.artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
        files = self.manifest["files"]
        self.user_ids = self._load(files["user_ids"]["name"])
        self.article_ids = self._load(files["article_ids"]["name"])
        self.user_factors = self._load(files["user_factors"]["name"])
        self.item_factors = self._load(files["item_factors"]["name"])
        self.history_offsets = self._load(files["history_offsets"]["name"])
        self.history_article_ids = self._load(
            files["history_article_ids"]["name"]
        )
        self.candidate_article_ids = self._load(
            files["candidate_article_ids"]["name"]
        ).astype(np.int64, copy=False)
        self.popular_article_ids = self._load(
            files["popular_article_ids"]["name"]
        ).astype(np.int64, copy=False)
        self.top_k = int(self.manifest.get("top_k", 5))
        self._validate()

        self.candidate_inner_ids = np.searchsorted(
            self.article_ids,
            self.candidate_article_ids,
        )
        if np.any(self.candidate_inner_ids >= len(self.article_ids)) or not np.array_equal(
            self.article_ids[self.candidate_inner_ids],
            self.candidate_article_ids,
        ):
            raise ArtifactConfigurationError(
                "A candidate article is unknown to the ALS model"
            )
        self.candidate_factors = np.asarray(
            self.item_factors[self.candidate_inner_ids],
            dtype=np.float32,
        )

    def _load(self, filename: str) -> np.ndarray:
        return np.load(self.artifact_dir / filename, mmap_mode="r", allow_pickle=False)

    def _validate(self) -> None:
        if self.user_factors.ndim != 2 or self.item_factors.ndim != 2:
            raise ArtifactConfigurationError("ALS factors must be matrices")
        if self.user_factors.shape[0] != len(self.user_ids):
            raise ArtifactConfigurationError("User factors do not match user IDs")
        if self.item_factors.shape[0] != len(self.article_ids):
            raise ArtifactConfigurationError("Item factors do not match article IDs")
        if self.user_factors.shape[1] != self.item_factors.shape[1]:
            raise ArtifactConfigurationError("ALS factor dimensions do not match")
        if len(self.history_offsets) != len(self.user_ids) + 1:
            raise ArtifactConfigurationError("History offsets do not match user IDs")
        if int(self.history_offsets[-1]) != len(self.history_article_ids):
            raise ArtifactConfigurationError("History offsets are inconsistent")
        if np.any(np.diff(self.user_ids) <= 0):
            raise ArtifactConfigurationError("User IDs must be strictly sorted")
        if np.any(np.diff(self.article_ids) <= 0):
            raise ArtifactConfigurationError("Article IDs must be strictly sorted")

    def recommend(self, user_id: int, clicks: list[int] | None = None) -> Recommendation:
        user_index = int(np.searchsorted(self.user_ids, user_id))
        if user_index >= len(self.user_ids) or int(self.user_ids[user_index]) != user_id:
            return self._popular_fallback(user_id, seen=set(clicks or []))

        start = int(self.history_offsets[user_index])
        end = int(self.history_offsets[user_index + 1])
        history = np.asarray(self.history_article_ids[start:end], dtype=np.int64)
        scores = self.candidate_factors @ np.asarray(
            self.user_factors[user_index],
            dtype=np.float32,
        )
        scores[np.isin(self.candidate_article_ids, history)] = -np.inf
        order = np.lexsort((self.candidate_article_ids, -scores))
        valid = order[np.isfinite(scores[order])][: self.top_k]

        article_ids = self.candidate_article_ids[valid].astype(int).tolist()
        selected_scores: list[float | None] = [
            float(scores[position]) for position in valid
        ]
        if len(article_ids) < self.top_k:
            excluded = set(map(int, history)) | set(article_ids)
            fallback = self._popular_ids(excluded, self.top_k - len(article_ids))
            article_ids.extend(fallback)
            selected_scores.extend([None] * len(fallback))

        return Recommendation(
            user_id=user_id,
            article_ids=article_ids,
            scores=selected_scores,
            strategy="implicit_als",
        )

    def _popular_ids(self, seen: set[int], limit: int) -> list[int]:
        selected: list[int] = []
        for article_id in self.popular_article_ids:
            value = int(article_id)
            if value not in seen:
                selected.append(value)
                if len(selected) == limit:
                    break
        return selected

    def _popular_fallback(
        self,
        user_id: int,
        seen: set[int],
    ) -> Recommendation:
        article_ids = self._popular_ids(seen, self.top_k)
        return Recommendation(
            user_id=user_id,
            article_ids=article_ids,
            scores=[None] * len(article_ids),
            strategy="popular_fallback",
        )

    def info(self) -> dict[str, Any]:
        return {
            "model": self.manifest["model"],
            "version": self.manifest["version"],
            "users": len(self.user_ids),
            "articles": len(self.article_ids),
            "candidates": len(self.candidate_article_ids),
            "factors": self.user_factors.shape[1],
            "score_type": self.manifest["score_type"],
        }


_SERVICE: AlsRecommender | None = None
_SERVICE_LOCK = threading.Lock()


def get_recommender() -> AlsRecommender:
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                artifact_dir = ArtifactStore().materialize()
                _SERVICE = AlsRecommender(artifact_dir)
    return _SERVICE
