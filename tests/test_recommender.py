from pathlib import Path
import gc
import json
import sys
import tempfile
import unittest

import numpy as np


FUNCTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FUNCTION_DIR))

from recommender import AlsRecommender, parse_clicks


def write_artifacts(artifact_dir: Path) -> None:
    arrays = {
        "user_ids": np.asarray([10], dtype=np.int64),
        "article_ids": np.arange(7, dtype=np.int64),
        "user_factors": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "item_factors": np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.7, 0.3],
            ],
            dtype=np.float32,
        ),
        "history_offsets": np.asarray([0, 2], dtype=np.int64),
        "history_article_ids": np.asarray([0, 1], dtype=np.int32),
        "candidate_article_ids": np.arange(7, dtype=np.int32),
        "popular_article_ids": np.asarray(
            [3, 4, 5, 6, 2, 1, 0],
            dtype=np.int32,
        ),
    }
    files = {}
    for key, values in arrays.items():
        filename = f"{key}.npy"
        np.save(artifact_dir / filename, values, allow_pickle=False)
        files[key] = {"name": filename, "sha256": "not-used"}

    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": "implicit_als",
                "version": 2,
                "score_type": "latent_factor_dot_product",
                "top_k": 5,
                "files": files,
            }
        ),
        encoding="utf-8",
    )


class RecommenderTest(unittest.TestCase):
    def test_known_and_unknown_users(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            write_artifacts(artifact_dir)

            recommender = AlsRecommender(artifact_dir)
            known = recommender.recommend(10)
            unknown = recommender.recommend(999)

            self.assertEqual(known.strategy, "implicit_als")
            self.assertEqual(known.article_ids[0], 2)
            self.assertEqual(len(known.article_ids), 5)
            self.assertFalse({0, 1}.intersection(known.article_ids))
            self.assertEqual(unknown.strategy, "popular_fallback")
            self.assertEqual(unknown.article_ids, [3, 4, 5, 6, 2])

            del recommender, known, unknown
            gc.collect()

    def test_unknown_user_clicks_are_excluded_from_popularity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            write_artifacts(artifact_dir)

            recommender = AlsRecommender(artifact_dir)
            result = recommender.recommend(999, clicks=[3, 4])

            self.assertEqual(result.strategy, "popular_fallback")
            self.assertEqual(result.article_ids, [5, 6, 2, 1, 0])

            del recommender, result
            gc.collect()


class ParseClicksTest(unittest.TestCase):
    def test_missing_clicks_are_empty(self) -> None:
        self.assertEqual(parse_clicks(None), [])

    def test_valid_clicks_are_kept_in_order(self) -> None:
        self.assertEqual(parse_clicks([5, 0, 12]), [5, 0, 12])

    def test_invalid_clicks_are_rejected(self) -> None:
        for value in ["12", [1, -2], [1.5], [True], list(range(1_001))]:
            with self.subTest(value=value if not isinstance(value, list) or len(value) < 10 else "too long"):
                with self.assertRaises(ValueError):
                    parse_clicks(value)


if __name__ == "__main__":
    unittest.main()
