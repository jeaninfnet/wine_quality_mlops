import logging
from pathlib import Path

import pandas as pd

from src.wine_pipeline.build_features import build_white_wine_opinion_dataset


def test_build_white_wine_opinion(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "type": ["white", "white", "red", "white"],
            "quality": [5, 6, 7, 4],
            "alcohol": [10.0, 11.0, 12.0, 9.0],
            "ph": [3.2, 3.3, 3.1, 3.0],
        }
    )
    raw_path = tmp_path / "raw.parquet"
    feat_path = tmp_path / "feat.parquet"
    raw.to_parquet(raw_path, index=False)

    build_white_wine_opinion_dataset(raw_path, feat_path, logger=logging.getLogger("t"))

    out = pd.read_parquet(feat_path)
    assert "opinion" in out.columns
    assert len(out) == 3
    assert set(out["opinion"]) <= {0, 1}
