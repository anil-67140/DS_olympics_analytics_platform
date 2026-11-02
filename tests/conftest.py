import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from olympics.config import load_config  # noqa: E402
from olympics.data.clean import build_medal_table, clean_events  # noqa: E402
from olympics.data.synthetic import make_synthetic_events  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides=["ml.profile=fast", "ml.cv_top_k=1", "ml.cv_folds=3", "forecast.holdout_editions=4",
                                  "ml.models=[logistic_regression,decision_tree,hist_gradient_boosting]"])


@pytest.fixture(scope="session")
def raw():
    return make_synthetic_events()


@pytest.fixture(scope="session")
def cleaned(raw, cfg):
    events, regions = raw
    df, log = clean_events(events, regions, cfg)
    return df, build_medal_table(df), regions, log


@pytest.fixture(scope="session")
def pipeline_ws(tmp_path_factory, cfg):
    """One full demo pipeline run shared by the integration tests."""
    from olympics.pipeline import Pipeline
    root = tmp_path_factory.mktemp("ws")
    pipe = Pipeline(cfg, root, demo=True)
    pipe.run()
    return pipe
