"""Model zoo: 13 scikit-learn algorithms + a baseline, plus XGBoost / LightGBM / CatBoost when installed.

Each spec says how many training rows the model should see (slow, O(n^2)-ish models are capped) so a full
comparison stays tractable on a free Colab CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    family: str
    max_rows: int | None = None   # cap on training rows for this model (None = no cap)
    is_baseline: bool = False


def build_models(seed: int = 42, profile: str = "full", n_jobs: int = -1) -> list[ModelSpec]:
    fast = profile == "fast"
    n_trees, n_boost = (80, 80) if fast else (300, 300)
    specs = [
        ModelSpec("baseline_prior", DummyClassifier(strategy="prior"), "baseline", is_baseline=True),
        ModelSpec("logistic_regression", LogisticRegression(max_iter=1000, C=1.0), "linear"),
        ModelSpec("linear_svm_calibrated", CalibratedClassifierCV(LinearSVC(C=0.5, dual="auto", max_iter=3000), cv=3, method="sigmoid"), "linear"),
        ModelSpec("lda", LinearDiscriminantAnalysis(), "linear"),
        ModelSpec("gaussian_nb", GaussianNB(), "probabilistic"),
        ModelSpec("knn", KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=n_jobs), "instance", max_rows=15000 if fast else 40000),
        ModelSpec("decision_tree", DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=seed), "tree"),
        ModelSpec("random_forest", RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=5, n_jobs=n_jobs, random_state=seed), "bagging"),
        ModelSpec("extra_trees", ExtraTreesClassifier(n_estimators=n_trees, min_samples_leaf=5, n_jobs=n_jobs, random_state=seed), "bagging"),
        ModelSpec("bagging_trees", BaggingClassifier(DecisionTreeClassifier(max_depth=12, random_state=seed), n_estimators=20 if fast else 60, n_jobs=n_jobs, random_state=seed), "bagging"),
        ModelSpec("adaboost", AdaBoostClassifier(n_estimators=60 if fast else 200, learning_rate=0.5, random_state=seed), "boosting"),
        ModelSpec("gradient_boosting", GradientBoostingClassifier(n_estimators=60 if fast else 200, max_depth=3, subsample=0.7, random_state=seed), "boosting", max_rows=40000 if fast else 100000),
        ModelSpec("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=100 if fast else 400, learning_rate=0.06, l2_regularization=1.0, early_stopping=True, random_state=seed), "boosting"),
        ModelSpec("mlp_neural_net", MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=30 if fast else 150, random_state=seed), "neural", max_rows=40000 if fast else 100000),
    ]
    try:
        from xgboost import XGBClassifier  # type: ignore
        specs.append(ModelSpec("xgboost", XGBClassifier(n_estimators=n_boost, learning_rate=0.06, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr", n_jobs=n_jobs, random_state=seed), "boosting"))
    except Exception:  # noqa: BLE001
        pass
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        specs.append(ModelSpec("lightgbm", LGBMClassifier(n_estimators=n_boost, learning_rate=0.06, num_leaves=63, subsample=0.8, colsample_bytree=0.8, n_jobs=n_jobs, random_state=seed, verbose=-1), "boosting"))
    except Exception:  # noqa: BLE001
        pass
    try:
        from catboost import CatBoostClassifier  # type: ignore
        specs.append(ModelSpec("catboost", CatBoostClassifier(iterations=n_boost, learning_rate=0.08, depth=6, random_seed=seed, verbose=False, thread_count=None if n_jobs == -1 else n_jobs), "boosting"))
    except Exception:  # noqa: BLE001
        pass
    return specs
