from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


class RandomForestFeatureSelector:
    """Select a fixed number of named characteristics using a Random Forest."""

    def __init__(
        self,
        selected_feature_count: int = 15,
        n_estimators: int = 150,
        max_depth: int | None = 12,
        min_samples_leaf: int = 25,
        min_samples_split: int = 10,
        max_features: str | float | int | None = "sqrt",
        max_samples: float | int | None = 0.7,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.selected_feature_count = selected_feature_count
        self.imputer = SimpleImputer(strategy="median")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            max_samples=max_samples,
            criterion="squared_error",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_names_: list[str] | None = None
        self.importance_: pd.DataFrame | None = None
        self.selected_features_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame so selected feature names are retained")
        if X.empty:
            raise ValueError("X must contain at least one training row")

        self.feature_names_ = X.columns.tolist()
        transformed = self.imputer.fit_transform(X)
        self.model.fit(transformed, y)
        self.importance_ = pd.DataFrame(
            {"feature": self.feature_names_, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False, ignore_index=True)
        count = min(self.selected_feature_count, len(self.importance_))
        self.selected_features_ = self.importance_.head(count)["feature"].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("Call fit before transform")
        missing = set(self.selected_features_) - set(X.columns)
        if missing:
            raise ValueError(f"X is missing selected features: {sorted(missing)}")
        return X.loc[:, self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
