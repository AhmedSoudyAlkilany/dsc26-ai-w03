"""Failure-risk baseline for Tawseel COD parcels.

Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import math


# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "failed"

# Known when the parcel is created. Excluded on purpose:
#   parcel_id (row identifier), courier_notes and delivered_at (written after the
#   delivery attempt), failed (the label).
# customer_id and created_at are kept because add_history_features needs them; the
# pipeline itself does not use them as raw inputs.
FEATURE_COLUMNS: list[str] = [
    "customer_id", "merchant_id", "area", "created_at", "promised_slot",
    "cod_amount", "n_items", "address_text",
]

HISTORY_COLUMNS = ["customer_prior_parcels", "customer_prior_fail_rate"]
_CATEGORICAL = ["area", "merchant_id", "promised_slot"]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time customer history: only parcels strictly earlier than each row count."""
    out = df.copy()
    n = len(df)
    if n == 0:
        out["customer_prior_parcels"] = np.array([], dtype="int64")
        out["customer_prior_fail_rate"] = np.array([], dtype="float64")
        return out

    tmp = pd.DataFrame({
        "pos": np.arange(n),
        "c": df["customer_id"].to_numpy(),
        "t": pd.to_datetime(df["created_at"]).to_numpy(),
        "f": pd.to_numeric(df[TARGET]).to_numpy(dtype="float64"),
    })
    tmp = tmp.sort_values(["c", "t", "pos"], kind="stable")
    g = tmp.groupby("c", sort=False, dropna=False)
    # Counts/fails over all earlier rows of the customer in sorted order (self excluded)...
    tmp["cnt"] = g.cumcount()
    tmp["fails"] = g["f"].cumsum() - tmp["f"]
    # ...then rows sharing a timestamp take the value of the first of them,
    # so equal timestamps are never "prior" to each other.
    tg = tmp.groupby(["c", "t"], sort=False, dropna=False)
    tmp["cnt"] = tg["cnt"].transform("min")
    tmp["fails"] = tg["fails"].transform("min")
    tmp = tmp.sort_values("pos")

    cnt = tmp["cnt"].to_numpy(dtype="int64")
    fails = tmp["fails"].to_numpy(dtype="float64")
    rate = np.full(n, np.nan)
    np.divide(fails, cnt, out=rate, where=cnt > 0)
    out["customer_prior_parcels"] = cnt
    out["customer_prior_fail_rate"] = rate
    return out


def time_split(df: pd.DataFrame, time_col: str = "created_at",
               test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split; all rows sharing the boundary timestamp go to test."""
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be strictly between 0 and 1")
    n = len(df)
    if n == 0:
        return df.copy(), df.copy()
    n_test = math.ceil(n * test_fraction)
    ordered = df.sort_values(time_col, kind="stable")
    boundary = ordered[time_col].iloc[n - n_test]
    is_test = df[time_col] >= boundary
    return df[~is_test].copy(), df[is_test].copy()


# ---- preprocessing helpers (module-level so the pipeline can be pickled) ----

def _to_float(X):
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").astype("float64").to_numpy()


def _to_log_float(X):
    return np.log1p(np.clip(_to_float(X), 0, None))


def _to_clean_str(X):
    X = pd.DataFrame(X).astype(object)
    return X.where(X.notna(), "__missing__").astype(str).to_numpy()


def _address_features(X):
    """Two cheap, language-independent signals from the free-text address:
    is it empty, and does it contain any digit (building/street/apartment number)."""
    s = pd.DataFrame(X).iloc[:, 0].astype(object)
    s = s.where(s.notna(), "").astype(str).str.strip()
    empty = (s == "").to_numpy(dtype=float)
    has_digit = s.str.contains(r"\d", regex=True).to_numpy(dtype=float)
    return np.column_stack([empty, has_digit])


def _num(prefix_fn):
    return Pipeline([
        ("cast", FunctionTransformer(prefix_fn)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])


def build_pipeline(seed: int = 0) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cod", _num(_to_log_float), ["cod_amount"]),
            ("num", _num(_to_float), ["n_items", "customer_prior_parcels", "customer_prior_fail_rate"]),
            ("cat", Pipeline([
                ("clean", FunctionTransformer(_to_clean_str)),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), _CATEGORICAL),
            ("addr", FunctionTransformer(_address_features), ["address_text"]),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(C=0.3, max_iter=2000, random_state=seed)),
    ])


def evaluate(model, test_df: pd.DataFrame) -> dict:
    y = test_df[TARGET]
    p = model.predict_proba(test_df)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(np.mean(y)),
        "n": int(len(test_df)),
    }


def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict:
    model = build_pipeline(seed).fit(train_df, train_df[TARGET])
    dummy = DummyClassifier(strategy="prior").fit(train_df, train_df[TARGET])
    m, d = evaluate(model, test_df), evaluate(dummy, test_df)
    return {"model": m, "dummy": d, "lift_pr_auc": m["pr_auc"] - d["pr_auc"]}