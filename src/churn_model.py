"""Model building blocks shared by the notebooks, train.py and the MLproject."""
from __future__ import annotations

import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_data import NUMERIC

# MLflow 3 saves scikit-learn models with skops (safe loading, no arbitrary code).
# Types outside scikit-learn must be trusted explicitly when logging the model.
XGB_TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]


def make_pipeline(classifier) -> Pipeline:
    """Preprocessing travels WITH the model: no training-serving skew."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["COUNTRY"]),
    ])
    return Pipeline([("pre", pre), ("clf", classifier)])


def baseline() -> Pipeline:
    return make_pipeline(LogisticRegression(max_iter=1000))


class ChurnDecisionModel(mlflow.pyfunc.PythonModel):
    """Custom pyfunc: wraps the classifier and adds the business decision.

    Returns, for every customer, the churn probability, a risk band and whether
    the CRM team should send a retention offer. The threshold is a model
    parameter, versioned with the model, not hard-coded in an application.
    """

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def load_context(self, context):
        import mlflow.sklearn
        self.clf = mlflow.sklearn.load_model(context.artifacts["classifier"])

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None):
        threshold = (params or {}).get("threshold", self.threshold)
        proba = self.clf.predict_proba(model_input)[:, 1]
        # HIGH = above the threshold, MEDIUM = within half of it, LOW = the rest
        band = pd.cut(proba, [-np.inf, threshold / 2, threshold, np.inf], right=False,
                      labels=["LOW", "MEDIUM", "HIGH"])
        return pd.DataFrame({
            "churn_probability": proba.round(4),
            "risk_band": band.astype(str),
            "send_offer": proba >= threshold,
        })
