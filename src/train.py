"""Command-line training job — the same code a scheduler or CI would run.

    python src/train.py --max-depth 4 --n-estimators 300
    mlflow run . -P max_depth=4 --env-manager local
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from churn_data import history, time_split
from churn_model import XGB_TRUSTED_TYPES, make_pipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--experiment", default="churn-workshop")
    p.add_argument("--register", action="store_true", help="register in the Model Registry")
    a = p.parse_args()

    mlflow.set_experiment(a.experiment)
    X_tr, y_tr, X_va, y_va = time_split(history())

    with mlflow.start_run(run_name=f"cli_xgb_d{a.max_depth}"):
        params = {"max_depth": a.max_depth, "n_estimators": a.n_estimators,
                  "learning_rate": a.learning_rate}
        mlflow.log_params(params)
        mlflow.set_tag("entrypoint", "train.py")
        model = make_pipeline(XGBClassifier(**params, eval_metric="logloss")).fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        metrics = {"val_auc": roc_auc_score(y_va, proba),
                   "val_pr_auc": average_precision_score(y_va, proba)}
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model, name="model",
            signature=infer_signature(X_va, proba),
            input_example=X_va.head(3),
            skops_trusted_types=XGB_TRUSTED_TYPES,
            registered_model_name="churn-model" if a.register else None,
        )
        print({k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
