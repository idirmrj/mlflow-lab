# Solutions · "Your turn" exercises

## Lab 1 · Tracking

**1. Gradient boosting run**
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

with mlflow.start_run(run_name="gbm"):
    gbm = make_pipeline(GradientBoostingClassifier(n_estimators=200, max_depth=3)).fit(X_tr, y_tr)
    p = gbm.predict_proba(X_va)[:, 1]
    mlflow.log_param("model", "GradientBoosting")
    mlflow.log_metrics({"val_auc": roc_auc_score(y_va, p),
                        "val_pr_auc": average_precision_score(y_va, p)})
```

**2. Which run to ship?** Look beyond the top AUC: if the gap is under ~0.005 it is noise.
Prefer the simpler, faster, more explainable model (often the logistic baseline here), and check PR-AUC:
only ~10% of customers churn, so ranking the positives well matters more than overall AUC.

**3. Filter by tag**
```python
mlflow.search_runs(experiment_names=["churn-workshop"], filter_string="tags.stage = 'baseline'")
```

## Lab 2 · Tuning

```python
best_params = {
    "max_depth": int(best_params["max_depth"]),
    "n_estimators": int(best_params["n_estimators"]),
    "learning_rate": float(best_params["learning_rate"]),
    "subsample": float(best_params["subsample"]),
}
```
MLflow stores parameters as strings so one schema fits any value (numbers, lists, paths, class names)
and so they are searchable with simple filters. Metrics, by contrast, are always floats, because you plot,
compare and aggregate them.

## Lab 3 · Models and registry

**1. curl** — see the README, section 11, step 7.

**2. Rollback drill**
```python
client.set_registered_model_alias("churn-model", "champion", "1")
print(client.get_model_version_by_alias("churn-model", "champion").version)   # 1
client.set_registered_model_alias("churn-model", "champion", "2")
```
Anything loading `models:/churn-model@champion` picks up the change at its next load: no code change and
no redeployment of the calling application.

**3. Approval tag**
```python
client.set_model_version_tag("churn-model", "2", "approved_by", "risk-team")
```
In regulated settings (credit, insurance, health), a deployment pipeline can refuse to move `@champion`
to any version without an `approved_by` tag: a four-eyes principle enforced by automation.

## Lab 4 · Capstone (discussion)

1. **Retrain or new feature?** The August shock is a price change on BASIC. Retraining helps because
   `PLAN_TIER` and `TENURE_DAYS` already carry part of the signal, but a feature such as
   `PRICE_CHANGED_RECENTLY` or `MONTHLY_PRICE` would capture the cause directly and survive the next change.
2. **Rollback in one line**
   ```python
   client.set_registered_model_alias(NAME, "champion",
                                     client.get_model_version_by_alias(NAME, "previous").version)
   ```
3. **Automate vs human:** drift computation, retraining, evaluation on the holdout and registering the
   challenger can all run on a schedule. Moving `@champion` can be automatic for low-risk models behind a
   strict gate; for high-impact decisions keep a human approval (the `approved_by` tag from Lab 3).
