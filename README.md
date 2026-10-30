# MLflow hands-on lab · Predicting customer churn

A practical, step-by-step lab (about 4 hours) to learn **MLflow 3** by doing.
You will take one machine-learning model through its whole life: track experiments, tune it,
package it, register it, serve it as an API, detect when it stops working, retrain it and promote
the new version safely.

Everything runs **on your laptop**. No cloud account, no Docker, no GPU.

---

## Table of contents

1. [The scenario](#1--the-scenario)
2. [What you will learn](#2--what-you-will-learn)
3. [What is in this repository](#3--what-is-in-this-repository)
4. [Prerequisites](#4--prerequisites)
5. [Installation](#5--installation)
6. [The data](#6--the-data)
7. [MLflow in 5 minutes](#7--mlflow-in-5-minutes)
8. [Start MLflow and open the notebooks](#8--start-mlflow-and-open-the-notebooks)
9. [Lab 1 · Experiment tracking](#9--lab-1--experiment-tracking-50-min)
10. [Lab 2 · Hyperparameter tuning](#10--lab-2--hyperparameter-tuning-30-min)
11. [Lab 3 · Models, registry and serving](#11--lab-3--models-registry-and-serving-75-min)
12. [Lab 4 · Capstone: drift and retraining](#12--lab-4--capstone-drift-and-retraining-30-min)
13. [Bonus · Train from the command line](#13--bonus--train-from-the-command-line)
14. [Start over](#14--start-over)
15. [Troubleshooting](#15--troubleshooting)
16. [Going further](#16--going-further)

---

## 1 · The scenario

**StreamCo** is a (fictional) subscription service with 5,000 customers. Every month, the CRM
team sends retention offers to customers who are likely to cancel. Your job as the ML team is to
build the churn model behind that campaign — and to keep it working.

The story has a twist. In **July 2026** a competitor launches and StreamCo raises the price of its
cheapest plan (BASIC). Customer behaviour changes, and the model trained on February–June data
starts to fail in August. In Lab 4 you will detect this, retrain, and decide whether the new
model is good enough to replace the old one.

## 2 · What you will learn

| Lab | Topic | You will be able to… |
|---|---|---|
| 1 | Experiment tracking | log parameters, metrics, plots and datasets; compare runs in the UI and from code |
| 2 | Tuning | run a hyperparameter search with Optuna, organised as nested runs, and pick the winner automatically |
| 3 | Models, registry, serving | package a model with a signature, add business logic with a custom model, evaluate it, register versions, move aliases, batch-score and serve a REST API |
| 4 | Capstone | measure drift, confirm it with labels, retrain, compare champion vs challenger on fresh data and promote with a rollback path |

## 3 · What is in this repository

```
mlflow-lab/
├── README.md                 ← this guide
├── notebooks/
│   ├── 01_tracking.ipynb             Lab 1
│   ├── 02_tuning.ipynb               Lab 2
│   ├── 03_models_registry.ipynb      Lab 3
│   └── 04_capstone_retraining.ipynb  Lab 4
├── src/
│   ├── churn_data.py     generates the synthetic data (same as the CSV files)
│   ├── churn_model.py    the scikit-learn pipeline + ChurnDecisionModel (custom MLflow model)
│   └── train.py          command-line training script (used by the MLproject)
├── data/
│   ├── churn_history.csv  Feb–Jun 2026, 25,000 rows (train + validation)
│   ├── churn_july.csv     Jul 2026, 5,000 rows (batch scoring)
│   └── churn_august.csv   Aug 2026, 5,000 rows (after the market shock)
├── MLproject, python_env.yaml   reproducible project definition (bonus)
├── requirements.txt
└── solutions.md          answers to the "Your turn" exercises
```

## 4 · Prerequisites

| You need | Check with | Expected |
|---|---|---|
| Python **3.10 to 3.13** | `python --version` (or `python3 --version`) | `Python 3.1x.x` |
| Git | `git --version` | any version |
| About 1 GB free disk | — | for the Python packages |
| A web browser | — | Chrome, Firefox, Edge, Safari |

Basic Python and pandas are enough. You do **not** need to know MLflow.

You will use **three terminals** during the lab:

| Terminal | Runs | When |
|---|---|---|
| 1 | the MLflow tracking server | all the time |
| 2 | Jupyter (the notebooks) | all the time |
| 3 | the model API | Lab 3, step 7 only |

## 5 · Installation

Do this **before** the session if you can (≈10 minutes, mostly downloads).

### 5.1 Get the code

```bash
git clone https://github.com/aniskx/mlflow-lab.git
cd mlflow-lab
```

### 5.2 Create a virtual environment and install the packages

A virtual environment keeps the lab's packages separate from the rest of your computer.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> If PowerShell refuses to run the activation script, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

When the environment is active, your prompt starts with `(.venv)`.
**Every new terminal** needs `cd mlflow-lab` and the activation command again.

### 5.3 Check the installation

```bash
python -c "import mlflow, sklearn, xgboost, optuna; print('MLflow', mlflow.__version__)"
```

✅ Expected: `MLflow 3.17.2` or any newer `3.x`.

What you just installed:

| Package | Role in the lab |
|---|---|
| `mlflow` | tracking server, UI, model packaging, registry, serving |
| `scikit-learn` | preprocessing pipeline, logistic regression, metrics |
| `xgboost` | the gradient-boosted model we tune and ship |
| `optuna` | smart hyperparameter search (Lab 2) |
| `pandas`, `numpy`, `matplotlib` | data handling and plots |
| `jupyter` | runs the notebooks |

## 6 · The data

The data is **synthetic** (generated by `src/churn_data.py`), so it is free to use and identical
for everyone. The notebooks generate it in memory; the CSV files in `data/` contain exactly the
same rows if you want to explore them in pandas or Excel.

**One row = one customer at one monthly snapshot.** Behaviour columns describe the 30 days
*before* the snapshot; the label says whether the customer cancelled in the 30 days *after*.

| Column | Meaning |
|---|---|
| `CUSTOMER_ID` | customer identifier |
| `SNAPSHOT_TS` | snapshot date (1st of the month) |
| `LOGINS_30D` | number of logins in the last 30 days |
| `PURCHASES_30D` | number of purchases in the last 30 days |
| `SPEND_30D` | money spent in the last 30 days |
| `TICKETS_30D` | support tickets opened in the last 30 days |
| `TENURE_DAYS` | days since the customer signed up |
| `PLAN_TIER` | 0 = BASIC, 1 = STANDARD, 2 = PREMIUM |
| `COUNTRY` | FR, DE, ES, IT or UK |
| `CHURNED` | **target**: 1 if the customer cancelled in the next 30 days |

| File | Rows | Used for | Churn rate |
|---|---|---|---|
| `churn_history.csv` | 25,000 (5 snapshots × 5,000) | training (Feb–Apr) and validation (May–Jun) | 10.1% |
| `churn_july.csv` | 5,000 | batch scoring in Lab 3 | 9.2% |
| `churn_august.csv` | 5,000 | drift and retraining in Lab 4 | 25.0% |

**Why split by time?** In production a model always predicts the *future*. So we train on
February–April and validate on May–June, never on randomly shuffled rows. A random split would
let the model "see the future" and give over-optimistic results.

Quick look (optional):
```python
import pandas as pd
df = pd.read_csv("data/churn_history.csv")
df.groupby("PLAN_TIER")["CHURNED"].mean()   # cheaper plans churn more
```

To regenerate the CSV files: `python src/churn_data.py`.

## 7 · MLflow in 5 minutes

MLflow is an open-source platform for the machine-learning lifecycle. You use it from Python;
everything you log appears in a web UI.

### Architecture

```
  your notebook / script ──┐
                           ├──►  MLflow tracking server  ──►  backend store   (metadata: runs, params, metrics, registry)
  your browser (the UI) ───┘     http://127.0.0.1:5000   ──►  artifact store  (files: models, plots, datasets)
```

In this lab both stores are local: `mlflow.db` (a SQLite database) and the `mlartifacts/` folder.
In a company you would use PostgreSQL and cloud storage (S3, GCS, Azure Blob) — **the notebook
code does not change**, only the server configuration.

### Vocabulary

| Term | Meaning | Example in this lab |
|---|---|---|
| **Experiment** | a folder of runs answering one question | `churn-workshop` |
| **Run** | one execution of training code | `logreg_baseline` |
| **Parameter** | an input you chose (stored as text) | `max_depth = 3` |
| **Metric** | a number you measured, can have steps | `val_auc = 0.728` |
| **Tag** | a label to search and filter runs | `stage = baseline` |
| **Artifact** | any file attached to a run | `plots/roc_curve.png` |
| **Logged model** | a packaged model with its own ID | `models:/m-54082d…` |
| **Registered model** | a named model with numbered versions | `churn-model` v1, v2, v3 |
| **Alias** | a movable pointer to one version | `churn-model@champion` |

The three objects you will meet in Labs 3 and 4:

```
RUN  ───────────────►  LOGGED MODEL  ───────────────►  REGISTERED VERSION
how it was trained     what was produced               what production uses
(params, metrics…)     (weights, signature, deps)      (version number, aliases, tags)
```

## 8 · Start MLflow and open the notebooks

### 8.1 Terminal 1 — the tracking server (leave it running all day)

```bash
cd mlflow-lab
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

✅ Expected: the log ends with `Application startup complete.`
✅ Open **http://127.0.0.1:5000** in your browser — you should see the MLflow UI.

> **macOS:** if port 5000 is already used (AirPlay Receiver), use `--port 5050` and change
> `mlflow.set_tracking_uri("http://127.0.0.1:5000")` to `5050` in the first code cell of each notebook.

### 8.2 Terminal 2 — Jupyter

```bash
cd mlflow-lab
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
jupyter notebook notebooks/
```

Your browser opens the `notebooks/` folder. Open the notebooks **in order** (01 → 04) and run the
cells one by one with **Shift + Enter**. Read the text cells: they explain each step.

The first code cell of every notebook connects to the server:
```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("churn-workshop")
```
✅ It must print `MLflow 3.x`. If it hangs or fails, terminal 1 is not running.

---

## 9 · Lab 1 · Experiment tracking (50 min)

📓 `notebooks/01_tracking.ipynb`

**Goal:** turn every training attempt into an MLflow run, so nothing is lost and everything can
be compared.

### Step 1 · Load the data and split by time
The notebook loads Feb–Jun and splits it: train = Feb–Apr, validation = May–Jun.
✅ `(25000, 10) | churn rate: 0.101`

### Step 2 · Your first run: parameters, metrics, tags
```python
with mlflow.start_run(run_name="logreg_baseline"):
    model = baseline().fit(X_tr, y_tr)
    ...
    mlflow.log_params({...})      # what you chose
    mlflow.log_metrics({...})     # what you measured
    mlflow.set_tags({...})        # labels to search by
```
*Why:* a baseline gives you a reference score. Any fancier model must beat it.
`baseline()` is a scikit-learn **pipeline**: scaling + one-hot encoding of `COUNTRY` + logistic
regression. The preprocessing is part of the model, so it will travel with it later.

✅ In the UI: experiment `churn-workshop` → run `logreg_baseline` → val_auc ≈ **0.728**.

### Step 3 · Artifacts and datasets
You attach evidence to the run: a ROC curve, a confusion matrix, a feature list, a note, and the
**dataset** the model was trained on (`mlflow.log_input`).
*Why:* six months from now, you (or an auditor) must be able to see *why* this model was chosen
and *which data* it saw. MLflow stores a digest of the dataset: if the data changes, the digest changes.

✅ In the UI: run `logreg_with_artifacts` → **Artifacts** tab (plots, JSON, notes) and the
**Datasets** column (`churn_history`).

### Step 4 · Metrics over steps (a learning curve)
`mlflow.log_metric("val_auc", auc, step=n)` with a random forest of 10 → 200 trees.
*Why:* a metric logged with a step becomes a chart; you can see when adding trees stops helping.
✅ AUC rises from 0.722 to 0.727 and flattens. In the UI, `val_auc` is a curve.

### Step 5 · Autologging
`mlflow.sklearn.autolog()` logs parameters, training metrics and the model automatically.
*Watch out:* autolog records **training** metrics. A model can look perfect on training data and
fail on new data — so we still log the validation AUC ourselves.
✅ Run `xgb_autolog` with dozens of parameters; val_auc ≈ **0.720** (a deeper model is not automatically better).

### Step 6 · Query runs from code
```python
mlflow.search_runs(experiment_names=["churn-workshop"],
                   filter_string="metrics.val_auc > 0.5",
                   order_by=["metrics.val_auc DESC"])
```
*Why:* automated pipelines pick the best run with code, not by reading the UI.
✅ A table of 4 runs sorted by AUC.

### ✍️ Your turn (10 min)
1. Log a run with a `GradientBoostingClassifier` (AUC ≈ 0.738).
2. Select your 4 best runs in the UI → **Compare**. Which one would you ship, and why?
3. List only runs tagged `stage = baseline` with `search_runs`.

Answers: [`solutions.md`](solutions.md).

---

## 10 · Lab 2 · Hyperparameter tuning (30 min)

📓 `notebooks/02_tuning.ipynb`

**Goal:** search for better XGBoost settings without drowning in hundreds of runs.

### Step 1 · One trial = one child run
Each Optuna trial trains a model and logs its parameters and AUC in a run started with
`nested=True`.

### Step 2 · The search = one parent run
```python
with mlflow.start_run(run_name="xgb_optuna_search"):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=15)
```
*Why:* the **parent** keeps the summary (best score, best parameters); the **children** keep the
evidence. Optuna chooses each new trial based on the previous ones, which beats random search.
Takes 1–2 minutes.
✅ `best AUC: 0.7388 {'max_depth': 2, 'n_estimators': 400, ...}`

✅ In the UI: expand `xgb_optuna_search`, select all 15 children → **Compare** →
**Parallel coordinates**. Shallow trees with a small learning rate win on this data.

### Step 3 · Find the best trial from code
Child runs carry the tag `mlflow.parentRunId`, so you can query all trials of one search.
✅ `trial_6 0.7388 {...}`

### ✍️ Your turn (5 min)
Parameters come back from `search_runs` as **strings**. Convert them to `int`/`float`.
Why do you think MLflow stores parameters as text?

---

## 11 · Lab 3 · Models, registry and serving (75 min)

📓 `notebooks/03_models_registry.ipynb`

**Goal:** go from "a model in my notebook" to "a versioned model anyone can call".

### Step 1 · Log a model with its contract
```python
xgb_info = mlflow.sklearn.log_model(
    xgb, name="churn_xgb",
    signature=infer_signature(X_va, proba),   # input/output schema
    input_example=X_va.head(3),               # what a valid request looks like
    skops_trusted_types=XGB_TRUSTED_TYPES)    # see note below
```
- The **signature** is the model's contract: column names and types in, prediction out.
- MLflow 3 gives the model its own ID: `models:/m-…`.
- **Note:** MLflow saves scikit-learn models with *skops*, a safe format that cannot run hidden
  code when loading. Types from other libraries (XGBoost) must be trusted explicitly — that is
  what `skops_trusted_types` does.

✅ val_auc ≈ **0.735**. In the UI: experiment → **Models** tab → `churn_xgb` (signature, files, linked run).

### Step 2 · Load it back as a generic `pyfunc`
`mlflow.pyfunc.load_model(uri).predict(df)` works the same for every library — that is what
serving uses. The signature is enforced: send data without `TICKETS_30D` and it is rejected.
✅ `[0 0 0 0 0]` then `Rejected: MlflowException … Model is missing inputs ['TICKETS_30D']`

*Catch:* the generic `predict` returns 0/1 labels, not probabilities. The CRM team needs more.

### Step 3 · A custom model: ship the business decision with the model
`ChurnDecisionModel` (in `src/churn_model.py`) wraps the classifier and returns, for each customer:
`churn_probability`, `risk_band` (LOW / MEDIUM / HIGH) and `send_offer` (True/False).
The decision **threshold** is a model parameter (default 0.35) that callers can change per request.
*Why:* business rules live with the model and are versioned with it, instead of being hidden in an application.
✅ A small table; with `threshold=0.1` more customers move to MEDIUM / HIGH.

### Step 4 · Evaluate like a reviewer
`mlflow.models.evaluate(...)` computes standard metrics and plots and logs them.
✅ accuracy **0.907**, roc_auc **0.735**, recall **0.169**.
*Discuss:* 91% accuracy but only 17% of churners caught. Why? Only ~10% of customers churn, so
predicting "nobody churns" already gives 90% accuracy. Accuracy is misleading on imbalanced data.

### Step 5 · Register versions and use aliases
```python
mlflow.sklearn.log_model(lr, name="churn_logreg", registered_model_name="churn-model")  # → version 1
mlflow.register_model(xgb_info.model_uri, "churn-model")                                # → version 2
client.set_registered_model_alias("churn-model", "champion", 2)
champion = mlflow.pyfunc.load_model("models:/churn-model@champion")
```
- A **version** is immutable history. An **alias** is a decision you can change.
- Applications always load `@champion`. Promotion and rollback = moving the alias, no code change.
- (Older tutorials use "stages" like Staging/Production — they are deprecated; use aliases.)

✅ `versions: 1 2` and `champion is version 2`. In the UI: top bar → **Models** → `churn-model`.

### Step 6 · Batch-score July
The decision model is registered as `churn-decision` and scores the 5,000 July customers.
✅ `HIGH 286 · MEDIUM 169 · LOW 4545` → **286 offers to send**. Results saved to `data/scores_july.csv`.

### Step 7 · Serve the model as a REST API

Open **terminal 3**:

**macOS / Linux**
```bash
cd mlflow-lab && source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve -m "models:/churn-decision@champion" -p 5001 --env-manager local
```
**Windows (PowerShell)**
```powershell
cd mlflow-lab; .venv\Scripts\Activate.ps1
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
mlflow models serve -m "models:/churn-decision@champion" -p 5001 --env-manager local
```
✅ `Uvicorn running on http://127.0.0.1:5001`. Health check: http://127.0.0.1:5001/ping

`--env-manager local` reuses your current Python environment (fast for a lab). In production you
would let MLflow build the model's own environment, or package it with `mlflow models build-docker`.

Then run the last code cell of the notebook, or call the API with curl (macOS / Linux):
```bash
curl -X POST http://127.0.0.1:5001/invocations -H "Content-Type: application/json" -d '{
  "dataframe_split": {
    "columns": ["LOGINS_30D","PURCHASES_30D","SPEND_30D","TICKETS_30D","TENURE_DAYS","PLAN_TIER","COUNTRY"],
    "data": [[6, 1, 42.5, 5, 120, 0, "FR"], [30, 5, 410.0, 0, 900, 2, "DE"]]},
  "params": {"threshold": 0.3}}'
```
✅ The first customer (few logins, 5 tickets, BASIC) gets `"risk_band": "HIGH", "send_offer": true`
(probability ≈ 0.86); the second ≈ 0.02, LOW.

### ✍️ Your turn (10 min)
1. Call the API with curl.
2. **Rollback drill:** point `churn-model@champion` to version 1, check what loads, then move it back to 2.
3. Add an `approved_by` tag to version 2. Where would that matter in a real company?

> You can stop the API now (Ctrl + C in terminal 3); Lab 4 does not need it.

---

## 12 · Lab 4 · Capstone: drift and retraining (30 min)

📓 `notebooks/04_capstone_retraining.ipynb` — **run Lab 3 first**: Lab 4 uses its `@champion` alias.

**Goal:** August data arrives with labels. Detect the problem, retrain, and promote only if the
new model is genuinely better.

### Step 1 · Detect drift with the PSI
The **Population Stability Index** compares each feature's distribution in August with May–June.
Rule of thumb: < 0.1 stable · 0.1–0.25 investigate · > 0.25 act.
✅ LOGINS_30D **2.48**, TICKETS_30D **0.84**, PURCHASES_30D **0.55** → act. PLAN_TIER **0.00** → stable.

### Step 2 · Confirm with labels
✅ Champion AUC: **0.735** in May–June → **0.660** in August.

*Notice:* `PLAN_TIER` did not drift, yet the price change hit BASIC customers. That is **concept
drift**: same inputs, different outcome. Only labels reveal it. Input drift is an early warning;
performance on labels is the confirmation.

### Step 3 · Retrain and gate, in one tracked run
Inside the run `retrain_2026_08`:
1. log the drift numbers and the new training dataset;
2. train a **challenger** on history + 70% of August;
3. score champion and challenger on the **same fresh holdout** (the other 30% of August, used by neither model);
4. pass the **gate** only if the challenger beats the champion by at least 0.01 AUC;
5. register the challenger as a new version either way (it is evidence).

✅ `champion 0.651 vs challenger 0.723 -> gate PASSED`, version **3** registered.

### Step 4 · Promote, with a way back
```
@champion   -> version 3   (new)
@challenger -> version 3
@previous   -> version 2   (rollback in one call)
@baseline   -> version 1
```
✅ In the UI, open run `retrain_2026_08`: drift file, dataset, both AUCs, tag `gate=passed` and
the link to version 3. **That run is your audit trail.**

### ✍️ Discussion (5 min)
1. Is retraining enough, or does the price change call for a new feature?
2. Roll back to `@previous` in one line.
3. Which parts would you schedule automatically? Which should stay a human decision?

---

## 13 · Bonus · Train from the command line

Notebooks are for exploring; scripts with arguments are what CI pipelines and schedulers run.

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000        # Windows: $env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

# a plain script with arguments
python src/train.py --max-depth 3 --n-estimators 300

# the same, as an MLflow Project (reads the MLproject file)
mlflow run . -P max_depth=5 --env-manager local --experiment-name churn-workshop
```
✅ `{'val_auc': 0.7277, ...}`, then `{'val_auc': 0.7103, ...}` and `Run (ID '…') succeeded`.
Add `--register` to `train.py` to register the model in `churn-model`.

## 14 · Start over

1. Stop the servers (Ctrl + C in terminals 1 and 3).
2. Delete `mlflow.db` and the `mlartifacts/` folder.
3. Start the tracking server again (section 8.1) and re-run the notebooks from Lab 1.

## 15 · Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Connection refused` on port 5000, or the first cell hangs | the tracking server (terminal 1) is not running, or uses another port |
| Port 5000 already in use (macOS) | start the server with `--port 5050` and update `set_tracking_uri` in the notebooks |
| `ModuleNotFoundError: churn_data` | start Jupyter from the repository folder: `jupyter notebook notebooks/` |
| `ModuleNotFoundError: mlflow` | the virtual environment is not activated in this terminal |
| `untrusted types` when logging XGBoost | keep `skops_trusted_types=XGB_TRUSTED_TYPES` (already in the notebooks) |
| `evaluate()` error mentioning `StringDtype` | keep the `.astype({"COUNTRY": object})` line (pandas 3 compatibility) |
| `Run ... is already active` | run `mlflow.end_run()` in a cell, then retry |
| Changing a parameter value fails | parameters cannot change inside a run: start a new run |
| Lab 4: `alias champion not found` | run Lab 3 first, against the same tracking server |
| Serving: connection refused on 5001 | set `MLFLOW_TRACKING_URI` in terminal 3 before `mlflow models serve` |
| Windows: activation script blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

## 16 · Going further

- Add a segment check to the Lab 4 gate: the challenger must not be worse on BASIC customers.
- Add a `@shadow` alias and score July with both champion and shadow to compare.
- Run the tracking server with PostgreSQL and S3-compatible storage (e.g. MinIO).
- Explore MLflow's GenAI features: tracing, LLM evaluation, prompt registry.

**Documentation**
- [MLflow 3: what changed](https://mlflow.org/docs/latest/ml/mlflow-3/)
- [Model Registry](https://mlflow.org/docs/latest/ml/model-registry)
- [Self-hosting the tracking server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server)
- [Logging MLflow models in Snowflake's Model Registry](https://docs.snowflake.com/developer-guide/snowflake-ml/model-registry/built-in-models/mlflow)

---

*Tested end to end with Python 3.13, MLflow 3.17.2, scikit-learn 1.9, XGBoost 3.4, pandas 3.1, Optuna 5.0 (October 2026).
Numbers marked ✅ come from that run; the data is seeded, so you should get the same values (tiny differences are possible across library versions).*
