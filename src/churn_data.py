"""Synthetic StreamCo churn data — same story as the Snowflake lab.

Each customer has a hidden churn risk. Risky customers log in less, buy less and
open more support tickets. One row = one customer at one monthly snapshot, with
features computed over the 30 days before the snapshot and the label
"churned in the 30 days after".

drift=True simulates the July market shock from the Snowflake lab:
  * data drift   — everyone logs in less, tickets spike
  * concept drift — a price increase: established BASIC customers churn whatever
                    their behaviour, so the old behaviour-based patterns stop working
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["LOGINS_30D", "PURCHASES_30D", "SPEND_30D", "TICKETS_30D",
            "TENURE_DAYS", "PLAN_TIER", "COUNTRY"]
NUMERIC = [f for f in FEATURES if f != "COUNTRY"]
TARGET = "CHURNED"


def _customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    plan_tier = rng.integers(0, 3, n)                      # 0 BASIC, 1 STANDARD, 2 PREMIUM
    tenure0 = rng.integers(30, 1500, n)                    # days of tenure on 2026-01-01
    country = rng.choice(["FR", "DE", "ES", "IT", "UK"], n)
    risk = (rng.uniform(0, 0.6, n) + np.where(plan_tier == 0, 0.2, 0)
            - np.minimum(tenure0, 1000) / 5000)
    return pd.DataFrame({"CUSTOMER_ID": np.arange(1, n + 1), "PLAN_TIER": plan_tier,
                         "TENURE0": tenure0, "COUNTRY": country,
                         "_RISK": np.clip(risk, 0.02, 0.9)})


def make_churn(snapshots: list[str], n_customers: int = 5000, drift: bool = False,
               seed: int = 42) -> pd.DataFrame:
    """Return one row per (customer, snapshot)."""
    rng = np.random.default_rng(seed)
    cust = _customers(n_customers, rng)
    frames = []
    for snap in snapshots:
        ts = pd.Timestamp(snap)
        r = cust["_RISK"].to_numpy()
        activity = (0.6 - 0.5 * r) if drift else (1 - 0.85 * r)
        logins = rng.poisson(25 * activity)
        purchases = rng.poisson(4 * activity)
        spend = np.round(purchases * rng.uniform(5, 150, len(r)), 2)
        tickets = rng.poisson((1.5 + 6 * r) if drift else (0.3 + 4 * r))
        tenure = cust["TENURE0"].to_numpy() + (ts - pd.Timestamp("2026-01-01")).days
        basic = cust["PLAN_TIER"].to_numpy() == 0
        frustrated = (tickets >= 3) & (logins < 14)       # many tickets AND few logins
        if drift:   # price shock: established BASIC customers leave, whatever their behaviour
            p_churn = 0.08 + 0.05 * r + 0.05 * frustrated + 0.45 * (basic & (tenure >= 240))
        else:       # normal world: risk, frustration, and new BASIC customers
            p_churn = 0.02 + 0.20 * r + 0.35 * frustrated + 0.40 * (basic & (tenure < 240))
        frames.append(pd.DataFrame({
            "CUSTOMER_ID": cust["CUSTOMER_ID"],
            "SNAPSHOT_TS": ts,
            "LOGINS_30D": logins,
            "PURCHASES_30D": purchases,
            "SPEND_30D": spend,
            "TICKETS_30D": tickets,
            "TENURE_DAYS": tenure,
            "PLAN_TIER": cust["PLAN_TIER"],
            "COUNTRY": cust["COUNTRY"],
            TARGET: (rng.uniform(0, 1, len(r)) < np.clip(p_churn, 0, 0.95)).astype(int),
        }))
    return pd.concat(frames, ignore_index=True)


def history() -> pd.DataFrame:
    """Feb–Jun 2026: the data you train and validate on."""
    return make_churn(["2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01"])


def july() -> pd.DataFrame:
    """Jul 1 2026: a normal month (same world as training)."""
    return make_churn(["2026-07-01"], seed=7)


def august() -> pd.DataFrame:
    """Aug 1 2026: after the market shock (data + concept drift)."""
    return make_churn(["2026-08-01"], drift=True, seed=8)


def time_split(df: pd.DataFrame, cutoff: str = "2026-05-01"):
    """Train on snapshots before the cutoff, validate on the rest. Never shuffle time."""
    train, valid = df[df.SNAPSHOT_TS < cutoff], df[df.SNAPSHOT_TS >= cutoff]
    return train[FEATURES], train[TARGET], valid[FEATURES], valid[TARGET]


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    history().to_csv("data/churn_history.csv", index=False)
    july().to_csv("data/churn_july.csv", index=False)
    august().to_csv("data/churn_august.csv", index=False)
    print("Wrote data/churn_history.csv, data/churn_july.csv, data/churn_august.csv")
