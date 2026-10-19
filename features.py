"""Feature engineering: peer-relative, time, per-employee history, frequency."""
import numpy as np, pandas as pd
LIMIT = 5000
FEATURES = ["log_amount", "amt_ratio_peer", "z_peer", "is_weekend", "weekend_unusual",
            "is_month_end", "is_round", "near_limit", "emp_claims_7d", "dup_count",
            "same_day_total", "vendor_share", "vendor_claims_30d"]

def build_features(df):
    df = df.copy(); df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["log_amount"] = np.log1p(df.amount)
    g = df.groupby(["role", "category"])
    df["amt_ratio_peer"] = df.amount / g.amount.transform("median")
    df["z_peer"] = (df.log_amount - g.log_amount.transform("median")) / (g.log_amount.transform("std") + 1e-6)
    df["is_weekend"] = (df.date.dt.dayofweek >= 5).astype(int)
    cat_wk = df.groupby("category").is_weekend.transform("mean")
    df["weekend_unusual"] = df.is_weekend * (1 - cat_wk)
    df["is_month_end"] = (df.date.dt.is_month_end | (df.date.dt.day >= 28)).astype(int)
    df["is_round"] = ((df.amount % 100 == 0) & (df.amount >= 1000)).astype(int)
    df["near_limit"] = df.amount.between(0.9 * LIMIT, LIMIT - 0.01).astype(int)
    # rolling counts per employee (uses only past + same day)
    s = df.set_index("date")
    df["emp_claims_7d"] = s.groupby("emp_id").amount.transform(lambda x: x.rolling("7D").count()).values
    df["vendor_claims_30d"] = s.groupby(["emp_id", "vendor"]).amount.transform(lambda x: x.rolling("30D").count()).values
    df["dup_count"] = df.groupby(["emp_id", "amount"]).claim_id.transform("count") - 1
    df["same_day_total"] = df.groupby(["emp_id", "date"]).amount.transform("sum")
    df["vendor_share"] = df.groupby(["emp_id", "vendor"]).claim_id.transform("count") / \
                         df.groupby("emp_id").claim_id.transform("count")
    return df
