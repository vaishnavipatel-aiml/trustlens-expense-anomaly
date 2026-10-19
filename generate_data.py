"""Synthetic expense-claim generator with LABELLED injected anomalies.
Run: python generate_data.py  ->  data/claims.csv
"""
import numpy as np, pandas as pd, os
SEED = 42
rng = np.random.default_rng(SEED)
APPROVAL_LIMIT = 5000  # INR; claims above need manager approval

ROLES = {"Intern": 0.6, "Analyst": 1.0, "Manager": 1.6, "Director": 2.5}
DEPTS = ["Sales", "Engineering", "HR", "Finance", "Marketing"]
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Pune", "Jaipur"]
# base median amount (INR) and weekend probability per category
CATS = {"Meals": (450, .03), "Taxi": (350, .05), "Travel": (4000, .15),
        "Lodging": (3500, .12), "Software": (1500, .01),
        "Client Entertainment": (3000, .04), "Office Supplies": (800, .01)}
VENDORS = {c: [f"{c[:3].upper()}-V{i}" for i in range(12)] for c in CATS}

def make_employees(n=250):
    return pd.DataFrame({
        "emp_id": [f"E{i:04d}" for i in range(n)],
        "role": rng.choice(list(ROLES), n, p=[.2, .45, .25, .1]),
        "dept": rng.choice(DEPTS, n), "city": rng.choice(CITIES, n)})

def normal_claims(emps, n=20000):
    rows = []
    start = pd.Timestamp("2025-01-01")
    for _ in range(n):
        e = emps.iloc[rng.integers(len(emps))]
        cat = rng.choice(list(CATS))
        med, wk = CATS[cat]
        amt = med * ROLES[e.role] * rng.lognormal(0, 0.35)
        d = start + pd.Timedelta(days=int(rng.integers(0, 365)))
        # re-draw date to respect weekend probability
        want_wknd = rng.random() < wk
        while (d.dayofweek >= 5) != want_wknd:
            d = start + pd.Timedelta(days=int(rng.integers(0, 365)))
        rows.append(dict(emp_id=e.emp_id, date=d, category=cat,
                         vendor=rng.choice(VENDORS[cat]), amount=round(amt, 2),
                         is_anomaly=0, anomaly_type="normal"))
    return pd.DataFrame(rows)

def inject(df, emps, rate=0.04):
    """Append anomalies of 6 types (~rate of total)."""
    role = emps.set_index("emp_id").role
    new = []
    n_each = int(len(df) * rate / 6)
    for _ in range(n_each):  # 1 duplicate
        r = df.iloc[rng.integers(len(df))].copy()
        r["date"] += pd.Timedelta(days=int(rng.integers(0, 3)))
        r["is_anomaly"], r["anomaly_type"] = 1, "duplicate"; new.append(r)
    for _ in range(n_each // 2):  # 2 split claims just under the limit
        e = emps.emp_id.iloc[rng.integers(len(emps))]
        cat = rng.choice(["Client Entertainment", "Travel"]); v = rng.choice(VENDORS[cat])
        d = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 360)))
        for k in range(int(rng.integers(2, 4))):
            new.append(pd.Series(dict(emp_id=e, date=d + pd.Timedelta(days=k % 2), category=cat,
                vendor=v, amount=round(rng.uniform(4500, 4990), 2),
                is_anomaly=1, anomaly_type="split_under_limit")))
    for _ in range(n_each):  # 3 round amounts, inflated vs. normal
        r = df.iloc[rng.integers(len(df))].copy()
        r["amount"] = float(max(1000, round(r["amount"] * 1.6 / 500) * 500))
        r["is_anomaly"], r["anomaly_type"] = 1, "round_amount"; new.append(r)
    for _ in range(n_each):  # 4 inflated vs. peers
        r = df.iloc[rng.integers(len(df))].copy()
        r["amount"] = round(r["amount"] * rng.uniform(3, 6), 2)
        r["is_anomaly"], r["anomaly_type"] = 1, "inflated"; new.append(r)
    for _ in range(n_each):  # 5 weekend claim in a normally-weekday category
        r = df[df.category.isin(["Software", "Office Supplies", "Meals"])].iloc[
            rng.integers(len(df[df.category.isin(["Software", "Office Supplies", "Meals"])]))].copy()
        d = r["date"]; r["date"] = d + pd.Timedelta(days=(5 - d.dayofweek) % 7 + int(rng.integers(0, 2)))
        r["amount"] = round(r["amount"] * 1.8, 2)
        r["is_anomaly"], r["anomaly_type"] = 1, "weekend_unusual"; new.append(r)
    for _ in range(n_each // 8):  # 6 one employee hammering one vendor, inflated
        e = emps.emp_id.iloc[rng.integers(len(emps))]
        cat = rng.choice(list(CATS)); v = VENDORS[cat][0]
        base = CATS[cat][0] * ROLES[role[e]]
        d0 = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 330)))
        for _k in range(int(rng.integers(8, 13))):
            new.append(pd.Series(dict(emp_id=e, date=d0 + pd.Timedelta(days=int(rng.integers(0, 30))),
                category=cat, vendor=v, amount=round(base * rng.uniform(1.4, 1.9), 2),
                is_anomaly=1, anomaly_type="vendor_repeat")))
    out = pd.concat([df, pd.DataFrame(new)], ignore_index=True)
    return out.sort_values("date").reset_index(drop=True)

def main():
    emps = make_employees(); df = inject(normal_claims(emps), emps)
    df = df.merge(emps, on="emp_id"); df.insert(0, "claim_id", range(len(df)))
    os.makedirs("data", exist_ok=True); df.to_csv("data/claims.csv", index=False)
    print(df.shape, "anomaly rate:", round(df.is_anomaly.mean(), 4))
    print(df.anomaly_type.value_counts())


if __name__ == "__main__":
    main()
