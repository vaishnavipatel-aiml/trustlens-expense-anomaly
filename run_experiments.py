"""Baselines, evaluation, explanations. Run: python run_experiments.py"""
import numpy as np, pandas as pd, json, os
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from features import build_features, FEATURES, LIMIT
SEED = 42; TOP_FRAC = 0.05
df = build_features(pd.read_csv("data/claims.csv"))
# time-based split: train < Aug, validation Aug-Sep, test >= Oct (no future leakage)
tr = df[df.date < "2025-08-01"].copy()
va = df[(df.date >= "2025-08-01") & (df.date < "2025-10-01")].copy()
te = df[df.date >= "2025-10-01"].copy()

def prec_at_k(y, s, k): idx = np.argsort(-s)[:k]; return y[idx].mean()
def rec_at_frac(y, s, f):
    k = int(len(y) * f); idx = np.argsort(-s)[:k]; return y[idx].sum() / max(y.sum(), 1)
def per_type(te, s, f=TOP_FRAC):
    k = int(len(te) * f); top = set(np.argsort(-s)[:k]); out = {}
    for t in te.anomaly_type.unique():
        if t == "normal": continue
        ix = np.where(te.anomaly_type.values == t)[0]
        out[t] = round(np.mean([i in top for i in ix]), 3)
    return out

scores = {}
# 1) rule: amount over a fixed threshold
scores["Rule: amount>limit"] = te.amount.values
# 2) per-group z-score
scores["Z-score (role x category)"] = te.z_peer.abs().values
# 3) Isolation Forest (unsupervised)
iso = IsolationForest(n_estimators=300, random_state=SEED).fit(tr[FEATURES])
scores["Isolation Forest"] = -iso.score_samples(te[FEATURES])
# 4) Hybrid: IF rank + small bonus for policy-rule hits; weight chosen on VALIDATION only
rank = lambda x: pd.Series(x).rank(pct=True).values
def rules(d): return ((d.near_limit == 1) | (d.dup_count > 0) | (d.is_round == 1)).astype(float).values
iso_va = -iso.score_samples(va[FEATURES])
best_w, best_r = 0.0, -1
for w in [0.0, 0.03, 0.06, 0.1, 0.2, 0.3]:
    r = rec_at_frac(va.is_anomaly.values, rank(iso_va) + w * rules(va), TOP_FRAC)
    if r > best_r: best_w, best_r = w, r
print(f"Hybrid rule weight chosen on validation: {best_w} (val recall@5% = {best_r:.3f})")
os.makedirs("results", exist_ok=True); json.dump({"hybrid_weight": best_w}, open("results/chosen_weight.json", "w"))
scores["Hybrid (rules + IF)"] = rank(scores["Isolation Forest"]) + best_w * rules(te)
# 5) Supervised upper bound, and HELD-OUT anomaly types test
sup = HistGradientBoostingClassifier(random_state=SEED).fit(tr[FEATURES], tr.is_anomaly)
scores["HistGB supervised (upper bound)"] = sup.predict_proba(te[FEATURES])[:, 1]
HOLD = ["split_under_limit", "vendor_repeat"]
tr_h = tr[~tr.anomaly_type.isin(HOLD)]
sup_h = HistGradientBoostingClassifier(random_state=SEED).fit(tr_h[FEATURES], tr_h.is_anomaly)
scores[f"Supervised, unseen types held out"] = sup_h.predict_proba(te[FEATURES])[:, 1]

y = te.is_anomaly.values; k = int(len(te) * TOP_FRAC); rows = []
for name, s in scores.items():
    rows.append(dict(model=name, PR_AUC=round(average_precision_score(y, s), 3),
        precision_at_5pct=round(prec_at_k(y, s, k), 3), recall_at_5pct=round(rec_at_frac(y, s, TOP_FRAC), 3),
        **{f"rec_{t}": v for t, v in per_type(te, s).items()}))
res = pd.DataFrame(rows); os.makedirs("results", exist_ok=True)
res.to_csv("results/metrics.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(f"test rows={len(te)}, anomalies={y.sum()} ({y.mean():.3f})\n"); print(res.to_string(index=False))

# Auditor-budget curve: recall vs fraction reviewed
fr = [0.01, 0.02, 0.05, 0.10, 0.20]
bud = pd.DataFrame({n: [rec_at_frac(y, s, f) for f in fr] for n, s in scores.items()}, index=fr).round(3)
bud.index.name = "review_budget"; bud.to_csv("results/budget_curve.csv"); print("\nRecall vs review budget\n", bud.to_string())

# ---- Explanations for Isolation Forest (simple, model-agnostic occlusion) ----
med = tr[FEATURES].median()
def explain(row_df):
    """Reason = features whose replacement by the typical value lowers the anomaly score most."""
    base = -iso.score_samples(row_df[FEATURES])[0]; contrib = {}
    for f in FEATURES:
        r = row_df[FEATURES].copy(); r[f] = med[f]; contrib[f] = base - (-iso.score_samples(r)[0])
    return base, sorted(contrib.items(), key=lambda kv: -kv[1])
TEMPLATES = {"amt_ratio_peer": "amount is {v:.1f}x the median for this role/category",
    "z_peer": "amount is {v:.1f} std-devs from role/category norm", "is_round": "round-number amount",
    "near_limit": "just below the approval limit", "dup_count": "{v:.0f} other claim(s) with identical amount",
    "weekend_unusual": "weekend claim in a mostly-weekday category", "vendor_share": "{v:.0%} of this employee's claims go to one vendor",
    "vendor_claims_30d": "{v:.0f} claims at same vendor in 30 days", "same_day_total": "same-day total of Rs {v:,.0f}",
    "emp_claims_7d": "{v:.0f} claims in 7 days", "log_amount": "high absolute amount", "is_weekend": "weekend claim", "is_month_end": "month-end claim"}
top_idx = np.argsort(-scores["Isolation Forest"])[:200]; agree = faith = n = 0
MAP = {"duplicate": "dup_count", "split_under_limit": "near_limit", "round_amount": "is_round",
       "inflated": "amt_ratio_peer", "weekend_unusual": "weekend_unusual", "vendor_repeat": "vendor_share"}
for i in top_idx:
    r = te.iloc[[i]]
    if r.is_anomaly.iloc[0] == 0: continue
    base, c = explain(r); top3 = [f for f, _ in c[:3]]; n += 1
    agree += MAP[r.anomaly_type.iloc[0]] in top3 or (r.anomaly_type.iloc[0] in ("inflated", "round_amount") and "z_peer" in top3)
    r2 = r.copy(); r2[c[0][0]] = med[c[0][0]]                # faithfulness: neutralise top feature
    faith += (-iso.score_samples(r2[FEATURES])[0]) < base
print(f"\nExplanation check on {n} flagged true anomalies: injected cause in top-3 reasons = {agree/n:.2f}; "
      f"neutralising top reason lowers score = {faith/n:.2f}")
print("\nExample flagged claim:")
i = [j for j in top_idx if te.iloc[j].is_anomaly == 1][0]; r = te.iloc[[i]]; base, c = explain(r)
print(r[["claim_id", "emp_id", "category", "amount", "anomaly_type"]].to_string(index=False))
for f, v in c[:3]: print(" -", TEMPLATES[f].format(v=r[f].iloc[0]))
