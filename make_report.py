"""Makes plots + REPORT.md (filled with your real numbers). Run after run_experiments.py"""
import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, json, re
m = pd.read_csv("results/metrics.csv"); b = pd.read_csv("results/budget_curve.csv", index_col=0)
w = json.load(open("results/chosen_weight.json"))["hybrid_weight"]
log = open("results/log.txt").read()
ax = b.plot(marker="o", figsize=(8, 5)); ax.set_xlabel("Fraction of claims reviewed"); ax.set_ylabel("Recall of true anomalies")
ax.set_title("Auditor-budget curve"); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig("results/budget_curve.png", dpi=150); plt.close()
cols = [c for c in m.columns if c.startswith("rec_")]
ax = m.set_index("model")[cols].rename(columns=lambda c: c[4:]).plot.bar(figsize=(11, 5))
ax.set_ylabel("Recall in top 5%"); ax.set_title("Per-anomaly-type recall"); plt.xticks(rotation=25, ha="right")
plt.tight_layout(); plt.savefig("results/per_type_recall.png", dpi=150); plt.close()
expl = re.search(r"Explanation check.*", log).group(0)
table = m[["model", "PR_AUC", "precision_at_5pct", "recall_at_5pct"]].to_markdown(index=False) if hasattr(m, "to_markdown") else m.to_string(index=False)
open("REPORT.md", "w").write(f"""# TrustLens: Explainable ML for Detecting Suspicious Expense Claims

## 1. Problem and objective
Auditors cannot review every expense claim. TrustLens ranks claims by how unusual they are, explains each flag in plain
language, and helps prioritise a limited review budget. Output means "worth a look", never "fraud".

## 2. Data (synthetic; see data card)
- ~20.8k claims, 250 employees, 4 roles, 5 departments, 7 categories, amounts in INR.
- ~4% injected anomalies of 6 labelled types: duplicate, split-under-limit, round-amount, inflated vs peers,
  unusual weekend, repeated-vendor. No real employee data used.
- Split by time: train < Aug 2025, validation Aug-Sep, test >= Oct 2025.

## 3. Method
Features: peer-relative amount ratio/z-score, weekend/month-end, round/near-limit flags, rolling claim counts,
duplicates, same-day total, vendor share. Models: amount rule, per-group z-score, Isolation Forest, Hybrid
(IF rank + {w} x policy-rule hit, weight chosen on validation), supervised gradient boosting (upper bound) and a
supervised model with two anomaly types hidden from training.

## 4. Results (test period)
{table}

![budget](results/budget_curve.png)
![per type](results/per_type_recall.png)

Explainability: {expl}

## 5. Error analysis (TODO: write 5-6 lines after inspecting 20 missed claims)

## 6. Ethics, bias and safety
- Anomaly != fraud: unusual but legitimate claims (e.g. a senior leader's client dinner) will be flagged.
- Human-in-the-loop only; no automatic penalties. Show reasons so employees can contest flags.
- Privacy: real deployment needs access controls, data minimisation and retention limits.
- Bias: check flag rates by role, department and city; peer baselines may penalise small groups.
- Dual use: injection logic could teach evasion; share only abstracted descriptions.

## 7. Limitations
Synthetic data, injected anomalies are simpler than real fraud, features use full-dataset statistics
(peer medians, vendor share), single random seed, no real auditor feedback.
""")
print("Wrote REPORT.md and 2 plots in results/")
