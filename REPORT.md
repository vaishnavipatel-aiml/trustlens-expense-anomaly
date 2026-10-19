# TrustLens: Explainable ML for Detecting Suspicious Expense Claims

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
(IF rank + 0.06 x policy-rule hit, weight chosen on validation), supervised gradient boosting (upper bound) and a
supervised model with two anomaly types hidden from training.

## 4. Results (test period)
| model                             |   PR_AUC |   precision_at_5pct |   recall_at_5pct |
|:----------------------------------|---------:|--------------------:|-----------------:|
| Rule: amount>limit                |    0.118 |               0.145 |            0.172 |
| Z-score (role x category)         |    0.303 |               0.255 |            0.302 |
| Isolation Forest                  |    0.625 |               0.537 |            0.637 |
| Hybrid (rules + IF)               |    0.616 |               0.584 |            0.693 |
| HistGB supervised (upper bound)   |    0.96  |               0.804 |            0.953 |
| Supervised, unseen types held out |    0.76  |               0.592 |            0.702 |

![budget](results/budget_curve.png)
![per type](results/per_type_recall.png)

Explainability: Explanation check on 119 flagged true anomalies: injected cause in top-3 reasons = 1.00; neutralising top reason lowers score = 1.00

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
