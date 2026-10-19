"""Auditor dashboard. Run: streamlit run app.py  (after generate_data.py)"""
import streamlit as st, pandas as pd, numpy as np
from sklearn.ensemble import IsolationForest
from features import build_features, FEATURES
st.set_page_config(page_title="TrustLens", layout="wide")
st.title("TrustLens: claims prioritised for human review")
st.caption("Synthetic data. A high score means 'worth a look', not 'fraud'.")

@st.cache_data
def load():
    import os, generate_data
    if not os.path.exists("data/claims.csv"):
        generate_data.main()   # first run on a fresh server: create the synthetic data
    df = build_features(pd.read_csv("data/claims.csv"))
    iso = IsolationForest(n_estimators=300, random_state=42).fit(df[FEATURES])
    df["risk"] = (-iso.score_samples(df[FEATURES])); df["risk"] = df.risk.rank(pct=True) * 100
    med = df[FEATURES].median()
    return df, iso, med
df, iso, med = load()

def reason(row):
    base = -iso.score_samples(row[FEATURES].to_frame().T.astype(float))[0]; out = []
    for f in FEATURES:
        r = row[FEATURES].copy(); r[f] = med[f]
        out.append((f, base + iso.score_samples(r.to_frame().T.astype(float))[0]))
    return [f for f, _ in sorted(out, key=lambda x: -x[1])[:3]]

c1, c2, c3 = st.columns(3)
dept = c1.multiselect("Department", sorted(df.dept.unique()), default=list(df.dept.unique()))
cat = c2.multiselect("Category", sorted(df.category.unique()), default=list(df.category.unique()))
budget = c3.slider("Review budget (% of claims)", 1, 20, 5)
v = df[df.dept.isin(dept) & df.category.isin(cat)].sort_values("risk", ascending=False)
v = v.head(max(1, int(len(v) * budget / 100)))
st.dataframe(v[["claim_id", "emp_id", "dept", "category", "vendor", "date", "amount", "risk"]], width="stretch")
sel = st.selectbox("Explain claim", v.claim_id.tolist())
row = v[v.claim_id == sel].iloc[0]
st.write("Top reasons:", ", ".join(reason(row)))
st.write(row[FEATURES])
