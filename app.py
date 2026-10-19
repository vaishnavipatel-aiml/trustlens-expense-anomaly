"""TrustLens auditor dashboard. Run: streamlit run app.py"""
import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from features import build_features, FEATURES

st.set_page_config(page_title="TrustLens", page_icon="🔍", layout="wide")
st.title("🔍 TrustLens: claims prioritised for human review")
st.caption("Synthetic data. A high risk score means 'worth a look', not 'fraud'.")


@st.cache_resource
def load():
    import generate_data
    if not os.path.exists("data/claims.csv"):
        generate_data.main()  # first run on a fresh server: create the synthetic data
    df = build_features(pd.read_csv("data/claims.csv"))
    iso = IsolationForest(n_estimators=300, random_state=42).fit(df[FEATURES])
    df["risk"] = (-iso.score_samples(df[FEATURES]))
    df["risk"] = (df.risk.rank(pct=True) * 100).round(1)
    return df, iso, df[FEATURES].median()


df, iso, med = load()

# Plain-English sentence for each feature the model can point to
TEXT = {
    "amt_ratio_peer": lambda v: f"The amount is {v:.1f}x the typical amount for this role and category.",
    "z_peer": lambda v: f"The amount is {abs(v):.1f} standard deviations {'above' if v > 0 else 'below'} similar claims.",
    "is_round": lambda v: "The amount is a round number, which can suggest an estimate rather than a real receipt.",
    "near_limit": lambda v: "The amount sits just below the ₹5,000 approval limit.",
    "dup_count": lambda v: f"{v:.0f} other claim(s) from this employee have exactly the same amount.",
    "weekend_unusual": lambda v: "It was claimed on a weekend in a category that is rarely claimed on weekends.",
    "is_weekend": lambda v: "It was claimed on a weekend.",
    "is_month_end": lambda v: "It was claimed at the end of the month.",
    "vendor_share": lambda v: f"{v:.0%} of this employee's claims go to this one vendor.",
    "vendor_claims_30d": lambda v: f"This employee made {v:.0f} claims at this vendor within 30 days.",
    "same_day_total": lambda v: f"The employee's claims that day add up to ₹{v:,.0f}.",
    "emp_claims_7d": lambda v: f"The employee filed {v:.0f} claims within 7 days.",
    "log_amount": lambda v: "The absolute amount is high compared with all claims.",
}


def top_reasons(row, k=3):
    """Features whose replacement by a typical value lowers the anomaly score the most."""
    x = row[FEATURES].to_frame().T.astype(float)
    base = -iso.score_samples(x)[0]
    trial = pd.concat([x] * len(FEATURES), ignore_index=True)
    for i, f in enumerate(FEATURES):
        trial.loc[i, f] = med[f]
    drops = base + iso.score_samples(trial)
    return [FEATURES[i] for i in drops.argsort()[::-1][:k]]


# ---- Filters ----
c1, c2, c3 = st.columns(3)
dept = c1.multiselect("Department", sorted(df.dept.unique()), default=list(df.dept.unique()))
cat = c2.multiselect("Category", sorted(df.category.unique()), default=list(df.category.unique()))
budget = c3.slider("Review budget (% of claims)", 1, 20, 5)

v = df[df.dept.isin(dept) & df.category.isin(cat)].sort_values("risk", ascending=False)
v = v.head(max(1, int(len(v) * budget / 100))).copy()
v["date"] = v["date"].dt.date

m1, m2, m3 = st.columns(3)
m1.metric("Claims in review queue", f"{len(v):,}")
m2.metric("Total value in queue", f"₹{v.amount.sum():,.0f}")
m3.metric("Highest risk score", f"{v.risk.max():.1f}")

st.dataframe(
    v[["claim_id", "emp_id", "dept", "category", "vendor", "date", "amount", "risk"]],
    hide_index=True, width="stretch",
    column_config={
        "amount": st.column_config.NumberColumn("Amount (₹)", format="%.0f"),
        "risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.1f"),
    },
)

# ---- Explanation ----
st.subheader("Why was this flagged?")
if v.empty:
    st.info("No claims match the filters.")
    st.stop()

label = {r.claim_id: f"#{r.claim_id}  ·  {r.category}  ·  ₹{r.amount:,.0f}" for r in v.itertuples()}
sel = st.selectbox("Pick a claim to explain", list(label), format_func=label.get)
row = df[df.claim_id == sel].iloc[0]
reasons = top_reasons(row)

with st.container(border=True):
    st.markdown(f"**Claim #{sel}** · {row.category} · {row.dept} · employee {row.emp_id} · ₹{row.amount:,.0f}")
    st.markdown(f"This claim ranks in the **top {max(100 - row.risk, 0.1):.1f}%** of all claims for review. Main reasons:")
    for f in reasons:
        st.markdown(f"- {TEXT[f](row[f])}")
    st.caption("These are signals to check, not proof of wrongdoing. Verify the receipt and ask the manager before concluding anything.")

    # Optional AI-written note: only appears if an Anthropic key is set in Streamlit secrets
    try:
        key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        key = None
    if key and st.button("✨ Generate AI explanation"):
        try:
            import anthropic
            prompt = (f"You help a corporate expense auditor. In under 90 words, plain English, explain why this claim is "
                      f"unusual and give 2 verification steps. Never accuse anyone of fraud.\n"
                      f"Claim: {row.category}, Rs {row.amount:,.0f}, department {row.dept}.\n"
                      f"Signals: {' '.join(TEXT[f](row[f]) for f in reasons)}")
            msg = anthropic.Anthropic(api_key=key).messages.create(
                model="claude-sonnet-5-5", max_tokens=300, messages=[{"role": "user", "content": prompt}])
            st.info(msg.content[0].text)
        except Exception as e:
            st.warning(f"AI explanation unavailable: {e}")

with st.expander("Show all feature values for this claim"):
    st.dataframe(row[FEATURES].rename("value").to_frame(), width="stretch")
