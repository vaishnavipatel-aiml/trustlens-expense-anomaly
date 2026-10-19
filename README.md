# TrustLens
Explainable ranking of suspicious expense claims (synthetic data).

## Run everything (one command)
    python run_all.py          # installs missing libs, generates data, runs experiments, plots, writes REPORT.md
    python run_all.py --app    # same, then opens the dashboard

Dashboard only (after run_all.py): streamlit run app.py

Files: generate_data.py | features.py | run_experiments.py | make_report.py | app.py | run_all.py
Outputs: data/claims.csv, results/*.csv|png, REPORT.md (fill in section 5 yourself)
