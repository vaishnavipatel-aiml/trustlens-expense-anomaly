"""ONE COMMAND: python run_all.py   (installs deps, generates data, runs experiments, makes plots + report)
Add --app to launch the dashboard afterwards."""
import subprocess, sys
def run(*a): print("\n>>", " ".join(a)); subprocess.check_call([sys.executable, *a])
import importlib.util as iu
need = [p for p, m in [("pandas","pandas"),("numpy","numpy"),("scikit-learn","sklearn"),("matplotlib","matplotlib"),("tabulate","tabulate"),("streamlit","streamlit")] if iu.find_spec(m) is None]
if need:
    print("Installing:", need)
    if subprocess.call([sys.executable, "-m", "pip", "install", "-q", *need]): print("pip install failed for some packages; continuing")
run("generate_data.py")
import os; os.makedirs("results", exist_ok=True)
print("\n>> run_experiments.py"); 
out = subprocess.run([sys.executable, "run_experiments.py"], capture_output=True, text=True)
print(out.stdout); open("results/log.txt", "w").write(out.stdout)
if out.returncode: print(out.stderr); sys.exit(1)
run("make_report.py")
if "--app" in sys.argv: run("-m", "streamlit", "run", "app.py")
else: print("\nDone. Open REPORT.md. Dashboard: python run_all.py --app  (or: streamlit run app.py)")
