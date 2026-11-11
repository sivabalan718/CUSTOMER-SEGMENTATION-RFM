"""Re-run the whole project and every verification check, writing a JSON execution log.

1. Executes the notebooks in order (Phase 3 -> 8) with a Jupyter kernel and saves their outputs.
2. Runs independent_check.py (core Phases 3-5 from the raw Excel file), check_segments.py,
   sensitivity_analysis.py and, if Streamlit is installed, test_dashboard.py.

Works in the working layout (notebooks beside the data) and in the submission layout
(02_Notebooks / 03_Datasets_and_Results / 04_Verification / 05_Dashboard_Optional).
Usage:  python run_all.py            (from this folder or the project folder)
        python run_all.py --checks-only
"""
import datetime as dt
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = ["Phase3_Data_Preprocessing.ipynb", "Phase4_RFM_Feature_Engineering.ipynb",
             "Phase5_KMeans_Customer_Segmentation.ipynb",
             "Phase7_Customer_Segment_Interpretation.ipynb", "Phase8_AI_Business_Insights_Assistant.ipynb"]
nb_dir = next(d for d in [HERE.parent / "notebooks", HERE.parent, HERE.parent / "02_Notebooks"] if (d / NOTEBOOKS[0]).exists())
log = {"started": dt.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
       "layout": "submission" if nb_dir.name == "02_Notebooks" else "working", "notebooks": {}, "checks": {}}

if "--checks-only" in sys.argv and (HERE / "run_all_log.json").exists():
    prev = json.loads((HERE / "run_all_log.json").read_text(encoding="utf-8"))
    log["notebooks"] = prev.get("notebooks", {})
    log["notebooks_run_at"] = prev.get("notebooks_run_at", prev.get("started"))
else:
    log["notebooks_run_at"] = log["started"]

if "--checks-only" not in sys.argv:
    import nbformat
    from nbclient import NotebookClient
    for name in NOTEBOOKS:
        path = nb_dir / name
        t = time.time()
        nb = nbformat.read(path, as_version=4)
        try:
            NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(nb_dir)}}).execute()
            status = "PASS"
            err = None
        except Exception as e:  # keep going so the log shows every notebook
            msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).strip()
            status, err = "FAIL", f"{type(e).__name__}: {msg.splitlines()[-1][:300] if msg else ''}"
        nbformat.write(nb, path)
        n_err = sum(o.output_type == "error" for c in nb.cells if c.cell_type == "code" for o in c.outputs)
        log["notebooks"][name] = {"status": status if n_err == 0 else "FAIL", "seconds": round(time.time() - t, 1),
                                  "code_cells": sum(c.cell_type == "code" for c in nb.cells), "error_outputs": n_err,
                                  **({"error": err} if err else {})}
        print(name, log["notebooks"][name], flush=True)

checks = [("independent_check.py", "independent_check_output.json"),
          ("check_segments.py", "check_segments_output.json"),
          ("sensitivity_analysis.py", None),
          ("test_dashboard.py", "test_dashboard_output.json")]
for script, out_name in checks:
    if script == "test_dashboard.py" and importlib.util.find_spec("streamlit") is None:
        log["checks"][script] = {"status": "NOT RUN", "reason": "streamlit is not installed"}
        continue
    t = time.time()
    r = subprocess.run([sys.executable, str(HERE / script)], capture_output=True, text=True, encoding="utf-8")
    if out_name:
        (HERE / out_name).write_text(r.stdout, encoding="utf-8")
    entry = {"status": "PASS" if r.returncode == 0 else "FAIL", "exit_code": r.returncode, "seconds": round(time.time() - t, 1)}
    try:
        data = json.loads(r.stdout)
        if "n_pass" in data:
            entry.update(n_pass=data["n_pass"], n_fail=data["n_fail"])
    except json.JSONDecodeError:
        pass
    if r.returncode != 0:
        entry["stderr_tail"] = r.stderr.strip().splitlines()[-5:]
    log["checks"][script] = entry
    print(script, entry, flush=True)

log["finished"] = dt.datetime.now().isoformat(timespec="seconds")
log["overall"] = "PASS" if all(v["status"] in ("PASS", "NOT RUN") for v in [*log["notebooks"].values(), *log["checks"].values()]) else "FAIL"
(HERE / "run_all_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
print("OVERALL:", log["overall"])
sys.exit(0 if log["overall"] == "PASS" else 1)
