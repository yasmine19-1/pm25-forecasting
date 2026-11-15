"""Reproduce the Phase 1 tables and figures from the three raw CSVs."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    ("-m", "src.config"),
    ("-m", "src.first_look"),
    ("-m", "src.audit"),
    ("-m", "src.clean"),
    ("-m", "src.gap_report"),
    ("-m", "src.station_eligibility"),
    ("-m", "src.thursday_evidence"),
    ("-m", "src.folds", "--verify"),
    ("src/m2_eda.py",),
    ("src/m2_review.py",),
    ("src/m2_s5p_stability.py",),
    ("-m", "src.relationships"),
    ("-m", "src.stratified"),
    ("-m", "src.pipeline_check"),
    ("-m", "src.clipping_check"),
    ("temporal_analysis.py",),
    ("temporal_patterns.py",),
    ("weekly_cycle.py",),
    ("lead_lag_analysis.py",),
    ("drift_seasonality.py",),
    ("-m", "src.m4_figures"),
    ("-m", "src.rolling_check"),
    ("-m", "src.phase1_results"),
    ("-m", "src.folds", "--verify"),
)


def main():
    env = dict(os.environ, MPLCONFIGDIR=str(ROOT / "data" / ".matplotlib"))
    for args in COMMANDS:
        print("+ python", *args, flush=True)
        subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
