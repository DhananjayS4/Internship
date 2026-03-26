"""
Build WESAD Feature Dataset
Processes all available WESAD subjects and saves a unified CSV for model training.

Run from project root:
    python ml/build_wesad_dataset.py

Output: data/wesad_features.csv
"""

import os
import sys
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.wesad_loader import get_subject_ids, get_pkl_path, load_subject
from ml.wesad_features import extract_windows

# ─── CONFIG ───────────────────────────────────────────────────────────────────
WESAD_ROOT   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "WESAD", "WESAD")
OUTPUT_CSV   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "wesad_features.csv")
WINDOW_SEC   = 60       # seconds per window
OVERLAP      = 0.5      # 50% window overlap


def build():
    subject_ids = get_subject_ids(WESAD_ROOT)
    if not subject_ids:
        print(f"[ERROR] No subject folders found at: {WESAD_ROOT}")
        print("  Expected structure: data/WESAD/WESAD/S2/S2.pkl ...")
        sys.exit(1)

    print(f"Found {len(subject_ids)} subjects: {subject_ids}")
    all_rows = []

    for sid in subject_ids:
        pkl_path = get_pkl_path(WESAD_ROOT, sid)
        print(f"\n[{sid}] Loading {pkl_path} ...")
        try:
            subject = load_subject(pkl_path)
            rows = extract_windows(subject, window_sec=WINDOW_SEC, overlap=OVERLAP)
            print(f"  -> {len(rows)} windows extracted "
                  f"(relaxed={sum(1 for r in rows if r['label']==0)}, "
                  f"stressed={sum(1 for r in rows if r['label']==1)})")
            for row in rows:
                row["subject"] = sid
            all_rows.extend(rows)
        except Exception as e:
            print(f"  [WARN] Failed to load {sid}: {e}")
            continue

    if not all_rows:
        print("[ERROR] No feature rows extracted. Check WESAD path and pkl files.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    total     = len(df)
    relaxed   = (df["label"] == 0).sum()
    stressed  = (df["label"] == 1).sum()

    print(f"\n✅ Dataset saved -> {OUTPUT_CSV}")
    print(f"   Total windows : {total}")
    print(f"   Relaxed  (0)  : {relaxed}")
    print(f"   Stressed (1)  : {stressed}")
    print(f"\nColumns: {list(df.columns)}")
    print(df.head(3))


if __name__ == "__main__":
    build()
