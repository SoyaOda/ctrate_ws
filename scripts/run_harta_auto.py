#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import glob
import shlex
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Run HARTA automatic segmentation on DICOM cases")
    ap.add_argument("--harta_repo", required=True, help="HARTA repo path (e.g., ~/ctrate_ws/tools/HARTA)")
    ap.add_argument("--dicom_case", required=True, help="DICOM case directory path")
    ap.add_argument("--out_dir", required=True, help="Output directory for HARTA results")
    ap.add_argument("--python", default=sys.executable, help="Python executable in harta env")
    args = ap.parse_args()

    # Expand paths
    harta_repo = os.path.expanduser(args.harta_repo)
    dicom_case = os.path.expanduser(args.dicom_case)
    out_dir = os.path.expanduser(args.out_dir)
    
    # Check if HARTA repo exists
    auto_py = os.path.join(harta_repo, "_automatic_.py")
    if not os.path.exists(auto_py):
        sys.exit(f"ERROR: _automatic_.py not found at {auto_py}")
    
    # Check if DICOM directory exists
    if not os.path.exists(dicom_case):
        sys.exit(f"ERROR: DICOM directory not found: {dicom_case}")
    
    # Create output directory and subdirectories
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "slices"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "contours"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "fat"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "combined"), exist_ok=True)
    
    # Prepare Python script to run HARTA
    run_script = f"""
import sys
import os
sys.path.insert(0, '{harta_repo}')
os.chdir('{harta_repo}')

from _automatic_ import segmentEpicardialFat

# Run HARTA automatic segmentation
try:
    patient_id, no_slices, vol = segmentEpicardialFat(
        DICOM_DATASET='{dicom_case}',
        OUTPUT_FOLDER='{out_dir}/'
    )
    print(f"SUCCESS: Patient={{patient_id}}, Slices={{no_slices}}, Volume={{vol:.2f}}ml")
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    sys.exit(1)
"""
    
    # Run the script
    cmd = [args.python, "-c", run_script]
    print(f"[RUN] HARTA automatic segmentation for: {dicom_case}")
    print(f"[INFO] Output directory: {out_dir}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(result.stdout)
        if result.stderr:
            print(f"[STDERR] {result.stderr}", file=sys.stderr)
        
        if result.returncode != 0:
            print(f"[ERROR] HARTA returned code {result.returncode}")
            sys.exit(1)
        else:
            print("[OK] HARTA segmentation completed")
            
    except Exception as e:
        print(f"[ERROR] Failed to run HARTA: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()