#!/usr/bin/env python3
"""
1症例だけテストするスクリプト
"""

import os
import sys
import subprocess
from pathlib import Path
import json
from datetime import datetime

# テスト用に1症例だけ
TEST_CASE = "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"

# 出力ディレクトリ（E:ドライブのCT-RATE-v2直下）
OUTPUT_BASE_DIR = "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/eat_pat_v5_4_test"
TOTALSEG_BASE_DIR = "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/totalseg_test"

def extract_case_name(ct_path):
    """CTパスから症例名を抽出"""
    parts = Path(ct_path).stem.replace('.nii', '')
    return parts

def run_totalsegmentator_total(ct_path, output_dir):
    """TotalSegmentator - totalタスクを実行"""
    total_dir = output_dir / "total"
    total_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{Path(ct_path).parent}:/input",
        "-v", f"{str(total_dir)}:/output",
        "wasserth/totalsegmentator:2.10.0",
        "TotalSegmentator",
        "-i", f"/input/{Path(ct_path).name}",
        "-o", "/output",
        "--task", "total",
        "--ml"
    ]
    
    print(f"Running TotalSegmentator (total task)...")
    print(f"Command: {' '.join(cmd)}")
    
    # Dockerコマンドを実行
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running TotalSegmentator total task")
        print(f"Return code: {result.returncode}")
        print(f"Stderr: {result.stderr}")
        print(f"Stdout: {result.stdout}")
        return False
    
    print("TotalSegmentator total task completed successfully")
    return True

def main():
    """メイン処理"""
    print(f"Starting test at {datetime.now()}")
    print(f"Test case: {TEST_CASE}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    
    case_name = extract_case_name(TEST_CASE)
    
    # 出力ディレクトリを作成
    totalseg_dir = Path(TOTALSEG_BASE_DIR) / case_name
    output_dir = Path(OUTPUT_BASE_DIR) / case_name
    
    totalseg_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイルが存在するか確認
    if not Path(TEST_CASE).exists():
        print(f"Error: CT file not found: {TEST_CASE}")
        return False
    
    print(f"CT file exists: {TEST_CASE}")
    print(f"File size: {Path(TEST_CASE).stat().st_size / (1024*1024):.2f} MB")
    
    # TotalSegmentator - totalタスクを実行
    print(f"\nRunning TotalSegmentator (total task)...")
    if not run_totalsegmentator_total(TEST_CASE, totalseg_dir):
        print(f"Failed to run TotalSegmentator")
        return False
    
    # 結果ファイルを確認
    total_dir = totalseg_dir / "total"
    if total_dir.exists():
        nifti_files = list(total_dir.glob("*.nii.gz"))
        print(f"\nGenerated {len(nifti_files)} segmentation files:")
        for f in nifti_files[:5]:  # 最初の5個だけ表示
            print(f"  - {f.name}")
    
    print(f"\nTest completed at {datetime.now()}")
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)