#!/usr/bin/env python3
"""
tissue_typesモデルを事前にダウンロードするスクリプト
初回実行時のダウンロード時間を削減
"""

import subprocess
import sys

def download_tissue_model():
    """tissue_typesモデルをダウンロード"""
    
    # ダミーファイルを使用してモデルをダウンロード
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--ipc=host",
        "-v", "/tmp:/input",
        "-v", "/tmp:/output",
        "wasserth/totalsegmentator:2.10.0",
        "bash", "-c",
        "python -c \"from totalsegmentator.python_api import download_pretrained_weights; download_pretrained_weights(481)\""
    ]
    
    print("Downloading tissue_types model (Task 481)...")
    print("This is a one-time download of ~233MB")
    print("-" * 60)
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✓ Model downloaded successfully!")
        print("Now tissue_types task will run much faster.")
    else:
        print("\n✗ Failed to download model")
        return False
    
    return True

if __name__ == "__main__":
    success = download_tissue_model()
    sys.exit(0 if success else 1)