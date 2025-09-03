#!/usr/bin/env python3
"""
valid_fixedディレクトリから最初の10症例を処理するスクリプト
scripts/extract_eat_pat_improved_v5_4.pyを使用してEAT+PATマスクを生成
結果は/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/に保存

モデルの永続化：~/.totalsegmentatorをマウントして再利用
"""

import os
import sys
import subprocess
from pathlib import Path
import json
from datetime import datetime

# テスト用: 最初の2症例のみ処理
# フルバージョン（10症例）にするには、下のCASES_FULLを使用
CASES_FULL = [
    "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz",
    "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz",
    "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_3/valid_3_a/valid_3_a_1.nii.gz",
    "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_4/valid_4_a/valid_4_a_1.nii.gz",
    "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_5/valid_5_a/valid_5_a_1.nii.gz",
    "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_6/valid_6_a/valid_6_a_1.nii.gz",
    "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_7/valid_7_a/valid_7_a_1.nii.gz",
    "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_8/valid_8_a/valid_8_a_1.nii.gz",
    "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_9/valid_9_a/valid_9_a_1.nii.gz",
    "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_10/valid_10_a/valid_10_a_1.nii.gz"
]

# 処理する症例のリスト（テスト用: 最初の2症例のみ）
CASES = CASES_FULL[:2]  # 2症例のみ
# CASES = CASES_FULL    # 全10症例を処理する場合はこちらをコメントアウト解除

# 出力ディレクトリ（E:ドライブのCT-RATE-v2直下）
OUTPUT_BASE_DIR = "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/eat_pat_v5_4_results"
TOTALSEG_BASE_DIR = "/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/totalseg_outputs"

def extract_case_name(ct_path):
    """CTパスから症例名を抽出"""
    # /valid_1/valid_1_a/valid_1_a_1.nii.gz -> valid_1_a_1
    parts = Path(ct_path).stem.replace('.nii', '')
    return parts

def ensure_totalseg_cache_dir():
    """TotalSegmentatorのキャッシュディレクトリを確保"""
    cache_dir = Path.home() / ".totalsegmentator"
    cache_dir.mkdir(exist_ok=True)
    nnunet_dir = cache_dir / "nnunet" / "results"
    nnunet_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def run_totalsegmentator_total(ct_path, output_dir):
    """TotalSegmentator - totalタスクを実行"""
    total_dir = output_dir / "total"
    
    # ディレクトリ作成（エラーハンドリング付き）
    try:
        if not total_dir.exists():
            total_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass  # 既に存在する場合は問題なし
    except Exception as e:
        print(f"Warning: Could not create {total_dir}: {e}")
    
    # 既に処理済みかチェック（主要なマスクファイルの存在確認）
    # 心臓マスクの候補（どれか1つあればOK）
    heart_masks = ["heart_myocardium.nii.gz", "myocardium.nii.gz", "heart.nii.gz"]
    heart_exists = any((total_dir / f).exists() for f in heart_masks)
    
    # その他の必須マスク
    other_required = ["lung_left.nii.gz", "lung_right.nii.gz", 
                     "liver.nii.gz", "stomach.nii.gz", "spleen.nii.gz"]
    other_exists = [f for f in other_required if (total_dir / f).exists()]
    
    # 心臓マスク + その他5個のうち4個以上あれば処理済みとみなす
    if heart_exists and len(other_exists) >= 4:
        print(f"Total task already completed (heart + {len(other_exists)} other masks found)")
        print(f"  Skipping TotalSegmentator total task")
        return True
    
    # キャッシュディレクトリを確保
    cache_dir = ensure_totalseg_cache_dir()
    
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",  # GPUを使用
        "--ipc=host",     # IPCを共有
        "-v", f"{Path(ct_path).parent}:/input",
        "-v", f"{str(total_dir)}:/output",
        "-v", f"{str(cache_dir)}:/root/.totalsegmentator",  # モデルキャッシュをマウント
        "wasserth/totalsegmentator:2.10.0",
        "TotalSegmentator",
        "-i", f"/input/{Path(ct_path).name}",
        "-o", "/output",
        "--task", "total",
        "--ml"
    ]
    
    print(f"Running TotalSegmentator (total task)...")
    print(f"Command: {' '.join(cmd)}")
    print("Note: This may take 1-3 minutes per case with GPU...")
    print("-" * 60)
    
    # ログをリアルタイムで表示
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nError: TotalSegmentator total task failed with return code: {result.returncode}")
        return False
    
    print("TotalSegmentator total task completed successfully")
    return True

def run_totalsegmentator_tissue(ct_path, output_dir):
    """TotalSegmentator - tissue_typesタスクを実行"""
    tissue_dir = output_dir / "tissue_types"
    
    # ディレクトリ作成（エラーハンドリング付き）
    try:
        if not tissue_dir.exists():
            tissue_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass  # 既に存在する場合は問題なし
    except Exception as e:
        print(f"Warning: Could not create {tissue_dir}: {e}")
    
    # 既に処理済みかチェック
    torso_fat_file = tissue_dir / "torso_fat.nii.gz"
    if torso_fat_file.exists():
        print(f"Tissue_types task already completed (found torso_fat.nii.gz)")
        print(f"  Skipping TotalSegmentator tissue_types task")
        return True
    
    # キャッシュディレクトリを確保
    cache_dir = ensure_totalseg_cache_dir()
    
    # ライセンスキーを取得（環境変数または.envファイルから）
    license_key = os.environ.get('TOTALSEG_LICENSE_KEY', 'aca_DHDKRFJMHUX0PK')
    
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",  # GPUを使用
        "--ipc=host",     # IPCを共有
        "-v", f"{Path(ct_path).parent}:/input",
        "-v", f"{str(total_dir)}:/output",
        "-v", f"{str(cache_dir)}:/root/.totalsegmentator",  # モデルキャッシュをマウント
        "wasserth/totalsegmentator:2.10.0",
        "TotalSegmentator",
        "-i", f"/input/{Path(ct_path).name}",
        "-o", "/output",
        "--task", "tissue_types",
        "-l", license_key  # -ta を -l に変更（ライセンスキーの正しいオプション）
    ]
    
    print(f"Running TotalSegmentator (tissue_types task)...")
    print(f"Using license key: {license_key[:10]}...")
    print("-" * 60)
    
    # ログをリアルタイムで表示
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nError: tissue_types task failed with return code: {result.returncode}")
        return False
    
    print("TotalSegmentator tissue_types task completed successfully")
    return True

def run_eat_pat_extraction(ct_path, totalseg_dir, output_dir):
    """EAT+PAT抽出スクリプトを実行"""
    
    # 既に処理済みかチェック
    eat_pat_file = output_dir / "masks" / "eat_pat_mask.nii.gz"
    if eat_pat_file.exists():
        print(f"EAT+PAT extraction already completed (found eat_pat_mask.nii.gz)")
        print(f"  Skipping EAT+PAT extraction")
        return True
    
    # 一時的な統合ディレクトリをシンボリックリンクで作成（軽量）
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "merged"
        temp_path.mkdir()
        
        # totalタスクの結果をシンボリックリンク
        total_dir = totalseg_dir / "total"
        if total_dir.exists():
            for f in total_dir.glob("*.nii.gz"):
                (temp_path / f.name).symlink_to(f)
        
        # tissue_typesタスクの結果をシンボリックリンク
        tissue_dir = totalseg_dir / "tissue_types"
        if tissue_dir.exists():
            for f in tissue_dir.glob("*.nii.gz"):
                if not (temp_path / f.name).exists():
                    (temp_path / f.name).symlink_to(f)
        
        segmentation_dir = temp_path
        
        cmd = [
            "python3",
            "/home/soya/ctrate_ws/scripts/extract_eat_pat_improved_v5_4.py",
            str(ct_path),
            str(segmentation_dir),
            str(output_dir),
            "--z-range-mode", "rpa_to_diaphragm",  # 推奨モード
            "--verbose"
        ]
        
        print(f"Running EAT+PAT extraction...")
        print(f"  Using merged masks from: {segmentation_dir}")
        print(f"  Total masks: {len(list(total_dir.glob('*.nii.gz'))) if total_dir.exists() else 0}")
        print(f"  Tissue masks: {len(list(tissue_dir.glob('*.nii.gz'))) if tissue_dir.exists() else 0}")
        print("-" * 60)
        
        # ログをリアルタイムで表示
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\nError: EAT+PAT extraction failed with return code: {result.returncode}")
            return False
        
        print("\nEAT+PAT extraction completed successfully")
        return True

def process_case(ct_path, case_idx):
    """1症例を処理"""
    case_name = extract_case_name(ct_path)
    print(f"\n{'='*60}")
    print(f"Processing case {case_idx}: {case_name}")
    print(f"CT file: {ct_path}")
    print(f"{'='*60}")
    
    # 出力ディレクトリを作成（エラーハンドリング付き）
    totalseg_dir = Path(TOTALSEG_BASE_DIR) / case_name
    output_dir = Path(OUTPUT_BASE_DIR) / case_name
    
    # TotalSegmentatorディレクトリ
    try:
        if not totalseg_dir.exists():
            totalseg_dir.mkdir(parents=True, exist_ok=True)
        elif totalseg_dir.is_file():
            print(f"Error: {totalseg_dir} exists as a file, not a directory")
            return False
    except FileExistsError:
        # 既に存在する場合は問題なし
        pass
    except Exception as e:
        print(f"Warning: Could not create {totalseg_dir}: {e}")
        # 続行を試みる
    
    # 出力ディレクトリ
    try:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        elif output_dir.is_file():
            print(f"Error: {output_dir} exists as a file, not a directory")
            return False
    except FileExistsError:
        # 既に存在する場合は問題なし
        pass
    except Exception as e:
        print(f"Warning: Could not create {output_dir}: {e}")
        # 続行を試みる
    
    # ファイルが存在するか確認
    if not Path(ct_path).exists():
        print(f"Error: CT file not found: {ct_path}")
        return False
    
    # 1. TotalSegmentator - totalタスクを実行
    print(f"\nStep 1: Running TotalSegmentator (total task)...")
    if not run_totalsegmentator_total(ct_path, totalseg_dir):
        print(f"Failed to run TotalSegmentator total task for {case_name}")
        return False
    
    # 2. TotalSegmentator - tissue_typesタスクを実行（必須）
    print(f"\nStep 2: Running TotalSegmentator (tissue_types task)...")
    if not run_totalsegmentator_tissue(ct_path, totalseg_dir):
        print(f"Failed to run TotalSegmentator tissue_types task for {case_name}")
        return False
    
    # 3. EAT+PAT抽出を実行
    print(f"\nStep 3: Extracting EAT+PAT...")
    if not run_eat_pat_extraction(ct_path, totalseg_dir, output_dir):
        print(f"Failed to extract EAT+PAT for {case_name}")
        return False
    
    # 4. 結果を確認
    mask_file = output_dir / "masks" / "eat_pat_mask.nii.gz"
    if mask_file.exists():
        print(f"\n✓ Successfully created mask: {mask_file}")
        # ファイルサイズを確認
        size_mb = mask_file.stat().st_size / (1024 * 1024)
        print(f"  Mask file size: {size_mb:.2f} MB")
        
        # 統計ファイルも確認
        stats_file = output_dir / "statistics" / "eat_pat_analysis.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                if 'volume_ml' in stats:
                    print(f"  EAT+PAT volume: {stats['volume_ml']:.2f} ml")
    else:
        print(f"\n✗ Mask file not created: {mask_file}")
        return False
    
    return True

def main():
    """メイン処理"""
    print(f"Starting batch processing at {datetime.now()}")
    print(f"Processing {len(CASES)} cases (TEST MODE)" if len(CASES) < 10 else f"Processing {len(CASES)} cases")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"TotalSegmentator directory: {TOTALSEG_BASE_DIR}")
    
    # TotalSegmentatorのキャッシュディレクトリを初期化
    cache_dir = ensure_totalseg_cache_dir()
    print(f"Model cache directory: {cache_dir}")
    print("Note: First run will download models (~233MB), subsequent runs will be faster")
    print()
    
    # 出力ディレクトリを作成（エラーハンドリング付き）
    try:
        output_path = Path(OUTPUT_BASE_DIR)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        elif output_path.is_file():
            print(f"Error: {OUTPUT_BASE_DIR} exists as a file, not a directory")
            return False
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return False
    
    try:
        totalseg_path = Path(TOTALSEG_BASE_DIR)
        if not totalseg_path.exists():
            totalseg_path.mkdir(parents=True, exist_ok=True)
        elif totalseg_path.is_file():
            print(f"Error: {TOTALSEG_BASE_DIR} exists as a file, not a directory")
            return False
    except Exception as e:
        print(f"Error creating TotalSegmentator directory: {e}")
        # ディレクトリ作成に失敗してもWSLの場合は続行を試みる
        print("Attempting to continue despite directory creation error...")
    
    # 処理結果を記録
    results = []
    
    # 各症例を処理
    for idx, ct_path in enumerate(CASES, 1):
        success = process_case(ct_path, idx)
        results.append({
            "case_idx": idx,
            "ct_path": ct_path,
            "case_name": extract_case_name(ct_path),
            "success": success,
            "timestamp": str(datetime.now())
        })
        
        if not success:
            print(f"\nWarning: Failed to process case {idx}")
    
    # 結果をまとめて表示
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"Total cases: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed cases:")
        for r in results:
            if not r["success"]:
                print(f"  - Case {r['case_idx']}: {r['case_name']}")
    
    # 結果をJSONファイルに保存
    results_file = Path(OUTPUT_BASE_DIR) / "processing_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    print(f"\nBatch processing completed at {datetime.now()}")
    
    return successful == len(results)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)