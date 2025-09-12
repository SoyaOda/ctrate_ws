#!/usr/bin/env python3
"""
CT-RATE-v2データセット用の設定ファイル生成スクリプト
/mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed
内のすべてのNIfTIファイルを対象とする
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# CT-RATE-v2データセットのパス（WSL環境）
DATASET_PATH = "/mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed"

def find_all_nifti_files(base_dir: Path) -> List[Dict]:
    """
    CT-RATE-v2データセットから全NIfTIファイルを検索
    
    Args:
        base_dir: 検索ベースディレクトリ
    
    Returns:
        CTファイル情報のリスト
    """
    ct_files = []
    
    # すべての.nii.gzファイルを検索
    for ct_path in sorted(base_dir.rglob("*.nii.gz")):
        # 相対パスから名前を生成
        relative_path = ct_path.relative_to(base_dir)
        
        # パス構造から名前を生成
        # 例: valid_1/valid_1_a/valid_1_a_1.nii.gz -> valid_1_valid_1_a_1
        parts = []
        for part in relative_path.parts[:-1]:  # ディレクトリ部分
            parts.append(part)
        
        # ファイル名（拡張子除く）
        file_stem = ct_path.stem
        if file_stem.endswith('.nii'):
            file_stem = file_stem[:-4]
        parts.append(file_stem)
        
        name = "_".join(parts)
        
        ct_files.append({
            "path": str(ct_path),
            "name": name,
            "relative_path": str(relative_path)
        })
    
    return ct_files

def create_config(output_file: str = "ct_rate_v2_config.json", task: str = "total"):
    """
    CT-RATE-v2データセット用の設定ファイルを作成
    
    Args:
        output_file: 出力ファイル名
        task: TotalSegmentatorのタスク
    """
    base_dir = Path(DATASET_PATH)
    
    # ディレクトリの確認
    if not base_dir.exists():
        print(f"エラー: データセットディレクトリが見つかりません: {base_dir}")
        print("Windows H:\\ ドライブが正しくマウントされているか確認してください")
        return False
    
    print(f"データセットディレクトリ: {base_dir}")
    print("NIfTIファイルを検索中...")
    
    # すべてのNIfTIファイルを検索
    ct_files = find_all_nifti_files(base_dir)
    
    if not ct_files:
        print("エラー: NIfTIファイルが見つかりませんでした")
        return False
    
    # 設定データを作成
    config = {
        "description": "CT-RATE-v2 データセット全NIfTIファイル",
        "dataset_path": str(base_dir),
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "total_files": len(ct_files),
        "ct_files": ct_files
    }
    
    # ファイルに保存
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n設定ファイルを作成しました: {output_path}")
    print(f"総ファイル数: {len(ct_files)}")
    
    # 最初の数ファイルを表示
    print("\n最初の10ファイル:")
    for i, ct_file in enumerate(ct_files[:10], 1):
        print(f"  {i:3d}. {ct_file['name']}")
        print(f"       {ct_file['relative_path']}")
    
    if len(ct_files) > 10:
        print(f"\n  ... 他 {len(ct_files) - 10} ファイル")
    
    # パスの分布を表示
    print("\nディレクトリ別ファイル数:")
    dir_counts = {}
    for ct_file in ct_files:
        dir_name = Path(ct_file['relative_path']).parts[0]
        dir_counts[dir_name] = dir_counts.get(dir_name, 0) + 1
    
    for dir_name in sorted(dir_counts.keys()):
        print(f"  {dir_name}: {dir_counts[dir_name]} ファイル")
    
    return True

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CT-RATE-v2データセット用の設定ファイル生成"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="ct_rate_v2_config.json",
        help="出力設定ファイル名 [デフォルト: ct_rate_v2_config.json]"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="total",
        choices=["total", "tissue_types", "heartchambers_highres", "body"],
        help="TotalSegmentatorのタスク [デフォルト: total]"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="データセットの確認のみ（設定ファイルは作成しない）"
    )
    
    args = parser.parse_args()
    
    if args.check:
        # データセットの確認のみ
        base_dir = Path(DATASET_PATH)
        if base_dir.exists():
            ct_files = find_all_nifti_files(base_dir)
            print(f"データセット: {base_dir}")
            print(f"NIfTIファイル数: {len(ct_files)}")
        else:
            print(f"エラー: データセットが見つかりません: {base_dir}")
    else:
        # 設定ファイルを作成
        success = create_config(args.output, args.task)
        if success:
            print("\n完了！")
            print(f"次のコマンドでバッチ処理を開始できます:")
            print(f"  python batch_processor.py -c {args.output} -o <出力ディレクトリ>")

if __name__ == "__main__":
    main()