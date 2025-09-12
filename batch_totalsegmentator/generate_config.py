#!/usr/bin/env python3
"""
CTファイルリストの設定を自動生成するスクリプト
CT-RATE-v2データセットから全NIfTIファイルを検索して設定ファイルを作成

WSL環境用：Windows H:\ ドライブは /mnt/h/ としてアクセス
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime

def find_ct_files(base_dir: Path, pattern: str = "*.nii.gz") -> List[Dict]:
    """
    指定ディレクトリからCTファイルを検索
    
    Args:
        base_dir: 検索ベースディレクトリ
        pattern: ファイルパターン
    
    Returns:
        CTファイル情報のリスト
    """
    ct_files = []
    
    # 再帰的に検索
    for ct_path in sorted(base_dir.rglob(pattern)):
        # 相対パスから名前を生成
        relative_parts = ct_path.relative_to(base_dir).parts
        
        # 名前を生成（ディレクトリ構造を反映）
        name_parts = []
        for part in relative_parts[:-1]:  # 最後のファイル名以外
            name_parts.append(part)
        
        # ファイル名（拡張子除く）
        file_stem = ct_path.stem
        if file_stem.endswith('.nii'):
            file_stem = file_stem[:-4]
        name_parts.append(file_stem)
        
        name = "_".join(name_parts)
        
        ct_files.append({
            "path": str(ct_path),
            "name": name
        })
    
    return ct_files

def create_config(ct_files: List[Dict], output_file: Path, task: str = "total"):
    """
    設定ファイルを作成
    
    Args:
        ct_files: CTファイル情報のリスト
        output_file: 出力ファイルパス
        task: TotalSegmentatorのタスク
    """
    config = {
        "description": "TotalSegmentator バッチ処理用CTファイルリスト（自動生成）",
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "total_files": len(ct_files),
        "ct_files": ct_files
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"設定ファイルを作成しました: {output_file}")
    print(f"  総ファイル数: {len(ct_files)}")

def main():
    parser = argparse.ArgumentParser(
        description="CTファイルリストの設定を自動生成"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="CTファイルを検索するディレクトリ"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="config_auto.json",
        help="出力設定ファイル名 [デフォルト: config_auto.json]"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="*.nii.gz",
        help="検索パターン [デフォルト: *.nii.gz]"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="total",
        help="TotalSegmentatorのタスク [デフォルト: total]"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="最大ファイル数（テスト用）"
    )
    
    args = parser.parse_args()
    
    # 入力ディレクトリの確認
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {input_dir}")
        return
    
    # CTファイルを検索
    print(f"検索中: {input_dir}")
    ct_files = find_ct_files(input_dir, args.pattern)
    
    if not ct_files:
        print("CTファイルが見つかりませんでした")
        return
    
    # 制限がある場合
    if args.limit:
        ct_files = ct_files[:args.limit]
        print(f"制限: 最初の{args.limit}ファイルのみ使用")
    
    # ファイルリストを表示
    print("\n見つかったファイル:")
    for i, ct_file in enumerate(ct_files[:10], 1):  # 最初の10個のみ表示
        print(f"  {i}. {ct_file['name']}")
    if len(ct_files) > 10:
        print(f"  ... 他 {len(ct_files) - 10} ファイル")
    
    # 設定ファイルを作成
    output_file = Path(args.output)
    create_config(ct_files, output_file, args.task)

if __name__ == "__main__":
    main()