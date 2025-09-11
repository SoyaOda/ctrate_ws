#!/usr/bin/env python3
"""
TotalSegmentator バッチ推論テスト
- モデル1回ロード、複数症例連続処理
- Docker起動コスト削減効果の測定
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import logging

class BatchTotalSegTest:
    """バッチ推論テストクラス"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_batch_script(self) -> Path:
        """永続化コンテナ内でのバッチ処理スクリプト作成"""
        script_content = '''#!/bin/bash
# バッチ推論スクリプト
set -e

echo "🚀 バッチ推論開始"

# 引数確認
if [ "$#" -lt 2 ]; then
    echo "使用法: $0 <output_base_dir> <ct_file1> [ct_file2] ..."
    exit 1
fi

OUTPUT_BASE="$1"
shift  # 最初の引数を削除

# 各CTファイルを処理
for CT_FILE in "$@"; do
    # ファイル名から出力ディレクトリ名を生成
    BASENAME=$(basename "$CT_FILE" .nii.gz)
    OUTPUT_DIR="$OUTPUT_BASE/$BASENAME"
    
    echo "🔄 処理中: $CT_FILE → $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    
    # TotalSegmentator実行
    TotalSegmentator -i "$CT_FILE" -o "$OUTPUT_DIR" --task total --fast
    
    echo "✅ 完了: $BASENAME"
done

echo "🎉 バッチ処理完了"
'''
        
        script_path = self.output_dir / "batch_process.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 実行権限付与
        os.chmod(script_path, 0o755)
        return script_path
    
    def run_individual_processing(self, ct_files: List[Path]) -> Dict[str, Any]:
        """個別処理（従来方式）のベンチマーク"""
        self.logger.info("📊 個別処理ベンチマーク開始")
        
        results = {
            'method': 'individual',
            'ct_files': [str(f) for f in ct_files],
            'processing_times': [],
            'total_time': 0
        }
        
        total_start = time.time()
        
        for i, ct_file in enumerate(ct_files):
            self.logger.info(f"  処理 {i+1}/{len(ct_files)}: {ct_file.name}")
            
            output_dir = Path("/home/soya/ctrate_ws") / "pipeline" / "batch_test_results" / "individual" / ct_file.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            start_time = time.time()
            
            cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "-v", "/home/soya/ctrate_ws:/workspace",
                "totalseg-optimized:latest",  # 最適化イメージ使用
                "TotalSegmentator",
                "-i", f"/workspace/{ct_file.relative_to(Path('/home/soya/ctrate_ws'))}",
                "-o", f"/workspace/{output_dir.relative_to(Path('/home/soya/ctrate_ws'))}",
                "--task", "total", "--fast"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            execution_time = time.time() - start_time
            
            results['processing_times'].append({
                'ct_file': str(ct_file),
                'time': execution_time,
                'success': result.returncode == 0
            })
            
            self.logger.info(f"    実行時間: {execution_time:.1f}秒")
        
        results['total_time'] = time.time() - total_start
        return results
    
    def run_batch_processing(self, ct_files: List[Path]) -> Dict[str, Any]:
        """バッチ処理（永続化コンテナ）のベンチマーク"""
        self.logger.info("🚀 バッチ処理ベンチマーク開始")
        
        results = {
            'method': 'batch_persistent',
            'ct_files': [str(f) for f in ct_files],
            'total_time': 0
        }
        
        # バッチスクリプト作成
        script_path = self.create_batch_script()
        
        # コンテナ内にスクリプトをコピー
        copy_cmd = [
            "docker", "cp", str(script_path), "totalseg-daemon:/workspace/batch_process.sh"
        ]
        subprocess.run(copy_cmd, check=True)
        
        # バッチ出力ディレクトリ
        batch_output_dir = Path("/home/soya/ctrate_ws") / "pipeline" / "batch_test_results" / "batch"
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        total_start = time.time()
        
        # 永続化コンテナ内でバッチ処理実行
        ct_paths = [f"/workspace/{ct.relative_to(Path('/home/soya/ctrate_ws'))}" for ct in ct_files]
        batch_cmd = [
            "docker", "exec", "totalseg-daemon",
            "bash", "/workspace/batch_process.sh",
            f"/workspace/{batch_output_dir.relative_to(Path('/home/soya/ctrate_ws'))}"
        ] + ct_paths
        
        self.logger.info("  バッチ処理実行中...")
        result = subprocess.run(batch_cmd, capture_output=True, text=True)
        
        results['total_time'] = time.time() - total_start
        results['success'] = result.returncode == 0
        results['stdout'] = result.stdout
        results['stderr'] = result.stderr
        
        return results
    
    def compare_methods(self, ct_files: List[Path]) -> Dict[str, Any]:
        """個別処理 vs バッチ処理の比較"""
        self.logger.info(f"🔍 処理方式比較開始: {len(ct_files)}症例")
        
        # 個別処理
        individual_results = self.run_individual_processing(ct_files)
        
        # バッチ処理
        batch_results = self.run_batch_processing(ct_files)
        
        # 比較分析
        comparison = {
            'test_cases': len(ct_files),
            'individual_processing': individual_results,
            'batch_processing': batch_results,
            'improvement_analysis': {}
        }
        
        if individual_results['total_time'] > 0 and batch_results['total_time'] > 0:
            speedup = individual_results['total_time'] / batch_results['total_time']
            time_saved = individual_results['total_time'] - batch_results['total_time']
            
            comparison['improvement_analysis'] = {
                'speedup_ratio': speedup,
                'time_saved_seconds': time_saved,
                'time_saved_minutes': time_saved / 60,
                'individual_total_time': individual_results['total_time'],
                'batch_total_time': batch_results['total_time']
            }
            
            self.logger.info(f"📈 結果: {speedup:.2f}x高速化、{time_saved/60:.1f}分短縮")
        
        # 結果保存
        result_file = self.output_dir / "batch_comparison_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        return comparison


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TotalSegmentator バッチ推論テスト")
    parser.add_argument("--output-dir", "-o", 
                       default="/home/soya/ctrate_ws/pipeline/batch_test_results",
                       help="テスト結果出力ディレクトリ")
    parser.add_argument("--ct-files", nargs='+', required=True,
                       help="テスト対象CTファイルパス（複数指定）")
    
    args = parser.parse_args()
    
    ct_files = [Path(f) for f in args.ct_files]
    
    # テスト実行
    tester = BatchTotalSegTest(args.output_dir)
    results = tester.compare_methods(ct_files)
    
    print("\n🎉 バッチ処理テスト完了")
    print(f"結果: {args.output_dir}/batch_comparison_results.json")
    
    if 'improvement_analysis' in results:
        analysis = results['improvement_analysis']
        print(f"高速化: {analysis['speedup_ratio']:.2f}x")
        print(f"時間短縮: {analysis['time_saved_minutes']:.1f}分")


if __name__ == "__main__":
    main()