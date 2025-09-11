#!/usr/bin/env python3
"""
TotalSegmentator 高速化テストパイプライン
- total + tissue_types の2ステップ処理
- 品質比較（通常版 vs 高速化版）
- 処理時間・リソース測定
"""

import os
import sys
import time
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# GPUメモリ監視用
try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# リソース監視・品質評価用
import psutil
import numpy as np
import nibabel as nib


class TotalSegSpeedTest:
    """TotalSegmentator高速化テストクラス"""
    
    def __init__(self, output_dir: Path, license_key: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.license_key = license_key
        
        # ログ設定
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # GPU監視
        self.gpu_available = NVML_AVAILABLE
        if self.gpu_available:
            try:
                nvml.nvmlInit()
            except:
                self.gpu_available = False
        
        # テスト結果記録
        self.results = []
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """GPU メモリ使用量を取得"""
        if not self.gpu_available:
            return {}
        
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                'used_mb': round(memory_info.used / (1024**2)),
                'total_mb': round(memory_info.total / (1024**2)),
                'usage_percent': round(memory_info.used / memory_info.total * 100, 1)
            }
        except:
            return {}
    
    def run_totalseg_baseline(self, ct_path: Path, output_dir: Path) -> Dict[str, Any]:
        """通常版TotalSegmentator実行（ベースライン）"""
        self.logger.info("🔄 通常版TotalSegmentator実行開始")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_start = time.time()
        gpu_start = self.get_gpu_memory_usage()
        
        # Step 1: Total task
        step1_start = time.time()
        total_output = output_dir / "total"
        cmd1 = [
            "docker", "run", "--rm", "--gpus", "all",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", "/home/soya/ctrate_ws:/workspace",
            "wasserth/totalsegmentator:2.10.0",
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(Path('/home/soya/ctrate_ws'))}",
            "-o", f"/workspace/{total_output.relative_to(Path('/home/soya/ctrate_ws'))}",
            "--task", "total"
        ]
        
        self.logger.info("  実行中: total task")
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        step1_time = time.time() - step1_start
        
        if result1.returncode != 0:
            raise RuntimeError(f"Total task failed: {result1.stderr}")
        
        # Step 2: Tissue types task
        step2_start = time.time()
        tissue_output = output_dir / "tissue_types"
        cmd2 = [
            "docker", "run", "--rm", "--gpus", "all", 
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", "/home/soya/ctrate_ws:/workspace",
            "wasserth/totalsegmentator:2.10.0",
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(Path('/home/soya/ctrate_ws'))}",
            "-o", f"/workspace/{tissue_output.relative_to(Path('/home/soya/ctrate_ws'))}",
            "--task", "tissue_types"
        ]
        
        if self.license_key:
            cmd2.extend(["-ta", self.license_key])
        
        self.logger.info("  実行中: tissue_types task")
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        step2_time = time.time() - step2_start
        
        if result2.returncode != 0:
            self.logger.warning(f"Tissue types task failed (license required?): {result2.stderr}")
            step2_time = 0
        
        total_time = time.time() - total_start
        gpu_end = self.get_gpu_memory_usage()
        
        return {
            'method': 'baseline',
            'total_time': total_time,
            'step1_time': step1_time, 
            'step2_time': step2_time,
            'gpu_memory_start': gpu_start,
            'gpu_memory_end': gpu_end,
            'output_dir': str(output_dir)
        }
    
    def run_totalseg_fast(self, ct_path: Path, output_dir: Path) -> Dict[str, Any]:
        """高速化版TotalSegmentator実行"""
        self.logger.info("🚀 高速化版TotalSegmentator実行開始")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_start = time.time()
        gpu_start = self.get_gpu_memory_usage()
        
        # Step 1: Total task with --fast and --roi_subset
        step1_start = time.time()
        total_output = output_dir / "total"
        
        cmd1 = [
            "docker", "run", "--rm", "--gpus", "all",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", "/home/soya/ctrate_ws:/workspace", 
            "wasserth/totalsegmentator:2.10.0",
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(Path('/home/soya/ctrate_ws'))}",
            "-o", f"/workspace/{total_output.relative_to(Path('/home/soya/ctrate_ws'))}",
            "--task", "total",
            "--fast"  # 高速モデル使用のみ（roi_subsetは一旦無効）
        ]
        
        self.logger.info("  実行中: total task (--fast)")
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        step1_time = time.time() - step1_start
        
        if result1.returncode != 0:
            raise RuntimeError(f"Fast total task failed: {result1.stderr}")
        
        # Step 2: Tissue types task（--fast使用不可のため通常版）
        step2_start = time.time()
        tissue_output = output_dir / "tissue_types"
        cmd2 = [
            "docker", "run", "--rm", "--gpus", "all",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", "/home/soya/ctrate_ws:/workspace",
            "wasserth/totalsegmentator:2.10.0", 
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(Path('/home/soya/ctrate_ws'))}",
            "-o", f"/workspace/{tissue_output.relative_to(Path('/home/soya/ctrate_ws'))}",
            "--task", "tissue_types"
        ]
        
        if self.license_key:
            cmd2.extend(["-ta", self.license_key])
        
        self.logger.info("  実行中: tissue_types task (通常版)")
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        step2_time = time.time() - step2_start
        
        if result2.returncode != 0:
            self.logger.warning(f"Tissue types task failed: {result2.stderr}")
            step2_time = 0
        
        total_time = time.time() - total_start
        gpu_end = self.get_gpu_memory_usage()
        
        return {
            'method': 'fast_optimized',
            'total_time': total_time,
            'step1_time': step1_time,
            'step2_time': step2_time,
            'gpu_memory_start': gpu_start,
            'gpu_memory_end': gpu_end,
            'output_dir': str(output_dir),
            'optimizations': ['--fast']
        }
    
    def calculate_dice_coefficient(self, mask1_path: Path, mask2_path: Path) -> float:
        """2つのマスクのDice係数を計算"""
        try:
            img1 = nib.load(str(mask1_path))
            img2 = nib.load(str(mask2_path))
            
            data1 = img1.get_fdata() > 0.5
            data2 = img2.get_fdata() > 0.5
            
            intersection = np.sum(data1 & data2)
            total = np.sum(data1) + np.sum(data2)
            
            if total == 0:
                return 1.0  # 両方とも空の場合は完全一致
            
            return 2.0 * intersection / total
            
        except Exception as e:
            self.logger.warning(f"Dice calculation failed for {mask1_path.name}: {str(e)}")
            return -1.0  # エラー値
    
    def compare_mask_quality(self, baseline_dir: Path, fast_dir: Path) -> Dict[str, float]:
        """マスク品質比較"""
        self.logger.info("📊 マスク品質比較開始")
        
        quality_metrics = {}
        
        # 重要なマスクファイルをチェック
        important_masks = [
            "heart_myocardium.nii.gz", "lung_left.nii.gz", "lung_right.nii.gz",
            "liver.nii.gz", "stomach.nii.gz", "spleen.nii.gz"
        ]
        
        baseline_total = baseline_dir / "total"
        fast_total = fast_dir / "total"
        
        for mask_name in important_masks:
            baseline_mask = baseline_total / mask_name
            fast_mask = fast_total / mask_name
            
            if baseline_mask.exists() and fast_mask.exists():
                dice = self.calculate_dice_coefficient(baseline_mask, fast_mask)
                quality_metrics[mask_name] = dice
                self.logger.info(f"  {mask_name}: Dice = {dice:.3f}")
        
        # 脂肪マスク比較（tissue_types）
        baseline_tissue = baseline_dir / "tissue_types" / "torso_fat.nii.gz"
        fast_tissue = fast_dir / "tissue_types" / "torso_fat.nii.gz"
        
        if baseline_tissue.exists() and fast_tissue.exists():
            dice = self.calculate_dice_coefficient(baseline_tissue, fast_tissue)
            quality_metrics["torso_fat.nii.gz"] = dice
            self.logger.info(f"  torso_fat.nii.gz: Dice = {dice:.3f}")
        
        return quality_metrics
    
    def run_single_test(self, ct_path: Path, case_name: str) -> Dict[str, Any]:
        """単一症例のテスト実行"""
        self.logger.info(f"🧪 症例テスト開始: {case_name}")
        
        case_output_dir = self.output_dir / case_name
        
        # 既存の結果をクリア（クリーンスタート）
        if case_output_dir.exists():
            self.logger.info(f"  既存結果をクリア: {case_output_dir}")
            try:
                shutil.rmtree(case_output_dir)
            except PermissionError:
                # Dockerで作成されたroot権限ファイルの削除
                self.logger.info(f"  root権限ファイルをsudoで削除中...")
                subprocess.run(["sudo", "rm", "-rf", str(case_output_dir)], check=True)
        
        case_output_dir.mkdir(parents=True, exist_ok=True)
        
        # ベースライン実行
        baseline_dir = case_output_dir / "baseline"
        try:
            baseline_result = self.run_totalseg_baseline(ct_path, baseline_dir)
        except Exception as e:
            self.logger.error(f"ベースライン実行エラー: {str(e)}")
            baseline_result = {'method': 'baseline', 'error': str(e)}
        
        # 高速化版実行
        fast_dir = case_output_dir / "fast_optimized"
        try:
            fast_result = self.run_totalseg_fast(ct_path, fast_dir)
        except Exception as e:
            self.logger.error(f"高速化版実行エラー: {str(e)}")
            fast_result = {'method': 'fast_optimized', 'error': str(e)}
        
        # 品質比較
        quality_comparison = {}
        if baseline_dir.exists() and fast_dir.exists():
            try:
                quality_comparison = self.compare_mask_quality(baseline_dir, fast_dir)
            except Exception as e:
                self.logger.warning(f"品質比較エラー: {str(e)}")
        
        # 結果まとめ
        test_result = {
            'case_name': case_name,
            'ct_path': str(ct_path),
            'baseline': baseline_result,
            'fast_optimized': fast_result,
            'quality_comparison': quality_comparison,
            'timestamp': time.time()
        }
        
        # 速度改善率計算
        if 'total_time' in baseline_result and 'total_time' in fast_result:
            speedup = baseline_result['total_time'] / fast_result['total_time']
            test_result['speedup_ratio'] = speedup
            self.logger.info(f"⚡ 速度改善: {speedup:.2f}x 高速化")
        
        return test_result
    
    def run_batch_test(self, test_cases: List[tuple]) -> Dict[str, Any]:
        """複数症例のバッチテスト"""
        self.logger.info(f"🔄 バッチテスト開始: {len(test_cases)}症例")
        
        # 既存のバッチ結果ファイルをクリア
        batch_result_file = self.output_dir / "batch_test_results.json"
        if batch_result_file.exists():
            self.logger.info(f"  既存バッチ結果をクリア: {batch_result_file}")
            batch_result_file.unlink()
        
        batch_results = {
            'test_cases': [],
            'summary': {},
            'system_info': {
                'gpu_available': self.gpu_available,
                'gpu_memory_total': self.get_gpu_memory_usage().get('total_mb', 0)
            }
        }
        
        for ct_path, case_name in test_cases:
            test_result = self.run_single_test(Path(ct_path), case_name)
            batch_results['test_cases'].append(test_result)
            
            # 結果をJSONで保存（進捗確認用）
            progress_file = self.output_dir / f"progress_{case_name}.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=2, ensure_ascii=False)
        
        # サマリ統計
        speedups = [r.get('speedup_ratio', 0) for r in batch_results['test_cases'] if 'speedup_ratio' in r]
        quality_scores = []
        
        for result in batch_results['test_cases']:
            for dice in result.get('quality_comparison', {}).values():
                if dice > 0:  # エラー値除外
                    quality_scores.append(dice)
        
        batch_results['summary'] = {
            'total_cases': len(test_cases),
            'successful_cases': len(speedups),
            'average_speedup': np.mean(speedups) if speedups else 0,
            'average_dice_score': np.mean(quality_scores) if quality_scores else 0,
            'min_dice_score': np.min(quality_scores) if quality_scores else 0
        }
        
        # 最終結果保存
        final_result_file = self.output_dir / "batch_test_results.json"
        with open(final_result_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 バッチテスト完了!")
        self.logger.info(f"  平均高速化: {batch_results['summary']['average_speedup']:.2f}x")
        self.logger.info(f"  平均Dice係数: {batch_results['summary']['average_dice_score']:.3f}")
        self.logger.info(f"📄 結果保存: {final_result_file}")
        
        return batch_results


def main():
    parser = argparse.ArgumentParser(
        description="TotalSegmentator 高速化テストパイプライン",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="/home/soya/ctrate_ws/pipeline/totalseg_speed_test_results",
        help="テスト結果出力ディレクトリ"
    )
    
    parser.add_argument(
        "--license-key", "-ta",
        help="TotalSegmentator tissue_types用ライセンスキー"
    )
    
    parser.add_argument(
        "--single-test",
        help="単一症例テスト用CTファイルパス"
    )
    
    args = parser.parse_args()
    
    # テストケース定義
    test_cases = [
        ("/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz", "valid_1_a_1"),
        ("/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_2.nii.gz", "valid_1_a_2"),
        ("/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz", "valid_2_a_1")
    ]
    
    # テスト実行
    tester = TotalSegSpeedTest(
        output_dir=args.output_dir,
        license_key=args.license_key
    )
    
    if args.single_test:
        # 単一テスト
        case_name = Path(args.single_test).stem
        result = tester.run_single_test(Path(args.single_test), case_name)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # バッチテスト
        results = tester.run_batch_test(test_cases)
        print("バッチテスト完了。詳細は結果ファイルを参照してください。")


if __name__ == "__main__":
    main()