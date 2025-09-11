#!/usr/bin/env python3
"""
ROI Subset最適化テスト
EAT/PAT抽出に特化した最小臓器セットでのTotalSegmentator高速化
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import logging

class ROIOptimizedTest:
    """ROI最適化テストクラス"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # EAT/PAT抽出に最低限必要な臓器リスト（正確な臓器名使用）
        self.essential_organs = [
            "heart",               # 心臓（EAT抽出の基準）
            "lung_upper_lobe_left", # 左肺上葉
            "lung_upper_lobe_right", # 右肺上葉  
            "liver",               # 肝臓（PAT判定）
            "aorta",               # 大動脈（血管系）
            "esophagus"            # 食道（胸部構造）
        ]
        
        # 比較用：従来の全臓器vs最小セット
        self.optimization_levels = {
            "full": {
                "name": "全臓器セグメンテーション",
                "organs": None,  # 全臓器（デフォルト）
                "description": "TotalSegmentator標準（117臓器）"
            },
            "minimal": {
                "name": "最小臓器セット", 
                "organs": self.essential_organs,
                "description": "EAT/PAT抽出特化（6臓器）"
            }
        }
    
    def run_totalseg_with_roi(self, ct_path: Path, level: str, use_fast: bool = True) -> Dict[str, Any]:
        """指定されたROIレベルでTotalSegmentator実行"""
        config = self.optimization_levels[level]
        
        self.logger.info(f"🔄 実行中: {config['name']}")
        
        output_dir = self.output_dir / f"roi_test_{level}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # コマンド構築
        cmd = [
            "docker", "exec", "totalseg-daemon",
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(Path('/home/soya/ctrate_ws'))}",
            "-o", f"/workspace/{output_dir.relative_to(Path('/home/soya/ctrate_ws'))}",
            "--task", "total"
        ]
        
        # --fast オプション
        if use_fast:
            cmd.append("--fast")
        
        # ROI subset オプション
        if config["organs"]:
            cmd.extend(["--roi_subset"] + config["organs"])
        
        self.logger.info(f"  コマンド: {' '.join(cmd[-10:])}")  # 最後の部分のみ表示
        
        # 実行
        result = subprocess.run(cmd, capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        return {
            "level": level,
            "config": config,
            "execution_time": execution_time,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output_dir": str(output_dir)
        }
    
    def compare_roi_optimization(self, ct_path: Path) -> Dict[str, Any]:
        """ROI最適化効果の比較"""
        self.logger.info(f"🎯 ROI最適化テスト開始: {ct_path.name}")
        
        results = {
            "ct_path": str(ct_path),
            "test_results": {},
            "comparison": {}
        }
        
        # 各最適化レベルを実行
        for level in ["full", "minimal"]:
            try:
                result = self.run_totalseg_with_roi(ct_path, level)
                results["test_results"][level] = result
                
                self.logger.info(f"  {level}: {result['execution_time']:.1f}秒 "
                               f"({'成功' if result['success'] else '失敗'})")
                
            except Exception as e:
                self.logger.error(f"  {level} 実行エラー: {str(e)}")
                results["test_results"][level] = {
                    "level": level,
                    "success": False,
                    "error": str(e)
                }
        
        # 比較分析
        if "full" in results["test_results"] and "minimal" in results["test_results"]:
            full_result = results["test_results"]["full"]
            minimal_result = results["test_results"]["minimal"]
            
            if (full_result.get("success") and minimal_result.get("success") and 
                "execution_time" in full_result and "execution_time" in minimal_result):
                
                full_time = full_result["execution_time"]
                minimal_time = minimal_result["execution_time"]
                
                if minimal_time > 0:
                    speedup = full_time / minimal_time
                    time_saved = full_time - minimal_time
                    
                    results["comparison"] = {
                        "speedup_ratio": speedup,
                        "time_saved_seconds": time_saved,
                        "time_saved_minutes": time_saved / 60,
                        "full_time": full_time,
                        "minimal_time": minimal_time,
                        "organ_reduction": f"117 → {len(self.essential_organs)}臓器"
                    }
                    
                    self.logger.info(f"📈 ROI最適化効果: {speedup:.2f}x高速化、"
                                   f"{time_saved/60:.1f}分短縮")
        
        return results
    
    def analyze_output_quality(self, full_dir: Path, minimal_dir: Path) -> Dict[str, Any]:
        """出力品質分析（生成されたマスクの比較）"""
        self.logger.info("🔍 出力品質分析")
        
        analysis = {
            "masks_comparison": {},
            "coverage_analysis": {}
        }
        
        # 両方のディレクトリに存在するマスクを確認
        if full_dir.exists() and minimal_dir.exists():
            full_masks = set(f.name for f in full_dir.glob("*.nii.gz"))
            minimal_masks = set(f.name for f in minimal_dir.glob("*.nii.gz"))
            
            common_masks = full_masks & minimal_masks
            full_only = full_masks - minimal_masks
            minimal_only = minimal_masks - full_only
            
            analysis["coverage_analysis"] = {
                "full_masks_count": len(full_masks),
                "minimal_masks_count": len(minimal_masks), 
                "common_masks": len(common_masks),
                "full_only_count": len(full_only),
                "minimal_only_count": len(minimal_only),
                "essential_organs_found": [
                    organ for organ in self.essential_organs 
                    if f"{organ}.nii.gz" in minimal_masks
                ]
            }
            
            self.logger.info(f"  全臓器版: {len(full_masks)}マスク生成")
            self.logger.info(f"  最小版: {len(minimal_masks)}マスク生成")
            self.logger.info(f"  重要臓器検出: {len(analysis['coverage_analysis']['essential_organs_found'])}/{len(self.essential_organs)}")
        
        return analysis
    
    def run_comprehensive_test(self, ct_files: List[Path]) -> Dict[str, Any]:
        """包括的ROI最適化テスト"""
        self.logger.info(f"🚀 包括的ROI最適化テスト: {len(ct_files)}症例")
        
        comprehensive_results = {
            "test_cases": [],
            "summary": {}
        }
        
        speedup_ratios = []
        
        for ct_file in ct_files:
            self.logger.info(f"\n📁 テスト症例: {ct_file.name}")
            
            # ROI最適化テスト実行
            result = self.compare_roi_optimization(ct_file)
            
            # 品質分析追加
            if "test_results" in result:
                full_result = result["test_results"].get("full")
                minimal_result = result["test_results"].get("minimal")
                
                if full_result and minimal_result and full_result.get("success") and minimal_result.get("success"):
                    quality_analysis = self.analyze_output_quality(
                        Path(full_result["output_dir"]),
                        Path(minimal_result["output_dir"])
                    )
                    result["quality_analysis"] = quality_analysis
            
            comprehensive_results["test_cases"].append(result)
            
            # 統計用データ収集
            if "comparison" in result and "speedup_ratio" in result["comparison"]:
                speedup_ratios.append(result["comparison"]["speedup_ratio"])
        
        # サマリ統計
        if speedup_ratios:
            import numpy as np
            comprehensive_results["summary"] = {
                "total_cases": len(ct_files),
                "successful_cases": len(speedup_ratios),
                "average_speedup": np.mean(speedup_ratios),
                "median_speedup": np.median(speedup_ratios),
                "min_speedup": np.min(speedup_ratios),
                "max_speedup": np.max(speedup_ratios),
                "std_speedup": np.std(speedup_ratios)
            }
            
            self.logger.info(f"\n📊 ROI最適化サマリ:")
            self.logger.info(f"  平均高速化: {comprehensive_results['summary']['average_speedup']:.2f}x")
            self.logger.info(f"  成功率: {len(speedup_ratios)}/{len(ct_files)} ({len(speedup_ratios)/len(ct_files)*100:.1f}%)")
        
        # 結果保存
        result_file = self.output_dir / "roi_optimization_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        return comprehensive_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ROI Subset最適化テスト")
    parser.add_argument("--output-dir", "-o",
                       default="/home/soya/ctrate_ws/pipeline/roi_test_results",
                       help="テスト結果出力ディレクトリ")
    parser.add_argument("--ct-files", nargs='+', required=True,
                       help="テスト対象CTファイルパス")
    
    args = parser.parse_args()
    
    ct_files = [Path(f) for f in args.ct_files]
    
    # 永続化コンテナの存在確認
    check_cmd = ["docker", "ps", "-q", "-f", "name=totalseg-daemon"]
    result = subprocess.run(check_cmd, capture_output=True, text=True)
    
    if not result.stdout.strip():
        print("❌ エラー: 永続化コンテナ 'totalseg-daemon' が見つかりません")
        print("先に以下のコマンドで起動してください:")
        print("python3 pipeline/optimize_docker_env.py --start-daemon")
        sys.exit(1)
    
    # テスト実行
    tester = ROIOptimizedTest(args.output_dir)
    results = tester.run_comprehensive_test(ct_files)
    
    print(f"\n🎉 ROI最適化テスト完了")
    print(f"結果: {args.output_dir}/roi_optimization_results.json")
    
    if "summary" in results:
        summary = results["summary"]
        print(f"平均高速化: {summary['average_speedup']:.2f}x")
        print(f"成功率: {summary['successful_cases']}/{summary['total_cases']}")


if __name__ == "__main__":
    main()