#!/usr/bin/env python3
"""
完全最適化統合パイプライン
全最適化手法を統合した1症例完全処理システム

統合される最適化:
- TotalSegmentator ROI Subset最適化（6臓器特化）
- --fastオプション高速化
- 永続化コンテナ利用
- EAT/PAT抽出 → DeepDRR → マスク投影

作成者: Claude + Serena
"""

import os
import sys
import time
import json
import subprocess
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging


class ResourceMonitor:
    """リソース使用量監視"""
    
    def __init__(self):
        self.start_time = time.time()
        self.measurements = []
    
    def get_gpu_usage(self) -> Dict[str, Any]:
        """GPU使用量取得"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                   '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
                return {
                    'gpu_utilization': int(gpu_util),
                    'memory_used_mb': int(mem_used),
                    'memory_total_mb': int(mem_total),
                    'temperature': int(temp)
                }
        except:
            pass
        return {'error': 'GPU情報取得失敗'}
    
    def get_system_usage(self) -> Dict[str, Any]:
        """システムリソース使用量"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
    
    def record_measurement(self, step: str):
        """測定記録"""
        measurement = {
            'step': step,
            'timestamp': time.time(),
            'elapsed_seconds': time.time() - self.start_time,
            'gpu_info': self.get_gpu_usage(),
            'system_info': self.get_system_usage()
        }
        self.measurements.append(measurement)
        return measurement


class FullOptimizedPipeline:
    """完全最適化統合パイプライン"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ログ設定
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # リソース監視
        self.monitor = ResourceMonitor()
        
        # プロジェクトルート
        self.project_root = Path("/home/soya/ctrate_ws")
        
        # EAT/PAT抽出に必要な最小臓器セット（structures.csv準拠）
        self.essential_organs = [
            "heart",                      # 心臓（EAT抽出の基準）
            "lung_upper_lobe_left",       # 左上肺葉
            "lung_lower_lobe_left",       # 左下肺葉  
            "lung_upper_lobe_right",      # 右上肺葉
            "lung_middle_lobe_right",     # 右中肺葉
            "lung_lower_lobe_right",      # 右下肺葉
            "liver",                      # 肝臓（PAT判定）
            "stomach",                    # 胃（必須）
            "spleen",                     # 脾臓（必須）
            "aorta",                      # 大動脈
            "esophagus"                   # 食道
        ]
    
    def _merge_lung_masks(self, totalseg_dir: Path):
        """肺葉マスクを左右肺マスクに統合"""
        try:
            import nibabel as nib
            import numpy as np
            
            self.logger.info("  🫁 肺マスクの統合開始")
            
            # 左肺の統合
            left_lobes = ['lung_upper_lobe_left.nii.gz', 'lung_lower_lobe_left.nii.gz']
            self._merge_masks(totalseg_dir, left_lobes, 'lung_left.nii.gz')
            
            # 右肺の統合  
            right_lobes = ['lung_upper_lobe_right.nii.gz', 'lung_middle_lobe_right.nii.gz', 'lung_lower_lobe_right.nii.gz']
            self._merge_masks(totalseg_dir, right_lobes, 'lung_right.nii.gz')
            
            self.logger.info("  ✅ 肺マスク統合完了")
            
        except Exception as e:
            self.logger.warning(f"  ⚠️ 肺マスク統合失敗: {str(e)}")
    
    def _merge_masks(self, totalseg_dir: Path, mask_files: List[str], output_name: str):
        """複数マスクをOR演算で統合"""
        try:
            import nibabel as nib
            import numpy as np
            
            merged_mask = None
            affine = None
            
            for mask_file in mask_files:
                mask_path = totalseg_dir / mask_file
                if mask_path.exists():
                    img = nib.load(str(mask_path))
                    mask_data = img.get_fdata() > 0
                    
                    if merged_mask is None:
                        merged_mask = mask_data
                        affine = img.affine
                    else:
                        merged_mask = merged_mask | mask_data
            
            if merged_mask is not None and affine is not None:
                # 統合マスクを保存
                merged_img = nib.Nifti1Image(merged_mask.astype(np.uint8), affine)
                output_path = totalseg_dir / output_name
                nib.save(merged_img, str(output_path))
                self.logger.info(f"    {output_name} 生成完了")
                
        except Exception as e:
            self.logger.warning(f"    {output_name} 生成失敗: {str(e)}")

    def check_prerequisites(self) -> bool:
        """前提条件チェック"""
        self.logger.info("🔍 前提条件チェック開始")
        
        # 永続化コンテナ確認
        check_cmd = ["docker", "ps", "-q", "-f", "name=totalseg-daemon"]
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if not result.stdout.strip():
            self.logger.error("❌ 永続化コンテナ 'totalseg-daemon' が見つかりません")
            self.logger.error("先に起動してください: python3 pipeline/optimize_docker_env.py --start-daemon")
            return False
        
        self.logger.info("✅ 永続化コンテナ確認完了")
        return True
    
    def step1_optimized_totalseg(self, ct_path: Path) -> Dict[str, Any]:
        """ステップ1: ROI最適化TotalSegmentator（total + tissue_types統合）"""
        self.logger.info("🎯 ステップ1: ROI最適化TotalSegmentator開始（total + tissue_types）")
        
        step_start = time.time()
        self.monitor.record_measurement("step1_start")
        
        # 出力ディレクトリ
        totalseg_dir = self.output_dir / "totalseg_roi_optimized"
        totalseg_dir.mkdir(parents=True, exist_ok=True)
        
        # ステップ1a: ROI Subset + --fast TotalSegmentator (total)
        cmd_total = [
            "docker", "exec", "totalseg-daemon",
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(self.project_root)}",
            "-o", f"/workspace/{totalseg_dir.relative_to(self.project_root)}",
            "--task", "total",
            "--fast",
            "--roi_subset"
        ] + self.essential_organs
        
        self.logger.info(f"  実行1: TotalSegmentator --fast --roi_subset（{len(self.essential_organs)}臓器）")
        
        # total実行
        result_total = subprocess.run(cmd_total, capture_output=True, text=True)
        
        if result_total.returncode != 0:
            step_time = time.time() - step_start
            self.monitor.record_measurement("step1_end")
            return {
                "step": "totalseg_roi_optimized",
                "execution_time": step_time,
                "success": False,
                "output_dir": str(totalseg_dir),
                "stdout": result_total.stdout,
                "stderr": result_total.stderr,
                "error": "total task failed"
            }
        
        # ステップ1b: tissue_types（torso_fat取得、ライセンスキー使用）
        tissue_dir = totalseg_dir / "tissue"
        tissue_dir.mkdir(parents=True, exist_ok=True)
        
        cmd_tissue = [
            "docker", "exec", "totalseg-daemon",
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(self.project_root)}",
            "-o", f"/workspace/{tissue_dir.relative_to(self.project_root)}",
            "--task", "tissue_types",
            "-l", "aca_DHDKRFJMHUX0PK"  # .envからのライセンスキー
            # NOTE: tissue_typesタスクは--fastと併用不可
        ]
        
        self.logger.info("  実行2: TotalSegmentator tissue_types（torso_fat取得、ライセンス使用）")
        
        # tissue_types実行
        result_tissue = subprocess.run(cmd_tissue, capture_output=True, text=True)
        
        # torso_fatをメインディレクトリにコピー
        if result_tissue.returncode == 0:
            torso_fat_src = tissue_dir / "torso_fat.nii.gz"
            torso_fat_dst = totalseg_dir / "torso_fat.nii.gz"
            
            if torso_fat_src.exists():
                import shutil
                shutil.copy2(torso_fat_src, torso_fat_dst)
                self.logger.info("  ✅ torso_fatマスク統合完了")
        
        # 肺マスクの統合（EAT/PAT抽出スクリプト対応）
        self._merge_lung_masks(totalseg_dir)
        
        step_time = time.time() - step_start
        self.monitor.record_measurement("step1_end")
        
        step_result = {
            "step": "totalseg_roi_optimized",
            "execution_time": step_time,
            "success": result_total.returncode == 0 and result_tissue.returncode == 0,
            "output_dir": str(totalseg_dir),
            "stdout": result_total.stdout + "\n--- tissue_types ---\n" + result_tissue.stdout,
            "stderr": result_total.stderr + "\n--- tissue_types ---\n" + result_tissue.stderr,
            "total_task_success": result_total.returncode == 0,
            "tissue_task_success": result_tissue.returncode == 0
        }
        
        if step_result["success"]:
            # 生成マスク確認
            masks = list(totalseg_dir.glob("*.nii.gz"))
            step_result["generated_masks"] = len(masks)
            step_result["mask_files"] = [m.name for m in masks]
            
            self.logger.info(f"✅ ステップ1完了: {step_time:.1f}秒, {len(masks)}マスク生成")
        else:
            self.logger.error(f"❌ ステップ1失敗: total={result_total.returncode}, tissue={result_tissue.returncode}")
        
        return step_result
    
    def step2_eat_pat_extraction(self, ct_path: Path, totalseg_dir: Path) -> Dict[str, Any]:
        """ステップ2: EAT/PAT抽出"""
        self.logger.info("🫀 ステップ2: EAT/PAT抽出開始")
        
        step_start = time.time()
        self.monitor.record_measurement("step2_start")
        
        # 出力ディレクトリ
        eat_pat_dir = self.output_dir / "eat_pat_extracted"
        
        # EAT/PAT抽出実行
        cmd = [
            "timeout", "600",  # タイムアウト延長（10分）
            "docker", "run", "--rm", "--gpus", "all",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{self.project_root}:/workspace",
            "wasserth/totalsegmentator:2.10.0",
            "python3", "/workspace/scripts/extract_eat_pat_improved_v5_4.py",
            f"/workspace/{ct_path.relative_to(self.project_root)}",
            f"/workspace/{totalseg_dir.relative_to(self.project_root)}",
            f"/workspace/{eat_pat_dir.relative_to(self.project_root)}"
        ]
        
        self.logger.info("  EAT/PAT抽出スクリプト実行")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        step_time = time.time() - step_start
        
        self.monitor.record_measurement("step2_end")
        
        # 成功判定：EAT+PAT抽出ファイル生成確認
        eat_pat_mask = eat_pat_dir / "masks" / "eat_pat.nii.gz"
        analysis_json = eat_pat_dir / "statistics" / "eat_pat_analysis.json"
        
        success_by_output = eat_pat_mask.exists() and analysis_json.exists()
        
        step_result = {
            "step": "eat_pat_extraction", 
            "execution_time": step_time,
            "success": success_by_output,  # ファイル生成ベースでの判定
            "returncode": result.returncode,
            "output_dir": str(eat_pat_dir),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output_files_created": {
                "eat_pat_mask": eat_pat_mask.exists(),
                "analysis_json": analysis_json.exists()
            }
        }
        
        if step_result["success"]:
            self.logger.info(f"✅ ステップ2完了: {step_time:.1f}秒（出力ファイル確認済み）")
        else:
            self.logger.error(f"❌ ステップ2失敗: 必要ファイル未生成（returncode: {result.returncode}）")
        
        return step_result
    
    def step3_deepdrr_generation(self, ct_path: Path) -> Dict[str, Any]:
        """ステップ3: DeepDRR生成"""
        self.logger.info("📸 ステップ3: DeepDRR生成開始")
        
        step_start = time.time()
        self.monitor.record_measurement("step3_start")
        
        # DeepDRR出力ディレクトリ
        deepdrr_dir = self.output_dir / "deepdrr_output"
        deepdrr_dir.mkdir(parents=True, exist_ok=True)
        
        # DeepDRR実行（最終PA修正版）
        cmd = [
            "timeout", "300",
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{self.project_root}:/workspace",
            "deepdrr-japan-mirror:latest", "bash", "-c",
            f"""
            pip3 install torch torchvision pydicom opencv-python scipy --no-cache-dir > /dev/null 2>&1 &&
            echo '🎯 DeepDRR最終PA修正版実行' &&
            cd /workspace &&
            python3 deepdrr/test_deepdrr_final_pa_fixed.py
            """
        ]
        
        self.logger.info("  DeepDRR最終PA修正版実行")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        step_time = time.time() - step_start
        
        self.monitor.record_measurement("step3_end")
        
        step_result = {
            "step": "deepdrr_generation",
            "execution_time": step_time, 
            "success": result.returncode == 0,
            "output_dir": str(deepdrr_dir),
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
        if step_result["success"]:
            self.logger.info(f"✅ ステップ3完了: {step_time:.1f}秒")
        else:
            self.logger.error(f"❌ ステップ3失敗: {result.stderr}")
        
        return step_result
    
    def step4_mask_projection(self, ct_path: Path) -> Dict[str, Any]:
        """ステップ4: マスク投影"""
        self.logger.info("🎭 ステップ4: マスク投影開始")
        
        step_start = time.time()
        self.monitor.record_measurement("step4_start")
        
        # マスク投影実行（厚み考慮最新版）
        cmd = [
            "timeout", "300", 
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{self.project_root}:/workspace",
            "deepdrr-japan-mirror:latest", "bash", "-c",
            f"""
            pip3 install torch torchvision pydicom opencv-python scipy --no-cache-dir > /dev/null 2>&1 &&
            echo '🎯 マスク投影gpt5pro8最終版実行' &&
            cd /workspace &&
            python3 mask_projection/src/mask_deepdrr_projection_thickness_gpt5pro8.py
            """
        ]
        
        self.logger.info("  マスク投影gpt5pro8最終版実行")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        step_time = time.time() - step_start
        
        self.monitor.record_measurement("step4_end")
        
        step_result = {
            "step": "mask_projection",
            "execution_time": step_time,
            "success": result.returncode == 0, 
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
        if step_result["success"]:
            self.logger.info(f"✅ ステップ4完了: {step_time:.1f}秒")
        else:
            self.logger.error(f"❌ ステップ4失敗: {result.stderr}")
        
        return step_result
    
    def run_full_pipeline(self, ct_path: Path) -> Dict[str, Any]:
        """完全パイプライン実行"""
        self.logger.info(f"🚀 完全最適化パイプライン開始: {ct_path.name}")
        
        pipeline_start = time.time()
        self.monitor.record_measurement("pipeline_start")
        
        # 前提条件チェック
        if not self.check_prerequisites():
            return {"success": False, "error": "前提条件チェック失敗"}
        
        # 各ステップ実行
        results = {
            "ct_file": str(ct_path),
            "pipeline_version": "full_optimized_v1.0",
            "optimization_features": [
                "ROI Subset (6臓器特化)",
                "--fast オプション",
                "永続化コンテナ",
                "統合実行"
            ],
            "steps": {}
        }
        
        # ステップ1: ROI最適化TotalSegmentator
        step1_result = self.step1_optimized_totalseg(ct_path)
        results["steps"]["step1"] = step1_result
        
        if not step1_result["success"]:
            results["success"] = False
            results["failed_at"] = "step1"
            return results
        
        totalseg_dir = Path(step1_result["output_dir"])
        
        # ステップ2: EAT/PAT抽出
        step2_result = self.step2_eat_pat_extraction(ct_path, totalseg_dir)
        results["steps"]["step2"] = step2_result
        
        if not step2_result["success"]:
            results["success"] = False
            results["failed_at"] = "step2"
            return results
        
        # ステップ3: DeepDRR生成
        step3_result = self.step3_deepdrr_generation(ct_path)
        results["steps"]["step3"] = step3_result
        
        if not step3_result["success"]:
            results["success"] = False
            results["failed_at"] = "step3"
            return results
        
        # ステップ4: マスク投影
        step4_result = self.step4_mask_projection(ct_path)
        results["steps"]["step4"] = step4_result
        
        if not step4_result["success"]:
            results["success"] = False
            results["failed_at"] = "step4"
            return results
        
        # 完了処理
        pipeline_time = time.time() - pipeline_start
        self.monitor.record_measurement("pipeline_end")
        
        results.update({
            "success": True,
            "total_execution_time": pipeline_time,
            "resource_measurements": self.monitor.measurements,
            "summary": {
                "step1_time": step1_result["execution_time"],
                "step2_time": step2_result["execution_time"],
                "step3_time": step3_result["execution_time"],
                "step4_time": step4_result["execution_time"],
                "total_time": pipeline_time,
                "generated_masks": step1_result.get("generated_masks", 0)
            }
        })
        
        self.logger.info(f"🎉 完全パイプライン成功: {pipeline_time:.1f}秒")
        self.logger.info(f"  - ステップ1(TotalSeg): {step1_result['execution_time']:.1f}秒")
        self.logger.info(f"  - ステップ2(EAT/PAT): {step2_result['execution_time']:.1f}秒")
        self.logger.info(f"  - ステップ3(DeepDRR): {step3_result['execution_time']:.1f}秒") 
        self.logger.info(f"  - ステップ4(マスク投影): {step4_result['execution_time']:.1f}秒")
        
        return results
    
    def save_results(self, results: Dict[str, Any]) -> Path:
        """結果保存"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ct_name = Path(results["ct_file"]).stem
        result_file = self.output_dir / f"full_pipeline_results_{ct_name}_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📄 結果保存: {result_file}")
        return result_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="完全最適化統合パイプライン")
    parser.add_argument("--ct-file", required=True, help="処理対象CTファイル")
    parser.add_argument("--output-dir", "-o", 
                       default="/home/soya/ctrate_ws/pipeline/full_pipeline_test",
                       help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    ct_path = Path(args.ct_file)
    if not ct_path.exists():
        print(f"❌ エラー: CTファイルが見つかりません: {ct_path}")
        sys.exit(1)
    
    # パイプライン実行
    pipeline = FullOptimizedPipeline(args.output_dir)
    results = pipeline.run_full_pipeline(ct_path)
    
    # 結果保存
    result_file = pipeline.save_results(results)
    
    # サマリ表示
    print(f"\n🎊 完全最適化パイプライン結果")
    print(f"成功: {'✅' if results.get('success') else '❌'}")
    
    if results.get("success"):
        summary = results["summary"]
        print(f"総実行時間: {summary['total_time']:.1f}秒")
        print(f"  - TotalSeg(ROI最適化): {summary['step1_time']:.1f}秒")
        print(f"  - EAT/PAT抽出: {summary['step2_time']:.1f}秒")
        print(f"  - DeepDRR生成: {summary['step3_time']:.1f}秒")
        print(f"  - マスク投影: {summary['step4_time']:.1f}秒")
        print(f"生成マスク数: {summary['generated_masks']}")
    else:
        print(f"失敗ステップ: {results.get('failed_at', 'unknown')}")
    
    print(f"詳細結果: {result_file}")


if __name__ == "__main__":
    main()