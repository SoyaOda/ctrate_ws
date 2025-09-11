#!/usr/bin/env python3
"""
個別スクリプト実行による統合パイプライン
各ステップで元のスクリプトを直接実行し、すべての出力とビジュアライゼーションを生成
"""

import subprocess
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any
import shutil
import os

class IndividualScriptsPipeline:
    """個別スクリプト実行パイプライン"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ロガー設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.project_root = Path("/home/soya/ctrate_ws")
        
    def step1_totalsegmentator(self, ct_path: Path) -> Dict[str, Any]:
        """ステップ1: TotalSegmentatorによるセグメンテーション"""
        self.logger.info("🎯 ステップ1: TotalSegmentator実行開始")
        
        start_time = time.time()
        
        # 出力ディレクトリ
        totalseg_dir = self.output_dir / "totalseg_output"
        totalseg_dir.mkdir(parents=True, exist_ok=True)
        
        # TotalSegmentator実行（永続化コンテナ使用）
        cmd = [
            "docker", "exec", "totalseg-daemon",
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(self.project_root)}",
            "-o", f"/workspace/{totalseg_dir.relative_to(self.project_root)}",
            "--task", "total",
            "--fast",
            "--roi_subset",
            "heart", "lung_upper_lobe_left", "lung_lower_lobe_left",
            "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
            "liver", "stomach", "spleen", "aorta", "esophagus"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # tissue_typesタスク（torso_fat取得）
        tissue_dir = totalseg_dir / "tissue"
        tissue_dir.mkdir(parents=True, exist_ok=True)
        
        cmd_tissue = [
            "docker", "exec", "totalseg-daemon",
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(self.project_root)}",
            "-o", f"/workspace/{tissue_dir.relative_to(self.project_root)}",
            "--task", "tissue_types",
            "-l", "aca_DHDKRFJMHUX0PK"
        ]
        
        result_tissue = subprocess.run(cmd_tissue, capture_output=True, text=True)
        
        # torso_fatコピー
        if result_tissue.returncode == 0:
            torso_fat_src = tissue_dir / "torso_fat.nii.gz"
            if torso_fat_src.exists():
                shutil.copy2(torso_fat_src, totalseg_dir / "torso_fat.nii.gz")
        
        # 肺マスク統合
        self._merge_lung_masks(totalseg_dir)
        
        execution_time = time.time() - start_time
        
        return {
            "step": "totalsegmentator",
            "execution_time": execution_time,
            "success": result.returncode == 0,
            "output_dir": str(totalseg_dir)
        }
    
    def _merge_lung_masks(self, totalseg_dir: Path):
        """肺マスクの統合"""
        try:
            import nibabel as nib
            import numpy as np
            
            # 左肺
            left_lobes = ['lung_upper_lobe_left.nii.gz', 'lung_lower_lobe_left.nii.gz']
            merged_left = None
            affine = None
            
            for lobe in left_lobes:
                lobe_path = totalseg_dir / lobe
                if lobe_path.exists():
                    img = nib.load(str(lobe_path))
                    data = img.get_fdata() > 0
                    if merged_left is None:
                        merged_left = data
                        affine = img.affine
                    else:
                        merged_left = merged_left | data
            
            if merged_left is not None:
                nib.save(nib.Nifti1Image(merged_left.astype(np.uint8), affine),
                        str(totalseg_dir / "lung_left.nii.gz"))
            
            # 右肺
            right_lobes = ['lung_upper_lobe_right.nii.gz', 'lung_middle_lobe_right.nii.gz', 
                          'lung_lower_lobe_right.nii.gz']
            merged_right = None
            
            for lobe in right_lobes:
                lobe_path = totalseg_dir / lobe
                if lobe_path.exists():
                    img = nib.load(str(lobe_path))
                    data = img.get_fdata() > 0
                    if merged_right is None:
                        merged_right = data
                        affine = img.affine
                    else:
                        merged_right = merged_right | data
            
            if merged_right is not None:
                nib.save(nib.Nifti1Image(merged_right.astype(np.uint8), affine),
                        str(totalseg_dir / "lung_right.nii.gz"))
                        
        except Exception as e:
            self.logger.warning(f"肺マスク統合失敗: {e}")
    
    def step2_eat_pat_extraction(self, ct_path: Path, totalseg_dir: Path) -> Dict[str, Any]:
        """ステップ2: EAT/PAT抽出（元のスクリプト実行）"""
        self.logger.info("🫀 ステップ2: EAT/PAT抽出開始")
        
        start_time = time.time()
        
        # 出力ディレクトリ
        eat_pat_dir = self.output_dir / "eat_pat_output"
        
        # 元のスクリプトを実行
        cmd = [
            "timeout", "600",
            "docker", "run", "--rm", "--gpus", "all",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{self.project_root}:/workspace",
            "wasserth/totalsegmentator:2.10.0",
            "python3", "/workspace/scripts/extract_eat_pat_improved_v5_4.py",
            f"/workspace/{ct_path.relative_to(self.project_root)}",
            f"/workspace/{totalseg_dir.relative_to(self.project_root)}",
            f"/workspace/{eat_pat_dir.relative_to(self.project_root)}"
        ]
        
        self.logger.info(f"  実行コマンド: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        execution_time = time.time() - start_time
        
        # 出力ファイル確認
        eat_pat_mask = eat_pat_dir / "masks" / "eat_pat.nii.gz"
        analysis_json = eat_pat_dir / "statistics" / "eat_pat_analysis.json"
        
        return {
            "step": "eat_pat_extraction",
            "execution_time": execution_time,
            "success": eat_pat_mask.exists() and analysis_json.exists(),
            "output_dir": str(eat_pat_dir),
            "files_created": {
                "eat_pat_mask": eat_pat_mask.exists(),
                "analysis_json": analysis_json.exists()
            }
        }
    
    def step3_deepdrr_generation(self, ct_path: Path) -> Dict[str, Any]:
        """ステップ3: DeepDRR生成（元のスクリプト実行）"""
        self.logger.info("📸 ステップ3: DeepDRR生成開始")
        
        start_time = time.time()
        
        # 出力ディレクトリ準備
        deepdrr_dir = self.output_dir / "deepdrr_output"
        deepdrr_dir.mkdir(parents=True, exist_ok=True)
        
        # 元のスクリプトを修正して出力先を変更するための一時スクリプト作成
        temp_script = self.output_dir / "temp_deepdrr.py"
        
        # 元のスクリプトを読み込んで出力先を変更
        original_script = self.project_root / "deepdrr/test_deepdrr_final_pa_fixed.py"
        with open(original_script, 'r') as f:
            script_content = f.read()
        
        # 出力ディレクトリを変更
        script_content = script_content.replace(
            'outdir = Path("/workspace/deepdrr/outputs")',
            f'outdir = Path("/workspace/{deepdrr_dir.relative_to(self.project_root)}")'
        )
        
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        # DeepDRR実行
        cmd = [
            "timeout", "300",
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{self.project_root}:/workspace",
            "deepdrr-japan-mirror:latest", "bash", "-c",
            f"""
            pip3 install torch torchvision pydicom opencv-python scipy --no-cache-dir > /dev/null 2>&1 &&
            echo '🎯 DeepDRR最終PA修正版実行' &&
            cd /workspace &&
            python3 {temp_script.relative_to(self.project_root)}
            """
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        execution_time = time.time() - start_time
        
        # 一時スクリプト削除
        if temp_script.exists():
            temp_script.unlink()
        
        return {
            "step": "deepdrr_generation",
            "execution_time": execution_time,
            "success": result.returncode == 0,
            "output_dir": str(deepdrr_dir),
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def step4_mask_projection(self, ct_path: Path, eat_pat_dir: Path) -> Dict[str, Any]:
        """ステップ4: マスク投影（元のスクリプト実行）"""
        self.logger.info("🎭 ステップ4: マスク投影開始")
        
        start_time = time.time()
        
        # 出力ディレクトリ準備
        projection_dir = self.output_dir / "mask_projection_output"
        projection_dir.mkdir(parents=True, exist_ok=True)
        
        # 一時スクリプト作成（出力先変更）
        temp_script = self.output_dir / "temp_projection.py"
        original_script = self.project_root / "mask_projection/src/mask_deepdrr_projection_thickness_gpt5pro8.py"
        
        with open(original_script, 'r') as f:
            script_content = f.read()
        
        # パスを変更
        script_content = script_content.replace(
            'mask_nifti = Path("/workspace/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz")',
            f'mask_nifti = Path("/workspace/{eat_pat_dir.relative_to(self.project_root)}/masks/eat_pat.nii.gz")'
        )
        script_content = script_content.replace(
            'outdir = Path("/workspace/mask_projection/outputs")',
            f'outdir = Path("/workspace/{projection_dir.relative_to(self.project_root)}")'
        )
        
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        # マスク投影実行
        cmd = [
            "timeout", "300",
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{self.project_root}:/workspace",
            "deepdrr-japan-mirror:latest", "bash", "-c",
            f"""
            pip3 install torch torchvision pydicom opencv-python scipy --no-cache-dir > /dev/null 2>&1 &&
            echo '🎯 マスク投影gpt5pro8最終版実行' &&
            cd /workspace &&
            python3 {temp_script.relative_to(self.project_root)}
            """
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        execution_time = time.time() - start_time
        
        # 一時スクリプト削除
        if temp_script.exists():
            temp_script.unlink()
        
        # 生成ファイル確認
        generated_files = list(projection_dir.glob("*.png")) + list(projection_dir.glob("*.npy"))
        
        return {
            "step": "mask_projection",
            "execution_time": execution_time,
            "success": result.returncode == 0,
            "output_dir": str(projection_dir),
            "generated_files": len(generated_files),
            "files": [f.name for f in generated_files],
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def run_pipeline(self, ct_path: Path) -> Dict[str, Any]:
        """パイプライン実行"""
        self.logger.info(f"🚀 個別スクリプトパイプライン開始: {ct_path.name}")
        
        pipeline_start = time.time()
        results = {
            "ct_file": str(ct_path),
            "pipeline_version": "individual_scripts_v1.0",
            "steps": {}
        }
        
        # ステップ1: TotalSegmentator
        step1_result = self.step1_totalsegmentator(ct_path)
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
        
        eat_pat_dir = Path(step2_result["output_dir"])
        
        # ステップ3: DeepDRR生成
        step3_result = self.step3_deepdrr_generation(ct_path)
        results["steps"]["step3"] = step3_result
        
        if not step3_result["success"]:
            results["success"] = False
            results["failed_at"] = "step3"
            return results
        
        # ステップ4: マスク投影
        step4_result = self.step4_mask_projection(ct_path, eat_pat_dir)
        results["steps"]["step4"] = step4_result
        
        if not step4_result["success"]:
            results["success"] = False
            results["failed_at"] = "step4"
            return results
        
        # 完了
        pipeline_time = time.time() - pipeline_start
        
        results.update({
            "success": True,
            "total_execution_time": pipeline_time,
            "summary": {
                "step1_time": step1_result["execution_time"],
                "step2_time": step2_result["execution_time"],
                "step3_time": step3_result["execution_time"],
                "step4_time": step4_result["execution_time"],
                "total_time": pipeline_time,
                "visualization_files": step4_result.get("files", [])
            }
        })
        
        self.logger.info(f"🎉 パイプライン完了: {pipeline_time:.1f}秒")
        self.logger.info(f"  - ステップ1(TotalSeg): {step1_result['execution_time']:.1f}秒")
        self.logger.info(f"  - ステップ2(EAT/PAT): {step2_result['execution_time']:.1f}秒")
        self.logger.info(f"  - ステップ3(DeepDRR): {step3_result['execution_time']:.1f}秒")
        self.logger.info(f"  - ステップ4(マスク投影): {step4_result['execution_time']:.1f}秒")
        
        return results
    
    def save_results(self, results: Dict[str, Any]) -> Path:
        """結果保存"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = self.output_dir / f"pipeline_results_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📄 結果保存: {result_file}")
        return result_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="個別スクリプト実行パイプライン")
    parser.add_argument("--ct-file", default="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz",
                       help="処理対象CTファイル")
    parser.add_argument("--output-dir", "-o", 
                       default="/home/soya/ctrate_ws/pipeline/individual_scripts_test",
                       help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    ct_path = Path(args.ct_file)
    if not ct_path.exists():
        print(f"❌ エラー: CTファイルが見つかりません: {ct_path}")
        sys.exit(1)
    
    # パイプライン実行
    pipeline = IndividualScriptsPipeline(args.output_dir)
    results = pipeline.run_pipeline(ct_path)
    
    # 結果保存
    result_file = pipeline.save_results(results)
    
    # サマリ表示
    print(f"\n🎊 個別スクリプトパイプライン結果")
    print(f"成功: {'✅' if results.get('success') else '❌'}")
    
    if results.get("success"):
        summary = results["summary"]
        print(f"総実行時間: {summary['total_time']:.1f}秒")
        print(f"  - TotalSeg: {summary['step1_time']:.1f}秒")
        print(f"  - EAT/PAT抽出: {summary['step2_time']:.1f}秒")
        print(f"  - DeepDRR生成: {summary['step3_time']:.1f}秒")
        print(f"  - マスク投影: {summary['step4_time']:.1f}秒")
        print(f"生成ビジュアライゼーション: {len(summary.get('visualization_files', []))}ファイル")
        for f in summary.get('visualization_files', []):
            print(f"  - {f}")
    else:
        print(f"失敗ステップ: {results.get('failed_at', 'unknown')}")
    
    print(f"詳細結果: {result_file}")


if __name__ == "__main__":
    main()