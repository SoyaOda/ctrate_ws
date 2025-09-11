#!/usr/bin/env python3
"""
8月31日のEAT/PAT抽出を完全再現する統合パイプライン
heartchambers_highresタスク + 8月31日パラメータを使用
"""

import subprocess
import sys
import time
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

class HeartchambersAug31Pipeline:
    """8月31日再現パイプライン"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ロギング設定
        log_file = self.output_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.project_root = Path("/home/soya/ctrate_ws")
        
        # 8月31日のパラメータ（固定値）
        self.aug31_params = {
            "lung_contact_mm": 0.5,
            "contact_eps_mm": 3.0,
            "abdomen_gap_mm": 0.0,
            "max_heart_dist_mm": 35.0,
            "lung_dom_ratio": 1.5,
            "seal_diaphragm_holes_mm": 3.0,
            "dilation_radius": 15.0
        }
        
        # ライセンスキー（CLAUDE.mdから）
        self.license_key = "aca_DHDKRFJMHUX0PK"
        
    def run_command(self, cmd: list, description: str, timeout: int = 600) -> Dict[str, Any]:
        """コマンド実行とロギング"""
        self.logger.info(f"🚀 {description}")
        self.logger.debug(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info(f"✅ {description} 完了 ({execution_time:.1f}秒)")
            else:
                self.logger.error(f"❌ {description} 失敗")
                self.logger.error(f"stderr: {result.stderr}")
            
            return {
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            self.logger.error(f"⏱️ {description} タイムアウト")
            return {
                "success": False,
                "execution_time": timeout,
                "error": "Timeout"
            }
    
    def step1_heartchambers_highres(self, ct_path: Path) -> Dict[str, Any]:
        """ステップ1: heartchambers_highresタスク実行"""
        self.logger.info("=" * 60)
        self.logger.info("ステップ1: heartchambers_highresタスク")
        self.logger.info("=" * 60)
        
        totalseg_dir = self.output_dir / "totalseg_heartchambers"
        
        # 既存ディレクトリがある場合は確認のみ
        if totalseg_dir.exists():
            self.logger.info("既存のTotalSegmentator出力を確認中...")
            
            # 必要なファイルの存在確認
            heart_components = [
                "heart_myocardium", "heart_atrium_left", "heart_atrium_right",
                "heart_ventricle_left", "heart_ventricle_right"
            ]
            
            all_exist = True
            for component in heart_components:
                if not (totalseg_dir / f"{component}.nii.gz").exists():
                    all_exist = False
                    break
            
            if all_exist:
                self.logger.info("✅ 既存のheartchambers_highres出力を使用")
                return {
                    "success": True,
                    "execution_time": 0,
                    "output_dir": str(totalseg_dir),
                    "skipped": True,
                    "message": "既存の出力を使用"
                }
            else:
                self.logger.info("必要なファイルが不足しているため、再実行します")
        
        # heartchambers_highresタスク実行
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{self.project_root}:/workspace",
            "wasserth/totalsegmentator:2.10.0",
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(self.project_root)}",
            "-o", f"/workspace/{totalseg_dir.relative_to(self.project_root)}",
            "--task", "heartchambers_highres",
            "-l", self.license_key
        ]
        
        result = self.run_command(cmd, "heartchambers_highresタスク", timeout=600)
        
        if result["success"]:
            # 生成ファイル確認
            heart_components = [
                "heart_myocardium", "heart_atrium_left", "heart_atrium_right",
                "heart_ventricle_left", "heart_ventricle_right"
            ]
            
            missing = []
            for component in heart_components:
                if not (totalseg_dir / f"{component}.nii.gz").exists():
                    missing.append(component)
            
            if missing:
                self.logger.error(f"必須コンポーネントが不足: {missing}")
                result["success"] = False
            else:
                self.logger.info(f"✅ 全5つの心臓コンポーネント生成確認")
        
        result["output_dir"] = str(totalseg_dir)
        return result
    
    def step2_total_task(self, ct_path: Path) -> Dict[str, Any]:
        """ステップ2: totalタスク（torso_fat等の取得）"""
        self.logger.info("=" * 60)
        self.logger.info("ステップ2: totalタスク（臓器マスク）")
        self.logger.info("=" * 60)
        
        totalseg_dir = self.output_dir / "totalseg_heartchambers"
        
        # 既にtotalタスクの出力があるか確認
        required_organs = ["liver", "stomach", "spleen", "aorta", "inferior_vena_cava"]
        all_exist = True
        for organ in required_organs:
            if not (totalseg_dir / f"{organ}.nii.gz").exists():
                all_exist = False
                break
        
        if all_exist:
            self.logger.info("✅ 既存のtotalタスク出力を使用")
            return {
                "success": True,
                "execution_time": 0,
                "skipped": True,
                "message": "既存の出力を使用"
            }
        
        # totalタスク実行
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{self.project_root}:/workspace",
            "wasserth/totalsegmentator:2.10.0",
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(self.project_root)}",
            "-o", f"/workspace/{totalseg_dir.relative_to(self.project_root)}",
            "--task", "total"
        ]
        
        result = self.run_command(cmd, "totalタスク", timeout=600)
        
        return result
    
    def step3_tissue_types(self, ct_path: Path) -> Dict[str, Any]:
        """ステップ3: tissue_typesタスク（torso_fat）"""
        self.logger.info("=" * 60)
        self.logger.info("ステップ3: tissue_typesタスク（torso_fat）")
        self.logger.info("=" * 60)
        
        totalseg_dir = self.output_dir / "totalseg_heartchambers"
        tissue_dir = totalseg_dir / "tissue"
        
        # 既にtorso_fatがあるか確認
        if (totalseg_dir / "torso_fat.nii.gz").exists():
            self.logger.info("✅ 既存のtorso_fatを使用")
            return {
                "success": True,
                "execution_time": 0,
                "skipped": True,
                "message": "既存のtorso_fatを使用"
            }
        
        # tissue_typesタスク実行
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{self.project_root}:/workspace",
            "wasserth/totalsegmentator:2.10.0",
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(self.project_root)}",
            "-o", f"/workspace/{tissue_dir.relative_to(self.project_root)}",
            "--task", "tissue_types",
            "-l", self.license_key
        ]
        
        result = self.run_command(cmd, "tissue_typesタスク", timeout=300)
        
        if result["success"]:
            # 権限修正してからコピー
            self.logger.info("権限修正中...")
            perm_cmd = ["bash", "-c", 
                       f"echo 'odaSOYA6480' | sudo -S chown -R $(id -u):$(id -g) {tissue_dir}"]
            self.run_command(perm_cmd, "tissue_dir権限修正", timeout=30)
            
            # torso_fatをメインディレクトリにコピー
            torso_fat_src = tissue_dir / "torso_fat.nii.gz"
            torso_fat_dst = totalseg_dir / "torso_fat.nii.gz"
            
            if torso_fat_src.exists():
                try:
                    shutil.copy2(torso_fat_src, torso_fat_dst)
                    self.logger.info("✅ torso_fatをコピー完了")
                except PermissionError:
                    # メインディレクトリの権限も修正
                    self.logger.info("メインディレクトリの権限修正中...")
                    perm_cmd2 = ["bash", "-c", 
                               f"echo 'odaSOYA6480' | sudo -S chown -R $(id -u):$(id -g) {totalseg_dir}"]
                    self.run_command(perm_cmd2, "totalseg_dir権限修正", timeout=30)
                    
                    # 再度コピー試行
                    shutil.copy2(torso_fat_src, torso_fat_dst)
                    self.logger.info("✅ torso_fatをコピー完了（権限修正後）")
            else:
                self.logger.error("❌ torso_fat.nii.gzが見つかりません")
                result["success"] = False
        
        return result
    
    def step4_fix_permissions(self) -> Dict[str, Any]:
        """ステップ4: 権限修正"""
        self.logger.info("=" * 60)
        self.logger.info("ステップ4: 権限修正")
        self.logger.info("=" * 60)
        
        totalseg_dir = self.output_dir / "totalseg_heartchambers"
        
        # sudoパスワードを使用して権限修正
        cmd = ["bash", "-c", 
               f"echo 'odaSOYA6480' | sudo -S chown -R $(id -u):$(id -g) {totalseg_dir}"]
        
        result = self.run_command(cmd, "権限修正", timeout=30)
        return result
    
    def step5_merge_lungs(self) -> Dict[str, Any]:
        """ステップ5: 肺マスク統合"""
        self.logger.info("=" * 60)
        self.logger.info("ステップ5: 肺マスク統合")
        self.logger.info("=" * 60)
        
        try:
            import nibabel as nib
            import numpy as np
            
            totalseg_dir = self.output_dir / "totalseg_heartchambers"
            
            # 既に統合済みの肺マスクがあるか確認
            if (totalseg_dir / "lung_left.nii.gz").exists() and (totalseg_dir / "lung_right.nii.gz").exists():
                self.logger.info("✅ 既存の統合肺マスクを使用")
                return {"success": True, "execution_time": 0, "skipped": True}
            
            # 左肺統合
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
                self.logger.info("✅ lung_left.nii.gz生成")
            
            # 右肺統合
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
                self.logger.info("✅ lung_right.nii.gz生成")
            
            return {"success": True, "execution_time": 0}
            
        except Exception as e:
            self.logger.error(f"肺マスク統合エラー: {e}")
            return {"success": False, "error": str(e)}
    
    def step6_eat_pat_extraction(self, ct_path: Path) -> Dict[str, Any]:
        """ステップ6: EAT/PAT抽出（8月31日パラメータ）"""
        self.logger.info("=" * 60)
        self.logger.info("ステップ6: EAT/PAT抽出（8月31日パラメータ）")
        self.logger.info("=" * 60)
        
        totalseg_dir = self.output_dir / "totalseg_heartchambers"
        eat_pat_dir = self.output_dir / "eat_pat_output"
        
        # パラメータ表示
        self.logger.info("8月31日パラメータ:")
        for key, value in self.aug31_params.items():
            self.logger.info(f"  {key}: {value}")
        
        # heartchambers_fixed版を使用
        script_path = "/workspace/scripts/extract_eat_pat_heartchambers_fixed.py"
        
        # スクリプトが存在しない場合は通常版を使用
        if not (self.project_root / "scripts/extract_eat_pat_heartchambers_fixed.py").exists():
            self.logger.warning("heartchambers_fixed版が見つかりません。通常版を使用します。")
            script_path = "/workspace/scripts/extract_eat_pat_improved_v5_4.py"
        
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{self.project_root}:/workspace",
            "wasserth/totalsegmentator:2.10.0",
            "python3", script_path,
            f"/workspace/{ct_path.relative_to(self.project_root)}",
            f"/workspace/{totalseg_dir.relative_to(self.project_root)}",
            f"/workspace/{eat_pat_dir.relative_to(self.project_root)}",
            "--lung-contact-mm", str(self.aug31_params["lung_contact_mm"]),
            "--contact-eps-mm", str(self.aug31_params["contact_eps_mm"]),
            "--abdomen-gap-mm", str(self.aug31_params["abdomen_gap_mm"]),
            "--max-heart-dist-mm", str(self.aug31_params["max_heart_dist_mm"]),
            "--lung-dom-ratio", str(self.aug31_params["lung_dom_ratio"]),
            "--seal-diaphragm-holes-mm", str(self.aug31_params["seal_diaphragm_holes_mm"]),
            "--dilation-radius", str(self.aug31_params["dilation_radius"])
        ]
        
        result = self.run_command(cmd, "EAT/PAT抽出", timeout=900)
        
        if result["success"]:
            # 結果ファイル確認
            eat_pat_mask = eat_pat_dir / "masks" / "eat_pat.nii.gz"
            analysis_json = eat_pat_dir / "statistics" / "eat_pat_analysis.json"
            
            if eat_pat_mask.exists() and analysis_json.exists():
                # 統計読み込み
                with open(analysis_json, 'r') as f:
                    stats = json.load(f)
                
                self.logger.info("📊 EAT/PAT抽出結果:")
                self.logger.info(f"  心臓体積: {stats['volumes_ml']['heart']:.2f} ml")
                self.logger.info(f"  EAT+PAT体積: {stats['volumes_ml']['eat_pat']:.2f} ml")
                self.logger.info(f"  Fat fraction: {stats['ratios_percent']['fat_fraction_in_shell']:.2f}%")
                
                result["statistics"] = stats
            else:
                self.logger.error("出力ファイルが見つかりません")
                result["success"] = False
        
        result["output_dir"] = str(eat_pat_dir)
        return result
    
    def compare_with_aug31(self) -> None:
        """8月31日の結果と比較"""
        self.logger.info("=" * 60)
        self.logger.info("8月31日との比較")
        self.logger.info("=" * 60)
        
        aug31_json = Path("/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/statistics/eat_pat_analysis.json")
        current_json = self.output_dir / "eat_pat_output/statistics/eat_pat_analysis.json"
        
        if aug31_json.exists() and current_json.exists():
            with open(aug31_json, 'r') as f:
                aug31_stats = json.load(f)
            with open(current_json, 'r') as f:
                current_stats = json.load(f)
            
            self.logger.info("📊 比較結果:")
            self.logger.info("=" * 50)
            self.logger.info("                    8月31日      今回         差")
            self.logger.info("-" * 50)
            
            # 心臓体積
            heart_aug31 = aug31_stats['volumes_ml']['heart']
            heart_current = current_stats['volumes_ml']['heart']
            self.logger.info(f"心臓体積:         {heart_aug31:>8.2f} ml  {heart_current:>8.2f} ml  {heart_current-heart_aug31:+7.2f} ml")
            
            # EAT+PAT体積
            eat_aug31 = aug31_stats['volumes_ml']['eat_pat']
            eat_current = current_stats['volumes_ml']['eat_pat']
            self.logger.info(f"EAT+PAT体積:      {eat_aug31:>8.2f} ml  {eat_current:>8.2f} ml  {eat_current-eat_aug31:+7.2f} ml")
            
            # Fat fraction
            ff_aug31 = aug31_stats['ratios_percent']['fat_fraction_in_shell']
            ff_current = current_stats['ratios_percent']['fat_fraction_in_shell']
            self.logger.info(f"Fat fraction:     {ff_aug31:>8.2f} %   {ff_current:>8.2f} %   {ff_current-ff_aug31:+7.2f} %")
            
            # MD5ハッシュ比較
            import hashlib
            
            aug31_mask = Path("/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz")
            current_mask = self.output_dir / "eat_pat_output/masks/eat_pat.nii.gz"
            
            if aug31_mask.exists() and current_mask.exists():
                with open(aug31_mask, 'rb') as f:
                    aug31_md5 = hashlib.md5(f.read()).hexdigest()
                with open(current_mask, 'rb') as f:
                    current_md5 = hashlib.md5(f.read()).hexdigest()
                
                self.logger.info("-" * 50)
                self.logger.info("MD5ハッシュ:")
                self.logger.info(f"  8月31日: {aug31_md5}")
                self.logger.info(f"  今回:    {current_md5}")
                self.logger.info(f"  一致:    {'✅ Yes' if aug31_md5 == current_md5 else '❌ No'}")
        else:
            self.logger.warning("比較用ファイルが見つかりません")
    
    def run_full_pipeline(self, ct_path: Path) -> Dict[str, Any]:
        """完全パイプライン実行"""
        self.logger.info("🚀 8月31日再現パイプライン開始")
        self.logger.info(f"CT: {ct_path}")
        self.logger.info(f"出力: {self.output_dir}")
        
        pipeline_start = time.time()
        results = {
            "ct_file": str(ct_path),
            "output_dir": str(self.output_dir),
            "aug31_params": self.aug31_params,
            "steps": {}
        }
        
        # 各ステップ実行
        steps = [
            ("step1_heartchambers", self.step1_heartchambers_highres, ct_path),
            ("step2_total", self.step2_total_task, ct_path),
            ("step3_tissue", self.step3_tissue_types, ct_path),
            ("step4_permissions", self.step4_fix_permissions, ),
            ("step5_lungs", self.step5_merge_lungs, ),
            ("step6_eat_pat", self.step6_eat_pat_extraction, ct_path)
        ]
        
        for step_name, step_func, *args in steps:
            result = step_func(*args)
            results["steps"][step_name] = result
            
            if not result["success"]:
                self.logger.error(f"❌ {step_name}で失敗")
                results["success"] = False
                results["failed_at"] = step_name
                return results
        
        # 完了
        pipeline_time = time.time() - pipeline_start
        results["success"] = True
        results["total_execution_time"] = pipeline_time
        
        # 比較実行
        self.compare_with_aug31()
        
        # 結果保存
        result_file = self.output_dir / f"pipeline_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info("=" * 60)
        self.logger.info(f"🎉 パイプライン完了")
        self.logger.info(f"総実行時間: {pipeline_time:.1f}秒")
        self.logger.info(f"結果: {result_file}")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="8月31日EAT/PAT抽出再現パイプライン",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--ct-file",
        default="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz",
        help="処理対象CTファイル"
    )
    parser.add_argument(
        "--output-dir",
        default="/home/soya/ctrate_ws/pipeline/aug31_reproduce",
        help="出力ディレクトリ"
    )
    
    args = parser.parse_args()
    
    ct_path = Path(args.ct_file)
    if not ct_path.exists():
        print(f"❌ CTファイルが見つかりません: {ct_path}")
        sys.exit(1)
    
    # パイプライン実行
    pipeline = HeartchambersAug31Pipeline(args.output_dir)
    results = pipeline.run_full_pipeline(ct_path)
    
    # 終了コード
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()