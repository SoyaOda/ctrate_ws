#!/usr/bin/env python3
"""
統合PAT+EATマスク生成パイプライン
3つの事前セグメンテーションからEAT/PAT抽出まで全てを統合
各ステップの実行時間測定機能付き
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging

# プロジェクトルートパスを追加
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import nibabel as nib
from scipy import ndimage

class IntegratedMaskPipeline:
    """統合マスク生成パイプライン"""
    
    def __init__(self, input_ct_path: str, output_dir: str, verbose: bool = False):
        """
        Args:
            input_ct_path: 入力CTファイルのパス
            output_dir: 出力ディレクトリ
            verbose: 詳細ログを出力するか
        """
        self.input_ct_path = Path(input_ct_path).resolve()  # 絶対パスに変換
        self.output_dir = Path(output_dir).resolve()  # 絶対パスに変換
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ログ設定
        self.setup_logging(verbose)
        
        # 時間測定用（階層的な構造）
        self.timing_results = {
            "steps": {},
            "total": 0
        }
        self.current_step_timings = {}  # 現在のステップのサブステップ時間
        
        # TotalSegmentatorライセンスキー
        self.license_key = "aca_DHDKRFJMHUX0PK"
        
    def setup_logging(self, verbose: bool):
        """ロギングの設定"""
        log_level = logging.DEBUG if verbose else logging.INFO
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # コンソールハンドラ
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # ファイルハンドラ
        log_file = self.output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
    def measure_time(self, func):
        """時間測定デコレータ的な関数"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            func_name = func.__name__
            self.timing_results[func_name] = elapsed_time
            self.logger.info(f"{func_name}: {elapsed_time:.2f}秒")
            return result
        return wrapper
    
    def measure_substep(self, step_name: str, substep_name: str):
        """サブステップの時間測定用コンテキストマネージャ"""
        class SubstepTimer:
            def __init__(self, pipeline, step, substep):
                self.pipeline = pipeline
                self.step = step
                self.substep = substep
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.time() - self.start_time
                # ステップが存在しない場合は初期化
                if self.step not in self.pipeline.timing_results["steps"]:
                    self.pipeline.timing_results["steps"][self.step] = {
                        "total": 0,
                        "substeps": {}
                    }
                # サブステップの時間を記録
                self.pipeline.timing_results["steps"][self.step]["substeps"][self.substep] = elapsed
                self.pipeline.logger.debug(f"  {self.substep}: {elapsed:.2f}秒")
                
        return SubstepTimer(self, step_name, substep_name)
    
    def _build_totalseg_command(self, task: str, output_path: Path) -> list:
        """TotalSegmentatorのDockerコマンドを構築"""
        cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",
            "-v", f"{self.input_ct_path.parent.resolve()}:/input",
            "-v", f"{output_path.resolve()}:/output",
            "-e", f"TOTALSEG_LICENSE_KEY={self.license_key}",
            "wasserth/totalsegmentator:2.10.0",
            "TotalSegmentator",
            "-i", f"/input/{self.input_ct_path.name}",
            "-o", "/output",
            "--task", task
        ]
        
        # heartchambers_highresとtissue_typesタスクにはライセンスキーが必要
        if task in ["heartchambers_highres", "tissue_types"]:
            cmd.extend(["-l", self.license_key])
            
        return cmd
    
    def run_totalsegmentator(self, task: str, output_subdir: str) -> Path:
        """
        TotalSegmentatorを実行
        
        Args:
            task: 実行するタスク名 (heartchambers_highres, total, tissue_types)
            output_subdir: 出力サブディレクトリ名
            
        Returns:
            出力ディレクトリのパス
        """
        output_path = self.output_dir / output_subdir
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"TotalSegmentator実行中: task={task}")
        
        # Dockerコマンドを構築（絶対パスを使用）
        cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",
            "-v", f"{self.input_ct_path.parent.resolve()}:/input",
            "-v", f"{output_path.resolve()}:/output",
            "-e", f"TOTALSEG_LICENSE_KEY={self.license_key}",
            "wasserth/totalsegmentator:2.10.0",
            "TotalSegmentator",
            "-i", f"/input/{self.input_ct_path.name}",
            "-o", "/output",
            "--task", task
        ]
        
        # heartchambers_highresとtissue_typesタスクにはライセンスキーが必要
        if task in ["heartchambers_highres", "tissue_types"]:
            cmd.extend(["-l", self.license_key])
        
        # 実行
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.debug(f"TotalSegmentator stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"TotalSegmentator stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"TotalSegmentator実行エラー: {e}")
            self.logger.error(f"stdout: {e.stdout}")
            self.logger.error(f"stderr: {e.stderr}")
            raise
            
        return output_path
    
    def step1_heartchambers_segmentation(self) -> Path:
        """ステップ1: 心臓セグメンテーション (heartchambers_highres)"""
        self.logger.info("=" * 60)
        self.logger.info("ステップ1: 心臓セグメンテーション開始")
        
        step_start_time = time.time()
        step_name = "step1_heartchambers"
        
        # ディレクトリ作成
        with self.measure_substep(step_name, "prepare_output_dir"):
            output_path = self.output_dir / "segmentation_heart"
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Dockerコマンド構築
        with self.measure_substep(step_name, "build_docker_command"):
            cmd = self._build_totalseg_command("heartchambers_highres", output_path)
        
        # TotalSegmentator実行
        with self.measure_substep(step_name, "run_docker_totalsegmentator"):
            self.logger.info(f"TotalSegmentator実行中: task=heartchambers_highres")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.logger.debug(f"TotalSegmentator stdout: {result.stdout}")
                if result.stderr:
                    self.logger.warning(f"TotalSegmentator stderr: {result.stderr}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"TotalSegmentator実行エラー: {e}")
                self.logger.error(f"stdout: {e.stdout}")
                self.logger.error(f"stderr: {e.stderr}")
                raise
        
        # ステップ全体の時間を記録
        step_elapsed = time.time() - step_start_time
        if step_name not in self.timing_results["steps"]:
            self.timing_results["steps"][step_name] = {"total": 0, "substeps": {}}
        self.timing_results["steps"][step_name]["total"] = step_elapsed
        self.logger.info(f"ステップ1完了: {step_elapsed:.2f}秒")
        
        return output_path
    
    def step2_total_segmentation(self) -> Path:
        """ステップ2: 全臓器セグメンテーション (total)"""
        self.logger.info("=" * 60)
        self.logger.info("ステップ2: 全臓器セグメンテーション開始")
        
        step_start_time = time.time()
        step_name = "step2_total"
        
        # ディレクトリ作成
        with self.measure_substep(step_name, "prepare_output_dir"):
            output_path = self.output_dir / "segmentation_total"
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Dockerコマンド構築
        with self.measure_substep(step_name, "build_docker_command"):
            cmd = self._build_totalseg_command("total", output_path)
        
        # TotalSegmentator実行
        with self.measure_substep(step_name, "run_docker_totalsegmentator"):
            self.logger.info(f"TotalSegmentator実行中: task=total")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.logger.debug(f"TotalSegmentator stdout: {result.stdout}")
                if result.stderr:
                    self.logger.warning(f"TotalSegmentator stderr: {result.stderr}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"TotalSegmentator実行エラー: {e}")
                self.logger.error(f"stdout: {e.stdout}")
                self.logger.error(f"stderr: {e.stderr}")
                raise
        
        # ステップ全体の時間を記録
        step_elapsed = time.time() - step_start_time
        if step_name not in self.timing_results["steps"]:
            self.timing_results["steps"][step_name] = {"total": 0, "substeps": {}}
        self.timing_results["steps"][step_name]["total"] = step_elapsed
        self.logger.info(f"ステップ2完了: {step_elapsed:.2f}秒")
        
        return output_path
    
    def step3_tissue_segmentation(self) -> Path:
        """ステップ3: 組織タイプセグメンテーション (tissue_types)"""
        self.logger.info("=" * 60)
        self.logger.info("ステップ3: 組織タイプセグメンテーション開始")
        
        step_start_time = time.time()
        step_name = "step3_tissue"
        
        # ディレクトリ作成
        with self.measure_substep(step_name, "prepare_output_dir"):
            output_path = self.output_dir / "segmentation_tissue"
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Dockerコマンド構築
        with self.measure_substep(step_name, "build_docker_command"):
            cmd = self._build_totalseg_command("tissue_types", output_path)
        
        # TotalSegmentator実行
        with self.measure_substep(step_name, "run_docker_totalsegmentator"):
            self.logger.info(f"TotalSegmentator実行中: task=tissue_types")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.logger.debug(f"TotalSegmentator stdout: {result.stdout}")
                if result.stderr:
                    self.logger.warning(f"TotalSegmentator stderr: {result.stderr}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"TotalSegmentator実行エラー: {e}")
                self.logger.error(f"stdout: {e.stdout}")
                self.logger.error(f"stderr: {e.stderr}")
                raise
        
        # ステップ全体の時間を記録
        step_elapsed = time.time() - step_start_time
        if step_name not in self.timing_results["steps"]:
            self.timing_results["steps"][step_name] = {"total": 0, "substeps": {}}
        self.timing_results["steps"][step_name]["total"] = step_elapsed
        self.logger.info(f"ステップ3完了: {step_elapsed:.2f}秒")
        
        return output_path
    
    def step4_integrate_segmentations(self, heart_dir: Path, total_dir: Path, tissue_dir: Path) -> Path:
        """
        ステップ4: セグメンテーション結果の統合
        余分なファイル移動なしで必要なマスクを統合
        """
        self.logger.info("=" * 60)
        self.logger.info("ステップ4: セグメンテーション結果の統合開始")
        
        step_start_time = time.time()
        step_name = "step4_integrate"
        
        # 統合先ディレクトリ
        integrated_dir = self.output_dir / "segmentation_integrated"
        integrated_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. heartchambers_highresタスクから心臓コンポーネントをシンボリックリンク
        with self.measure_substep(step_name, "link_heart_components"):
            # heartchambers_highresは"heart_"プレフィックス付きで出力される
            heart_components = ["heart_myocardium", "heart_atrium_left", "heart_atrium_right", 
                               "heart_ventricle_left", "heart_ventricle_right"]
            # 代替名（プレフィックスなし）もチェック
            heart_components_alt = ["myocardium", "atrium_left", "atrium_right",
                                   "ventricle_left", "ventricle_right"]
            
            heart_found = []
            for comp, comp_alt in zip(heart_components, heart_components_alt):
                src = heart_dir / f"{comp}.nii.gz"
                if not src.exists():
                    # プレフィックスなしバージョンを試す
                    src = heart_dir / f"{comp_alt}.nii.gz"
                
                if src.exists():
                    # 常にプレフィックスなしの名前でリンク
                    dst = integrated_dir / f"{comp_alt}.nii.gz"
                    if dst.exists():
                        dst.unlink()
                    dst.symlink_to(src.resolve())
                    self.logger.debug(f"  Linked: {comp_alt}.nii.gz")
                    heart_found.append(comp_alt)
        
        # 心臓コンポーネントを結合してheart.nii.gzを作成
        if heart_found:
            with self.measure_substep(step_name, "combine_heart_components"):
                self.logger.info("  心臓コンポーネントを結合中...")
                heart_mask = None
                affine = None
                header = None
                
                for comp in heart_found:
                    comp_path = integrated_dir / f"{comp}.nii.gz"
                    if comp_path.exists():
                        img = nib.load(str(comp_path))
                        data = img.get_fdata() > 0
                        if heart_mask is None:
                            heart_mask = data
                            affine = img.affine
                            header = img.header
                        else:
                            heart_mask |= data
                        self.logger.debug(f"    Added {comp} to heart mask")
                
                if heart_mask is not None:
                    heart_path = integrated_dir / "heart.nii.gz"
                    heart_img = nib.Nifti1Image(heart_mask.astype(np.uint8), affine, header)
                    nib.save(heart_img, str(heart_path))
                    self.logger.debug("  Created: heart.nii.gz (combined heart components)")
        
        # 2. totalタスクから肺葉と腹部臓器をシンボリックリンク
        with self.measure_substep(step_name, "link_organ_masks"):
            # 肺葉（TotalSegmentatorは葉ごとに出力）
            lung_lobes = [
                "lung_upper_lobe_left", "lung_lower_lobe_left",
                "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"
            ]
            for lobe in lung_lobes:
                src = total_dir / f"{lobe}.nii.gz"
                if src.exists():
                    dst = integrated_dir / f"{lobe}.nii.gz"
                    if dst.exists():
                        dst.unlink()
                    dst.symlink_to(src.resolve())
                    self.logger.debug(f"  Linked: {lobe}.nii.gz")
            
            # その他の臓器
            other_organs = ["liver", "stomach", "spleen",
                           "aorta", "pulmonary_artery", "inferior_vena_cava", "pulmonary_vein"]
            for organ in other_organs:
                src = total_dir / f"{organ}.nii.gz"
                if src.exists():
                    dst = integrated_dir / f"{organ}.nii.gz"
                    if dst.exists():
                        dst.unlink()
                    dst.symlink_to(src.resolve())
                    self.logger.debug(f"  Linked: {organ}.nii.gz")
        
        # 3. tissue_typesタスクからtorso_fatをシンボリックリンク
        with self.measure_substep(step_name, "link_torso_fat"):
            tissue_fat = tissue_dir / "tissue" / "torso_fat.nii.gz"
            if not tissue_fat.exists():
                tissue_fat = tissue_dir / "torso_fat.nii.gz"
            
            if tissue_fat.exists():
                dst = integrated_dir / "torso_fat.nii.gz"
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(tissue_fat.resolve())
                self.logger.debug(f"  Linked: torso_fat.nii.gz")
            else:
                self.logger.error("torso_fat.nii.gz not found!")
                raise FileNotFoundError("torso_fat.nii.gz not found in tissue segmentation output")
        
        # 4. 肺葉マスクの統合（各葉を結合してlung_left.nii.gzとlung_right.nii.gzを作成）
        with self.measure_substep(step_name, "combine_lung_lobes"):
            self.logger.info("  肺葉マスクを統合中...")
            
            # 左肺の葉を結合
            left_lobes = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
            lung_left = None
            affine = None
            header = None
            
            for lobe in left_lobes:
                lobe_path = integrated_dir / f"{lobe}.nii.gz"
                if lobe_path.exists():
                    img = nib.load(str(lobe_path))
                    data = img.get_fdata() > 0
                    if lung_left is None:
                        lung_left = data
                        affine = img.affine
                        header = img.header
                    else:
                        lung_left |= data
                    self.logger.debug(f"    Added {lobe}")
            
            if lung_left is not None:
                lung_left_path = integrated_dir / "lung_left.nii.gz"
                lung_left_img = nib.Nifti1Image(lung_left.astype(np.uint8), affine, header)
                nib.save(lung_left_img, str(lung_left_path))
                self.logger.debug("  Created: lung_left.nii.gz")
            
            # 右肺の葉を結合
            right_lobes = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
            lung_right = None
            
            for lobe in right_lobes:
                lobe_path = integrated_dir / f"{lobe}.nii.gz"
                if lobe_path.exists():
                    img = nib.load(str(lobe_path))
                    data = img.get_fdata() > 0
                    if lung_right is None:
                        lung_right = data
                        affine = img.affine
                        header = img.header
                    else:
                        lung_right |= data
                    self.logger.debug(f"    Added {lobe}")
            
            if lung_right is not None:
                lung_right_path = integrated_dir / "lung_right.nii.gz"
                lung_right_img = nib.Nifti1Image(lung_right.astype(np.uint8), affine, header)
                nib.save(lung_right_img, str(lung_right_path))
                self.logger.debug("  Created: lung_right.nii.gz")
            
            # 左右の肺を統合してlung.nii.gzも作成（オプション）
            if lung_left is not None and lung_right is not None:
                lung_combined = lung_left | lung_right
                lung_combined_img = nib.Nifti1Image(
                    lung_combined.astype(np.uint8), 
                    affine, 
                    header
                )
                lung_combined_path = integrated_dir / "lung.nii.gz"
                nib.save(lung_combined_img, str(lung_combined_path))
                self.logger.debug("  Created: lung.nii.gz (combined)")
        
        # ステップ全体の時間を記録
        step_elapsed = time.time() - step_start_time
        self.timing_results["steps"][step_name]["total"] = step_elapsed
        self.logger.info(f"ステップ4完了: {step_elapsed:.2f}秒")
        
        return integrated_dir
    
    def step5_extract_eat_pat(self, integrated_dir: Path) -> Dict:
        """
        ステップ5: EAT/PAT抽出処理
        extract_eat_pat_improved_v5_4.pyの処理を実行
        """
        self.logger.info("=" * 60)
        self.logger.info("ステップ5: EAT/PAT抽出処理開始")
        
        step_start_time = time.time()
        step_name = "step5_eat_pat"
        
        # モジュールインポート
        with self.measure_substep(step_name, "import_module"):
            from scripts.extract_eat_pat_improved_v5_4 import extract_eat_pat_improved_v5_4
        
        # 出力ディレクトリ準備
        with self.measure_substep(step_name, "prepare_output_dir"):
            eat_pat_output = self.output_dir / "eat_pat_output"
            eat_pat_output.mkdir(parents=True, exist_ok=True)
        
        # パラメータ設定
        with self.measure_substep(step_name, "prepare_parameters"):
            params = {
                "ct_path": str(self.input_ct_path),
                "totalseg_dir": str(integrated_dir),
                "output_dir": str(eat_pat_output),
                "dilation_radius_mm": 15.0,
                "use_proximal_vessels": True,
                "inferior_band_mm": 50.0,
                "lung_contact_mm": 1.0,
                "abdomen_gap_mm": 1.0,
                "max_heart_dist_mm": 20.0,
                "replace_by_contact": False,
                "no_z_limit": False,
                "contact_eps_mm": 1.0,
                "lung_dom_ratio": 2.0,
                "min_lung_touch_voxels": 150,
                "seal_diaphragm_holes_mm": 4.0,
                "post_trim_mm": 1.5,
                "seed_z_margin_mm": 10.0,
                "require_shell_touch": True,
                "z_range_mode": 'rpa_to_diaphragm',
                "verbose": self.logger.level == logging.DEBUG
            }
        
        # EAT/PAT抽出メイン処理
        # 内部の詳細な処理時間はextract_eat_pat_improved_v5_4.py内で測定される
        with self.measure_substep(step_name, "run_eat_pat_extraction"):
            try:
                stats = extract_eat_pat_improved_v5_4(**params)
                self.logger.info(f"  EAT+PAT volume: {stats['volumes_ml']['eat_pat']:.2f} ml")
                self.logger.info(f"  Shell volume: {stats['volumes_ml']['shell']:.2f} ml")
            except Exception as e:
                self.logger.error(f"EAT/PAT抽出エラー: {e}")
                raise
        
        # ステップ全体の時間を記録
        step_elapsed = time.time() - step_start_time
        if step_name not in self.timing_results["steps"]:
            self.timing_results["steps"][step_name] = {"total": 0, "substeps": {}}
        self.timing_results["steps"][step_name]["total"] = step_elapsed
        
        # extract_eat_pat_improved_v5_4.pyから詳細な時間情報を取得
        if "processing_time" in stats:
            # 各処理の詳細時間を追加（意味のある処理のみ）
            proc_time = stats["processing_time"]
            if "load_ct" in proc_time:
                self.timing_results["steps"][step_name]["substeps"]["eat_pat_load_ct"] = proc_time["load_ct"]
            if "build_heart_mask" in proc_time:
                self.timing_results["steps"][step_name]["substeps"]["eat_pat_build_heart"] = proc_time["build_heart_mask"]
            if "compute_shell" in proc_time:
                self.timing_results["steps"][step_name]["substeps"]["eat_pat_compute_shell"] = proc_time["compute_shell"]
            if "ilam_processing" in proc_time:
                self.timing_results["steps"][step_name]["substeps"]["eat_pat_ilam"] = proc_time["ilam_processing"]
        
        self.logger.info(f"ステップ5完了: {step_elapsed:.2f}秒")
        
        return stats
    
    def run(self):
        """パイプライン全体を実行"""
        total_start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("統合マスク生成パイプライン開始")
        self.logger.info(f"入力CT: {self.input_ct_path}")
        self.logger.info(f"出力先: {self.output_dir}")
        self.logger.info("=" * 60)
        
        try:
            # ステップ1: 心臓セグメンテーション
            heart_dir = self.step1_heartchambers_segmentation()
            
            # ステップ2: 全臓器セグメンテーション
            total_dir = self.step2_total_segmentation()
            
            # ステップ3: 組織タイプセグメンテーション
            tissue_dir = self.step3_tissue_segmentation()
            
            # ステップ4: セグメンテーション結果の統合
            integrated_dir = self.step4_integrate_segmentations(heart_dir, total_dir, tissue_dir)
            
            # ステップ5: EAT/PAT抽出処理
            eat_pat_stats = self.step5_extract_eat_pat(integrated_dir)
            
            # 合計時間
            total_elapsed_time = time.time() - total_start_time
            self.timing_results["total"] = total_elapsed_time
            
            # 古い形式の時間記録を削除（互換性のため存在する場合）
            for key in list(self.timing_results.keys()):
                if key not in ["steps", "total"]:
                    del self.timing_results[key]
            
            # 時間測定結果を保存
            self.save_timing_results()
            
            self.logger.info("=" * 60)
            self.logger.info(f"パイプライン完了: 合計 {total_elapsed_time:.2f}秒")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"パイプライン実行エラー: {e}")
            raise
    
    def save_timing_results(self):
        """時間測定結果を保存（階層的な形式）"""
        timing_file = self.output_dir / "timing_results.json"
        with open(timing_file, 'w') as f:
            json.dump(self.timing_results, f, indent=2)
        
        # 詳細サマリーも出力
        summary_file = self.output_dir / "timing_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("実行時間サマリー\n")
            f.write("=" * 60 + "\n\n")
            
            # 各ステップの詳細
            for step_name, step_data in self.timing_results["steps"].items():
                if isinstance(step_data, dict):
                    f.write(f"【{step_name}】 合計: {step_data.get('total', 0):.2f}秒\n")
                    if "substeps" in step_data:
                        for substep_name, substep_time in step_data["substeps"].items():
                            f.write(f"  ├─ {substep_name}: {substep_time:.2f}秒\n")
                    f.write("\n")
            
            # 全体の合計時間
            f.write("=" * 60 + "\n")
            f.write(f"パイプライン全体: {self.timing_results.get('total', 0):.2f}秒\n")
        
        self.logger.info(f"時間測定結果を保存: {timing_file}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="統合PAT+EATマスク生成パイプライン")
    parser.add_argument("input_ct", help="入力CTファイルのパス")
    parser.add_argument("output_dir", help="出力ディレクトリ")
    parser.add_argument("--verbose", action="store_true", help="詳細ログを出力")
    
    args = parser.parse_args()
    
    # パイプライン実行
    pipeline = IntegratedMaskPipeline(
        input_ct_path=args.input_ct,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    pipeline.run()


if __name__ == "__main__":
    main()