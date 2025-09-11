#!/usr/bin/env python3
"""
CTスキャン処理パイプライン - 統合版
3つのステップ（PAT/EAT抽出、DRR作成、マスク投影）を統合し、
リソース監視機能付きで実行する。

使用方法:
    python ct_processing_pipeline.py <CT_NIFTI_PATH> <OUTPUT_DIR> [options]
"""

import os
import sys
import time
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# GPUメモリ監視用
try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# リソース監視用
import psutil


class ResourceMonitor:
    """システムリソース監視クラス"""
    
    def __init__(self):
        self.gpu_available = NVML_AVAILABLE
        if self.gpu_available:
            try:
                nvml.nvmlInit()
                self.gpu_count = nvml.nvmlDeviceGetCount()
            except:
                self.gpu_available = False
                self.gpu_count = 0
        else:
            self.gpu_count = 0
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'gpu_count': self.gpu_count,
            'gpu_available': self.gpu_available
        }
        
        if self.gpu_available:
            gpu_info = []
            for i in range(self.gpu_count):
                try:
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info.append({
                        'id': i,
                        'name': name,
                        'memory_total_mb': round(memory_info.total / (1024**2)),
                        'memory_free_mb': round(memory_info.free / (1024**2)),
                        'memory_used_mb': round(memory_info.used / (1024**2))
                    })
                except:
                    pass
            info['gpu_info'] = gpu_info
        
        return info
    
    def get_current_usage(self) -> Dict[str, Any]:
        """現在のリソース使用量を取得"""
        usage = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': round(psutil.virtual_memory().used / (1024**3), 2)
        }
        
        if self.gpu_available:
            gpu_usage = []
            for i in range(self.gpu_count):
                try:
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage.append({
                        'id': i,
                        'memory_used_mb': round(memory_info.used / (1024**2)),
                        'memory_free_mb': round(memory_info.free / (1024**2)),
                        'memory_percent': round(memory_info.used / memory_info.total * 100, 1),
                        'gpu_utilization': utilization.gpu,
                        'memory_utilization': utilization.memory
                    })
                except:
                    pass
            usage['gpu_usage'] = gpu_usage
        
        return usage


class CTProcessingPipeline:
    """CTスキャン処理パイプラインメインクラス"""
    
    def __init__(self, ct_path: Path, output_dir: Path, verbose: bool = False):
        self.ct_path = Path(ct_path)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # リソース監視
        self.monitor = ResourceMonitor()
        self.resource_logs = []
        
        # ログ設定
        log_level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ステップ別出力ディレクトリ
        self.step1_output = self.output_dir / "step1_eat_pat"
        self.step2_output = self.output_dir / "step2_drr"  
        self.step3_output = self.output_dir / "step3_mask_projection"
        self.logs_output = self.output_dir / "logs"
        
        for dir_path in [self.step1_output, self.step2_output, self.step3_output, self.logs_output]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def log_resource_usage(self, step: str, stage: str):
        """リソース使用量をログ"""
        usage = self.monitor.get_current_usage()
        usage['step'] = step
        usage['stage'] = stage
        self.resource_logs.append(usage)
        
        if self.verbose:
            self.logger.info(f"{step} [{stage}]: CPU {usage['cpu_percent']:.1f}%, "
                           f"Memory {usage['memory_percent']:.1f}%")
            if usage.get('gpu_usage'):
                for gpu in usage['gpu_usage']:
                    self.logger.info(f"  GPU {gpu['id']}: Memory {gpu['memory_percent']:.1f}%, "
                                   f"Utilization {gpu['gpu_utilization']}%")
    
    def run_step1_eat_pat_extraction(self) -> bool:
        """Step 1: PAT+EAT抽出"""
        self.logger.info("🔄 Step 1: PAT+EAT抽出開始")
        start_time = time.time()
        self.log_resource_usage("step1", "start")
        
        try:
            # TotalSegmentator実行が必要
            totalseg_dir = self.step1_output / "totalseg"
            totalseg_dir.mkdir(exist_ok=True)
            
            # TotalSegmentator呼び出し - プロジェクトルートを基準に
            project_root = Path("/home/soya/ctrate_ws")
            cmd1 = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{project_root}:/workspace",
                "wasserth/totalsegmentator:2.10.0",
                "TotalSegmentator",
                "-i", f"/workspace/{self.ct_path.relative_to(project_root)}",
                "-o", f"/workspace/{totalseg_dir.relative_to(project_root)}",
                "--task", "total"
            ]
            
            self.logger.info("TotalSegmentator (total task) 実行中...")
            self.log_resource_usage("step1", "totalseg_start")
            
            result = subprocess.run(cmd1, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"TotalSegmentator failed: {result.stderr}")
                return False
                
            self.log_resource_usage("step1", "totalseg_complete")
            
            # EAT/PAT抽出スクリプト実行
            eat_pat_script = Path("/home/soya/ctrate_ws/scripts/extract_eat_pat_improved_v5_4.py")
            cmd2 = [
                "python3", str(eat_pat_script),
                str(self.ct_path),
                str(totalseg_dir),
                str(self.step1_output),
                "--verbose" if self.verbose else ""
            ]
            
            self.logger.info("EAT/PAT抽出スクリプト実行中...")
            self.log_resource_usage("step1", "extraction_start")
            
            result = subprocess.run(cmd2, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"EAT/PAT extraction failed: {result.stderr}")
                return False
            
            self.log_resource_usage("step1", "complete")
            elapsed = time.time() - start_time
            self.logger.info(f"✅ Step 1完了: {elapsed:.1f}秒")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Step 1エラー: {str(e)}")
            return False
    
    def run_step2_drr_generation(self) -> bool:
        """Step 2: DRR作成"""
        self.logger.info("🔄 Step 2: DRR作成開始")
        start_time = time.time()
        self.log_resource_usage("step2", "start")
        
        try:
            # DeepDRRスクリプトを修正してファイルパスを動的に設定
            temp_script = self.step2_output / "temp_deepdrr_script.py"
            
            # 元のスクリプトを読み込み、パスを修正
            deepdrr_script = Path("/home/soya/ctrate_ws/deepdrr/test_deepdrr_final_pa_fixed.py")
            with open(deepdrr_script, 'r') as f:
                script_content = f.read()
            
            # パスを動的に変更
            script_content = script_content.replace(
                'nifti = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")',
                f'nifti = Path("/workspace/{self.ct_path.relative_to(Path("/home/soya/ctrate_ws"))}")'
            )
            script_content = script_content.replace(
                'outdir = Path("/workspace/deepdrr/outputs")',
                f'outdir = Path("/workspace/{self.step2_output.relative_to(Path("/home/soya/ctrate_ws"))}")'
            )
            
            with open(temp_script, 'w') as f:
                f.write(script_content)
            
            # Docker実行
            cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", "/home/soya/ctrate_ws:/workspace",
                "deepdrr-japan-mirror:latest",
                "bash", "-c",
                ("pip3 install torch torchvision pydicom opencv-python --no-cache-dir -i https://pypi.python.org/simple/ > /dev/null 2>&1 && "
                 f"cd /workspace && python3 {temp_script.relative_to(Path('/home/soya/ctrate_ws'))}")
            ]
            
            self.logger.info("DeepDRR実行中...")
            self.log_resource_usage("step2", "deepdrr_start")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"DeepDRR failed: {result.stderr}")
                return False
            
            # 一時ファイル削除
            temp_script.unlink()
            
            self.log_resource_usage("step2", "complete")
            elapsed = time.time() - start_time
            self.logger.info(f"✅ Step 2完了: {elapsed:.1f}秒")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Step 2エラー: {str(e)}")
            return False
    
    def run_step3_mask_projection(self) -> bool:
        """Step 3: マスク投影"""
        self.logger.info("🔄 Step 3: マスク投影開始")
        start_time = time.time()
        self.log_resource_usage("step3", "start")
        
        try:
            # マスク投影スクリプトを修正してパスを動的に設定
            temp_script = self.step3_output / "temp_mask_projection_script.py"
            
            # 元のスクリプトを読み込み、パスを修正
            mask_script = Path("/home/soya/ctrate_ws/mask_projection/src/mask_deepdrr_projection_thickness_gpt5pro8.py")
            with open(mask_script, 'r') as f:
                script_content = f.read()
            
            # パスを動的に変更
            script_content = script_content.replace(
                'ct_nifti   = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")',
                f'ct_nifti   = Path("/workspace/{self.ct_path.relative_to(Path("/home/soya/ctrate_ws"))}")'
            )
            script_content = script_content.replace(
                'mask_nifti = Path("/workspace/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz")',
                f'mask_nifti = Path("/workspace/{self.step1_output.relative_to(Path("/home/soya/ctrate_ws"))}/masks/eat_pat.nii.gz")'
            )
            script_content = script_content.replace(
                'outdir = Path("/workspace/mask_projection/outputs")',
                f'outdir = Path("/workspace/{self.step3_output.relative_to(Path("/home/soya/ctrate_ws"))}")'
            )
            
            with open(temp_script, 'w') as f:
                f.write(script_content)
            
            # Docker実行
            cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", "/home/soya/ctrate_ws:/workspace",
                "deepdrr-japan-mirror:latest",
                "bash", "-c",
                ("pip3 install torch torchvision pydicom opencv-python scipy --no-cache-dir -i https://pypi.python.org/simple/ > /dev/null 2>&1 && "
                 f"cd /workspace && python3 {temp_script.relative_to(Path('/home/soya/ctrate_ws'))}")
            ]
            
            self.logger.info("マスク投影実行中...")
            self.log_resource_usage("step3", "projection_start")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Mask projection failed: {result.stderr}")
                return False
            
            # 一時ファイル削除
            temp_script.unlink()
            
            self.log_resource_usage("step3", "complete")
            elapsed = time.time() - start_time
            self.logger.info(f"✅ Step 3完了: {elapsed:.1f}秒")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Step 3エラー: {str(e)}")
            return False
    
    def save_resource_logs(self):
        """リソースログを保存"""
        log_file = self.logs_output / "resource_usage.json"
        
        # システム情報も追加
        full_log = {
            'system_info': self.monitor.get_system_info(),
            'pipeline_info': {
                'ct_path': str(self.ct_path),
                'output_dir': str(self.output_dir),
                'total_time': sum(log['timestamp'] for log in self.resource_logs if log['stage'] == 'complete') - 
                             sum(log['timestamp'] for log in self.resource_logs if log['stage'] == 'start')
            },
            'resource_logs': self.resource_logs
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(full_log, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 リソースログ保存: {log_file}")
    
    def run_pipeline(self) -> bool:
        """パイプライン全体を実行"""
        self.logger.info("🚀 CTスキャン処理パイプライン開始")
        self.logger.info(f"📁 入力CT: {self.ct_path}")
        self.logger.info(f"📂 出力ディレクトリ: {self.output_dir}")
        
        # システム情報をログ
        sys_info = self.monitor.get_system_info()
        self.logger.info(f"🖥️  システム情報: CPU {sys_info['cpu_count']}コア, "
                        f"Memory {sys_info['memory_total_gb']}GB, "
                        f"GPU {sys_info['gpu_count']}台")
        
        overall_start = time.time()
        
        try:
            # Step 1: PAT+EAT抽出
            if not self.run_step1_eat_pat_extraction():
                return False
            
            # Step 2: DRR作成  
            if not self.run_step2_drr_generation():
                return False
            
            # Step 3: マスク投影
            if not self.run_step3_mask_projection():
                return False
            
            # 完了
            total_time = time.time() - overall_start
            self.logger.info(f"🎉 パイプライン完了! 総時間: {total_time:.1f}秒")
            
            # リソースログ保存
            self.save_resource_logs()
            
            return True
            
        except Exception as e:
            self.logger.error(f"パイプラインエラー: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="CTスキャン処理パイプライン（PAT/EAT抽出 → DRR作成 → マスク投影）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "ct_path",
        help="CTスキャンのNIfTIファイルパス"
    )
    
    parser.add_argument(
        "output_dir", 
        help="出力ディレクトリパス"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細ログ出力"
    )
    
    args = parser.parse_args()
    
    # パイプライン実行
    pipeline = CTProcessingPipeline(
        ct_path=args.ct_path,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    success = pipeline.run_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()