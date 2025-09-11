#!/usr/bin/env python3
"""
TotalSegmentator Docker環境最適化
- 事前ビルド済みイメージ作成
- 永続化コンテナの管理
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Any


class DockerOptimizer:
    """Docker環境最適化クラス"""
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def build_optimized_image(self, image_name: str = "totalseg-optimized:latest") -> bool:
        """最適化済みTotalSegmentatorイメージをビルド"""
        self.logger.info(f"🔨 最適化イメージビルド開始: {image_name}")
        
        # Dockerfile作成
        dockerfile_content = """
FROM wasserth/totalsegmentator:2.10.0

# 事前にモデルをダウンロード（キャッシュ）
RUN totalseg_download_weights -t total

# 必要な追加ライブラリ
RUN pip install --no-cache-dir psutil nibabel

# 作業ディレクトリ設定
WORKDIR /workspace

# GPU最適化設定
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# メモリ使用最適化
ENV CUDA_VISIBLE_DEVICES=0

LABEL description="Pre-optimized TotalSegmentator with cached total models"
"""
        
        dockerfile_path = Path("/tmp/Dockerfile.totalseg-optimized")
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # イメージビルド
        cmd = [
            "docker", "build",
            "-t", image_name,
            "-f", str(dockerfile_path),
            "."
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp")
            if result.returncode == 0:
                self.logger.info(f"✅ イメージビルド完了: {image_name}")
                dockerfile_path.unlink()  # 一時ファイル削除
                return True
            else:
                self.logger.error(f"❌ イメージビルド失敗: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"❌ ビルドエラー: {str(e)}")
            return False
    
    def start_persistent_container(self, container_name: str = "totalseg-daemon", 
                                   image_name: str = "totalseg-optimized:latest") -> bool:
        """永続化コンテナを起動"""
        self.logger.info(f"🚀 永続化コンテナ起動: {container_name}")
        
        # 既存コンテナチェック・停止
        self.stop_persistent_container(container_name)
        
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--gpus", "all",
            "-v", "/home/soya/ctrate_ws:/workspace",
            "--restart", "unless-stopped",
            image_name,
            "sleep", "infinity"  # デーモンモード
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"✅ 永続化コンテナ起動完了: {container_name}")
                return True
            else:
                self.logger.error(f"❌ コンテナ起動失敗: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"❌ 起動エラー: {str(e)}")
            return False
    
    def stop_persistent_container(self, container_name: str = "totalseg-daemon") -> bool:
        """永続化コンテナを停止・削除"""
        try:
            # 停止
            subprocess.run(["docker", "stop", container_name], 
                          capture_output=True, text=True)
            # 削除
            subprocess.run(["docker", "rm", container_name], 
                          capture_output=True, text=True)
            return True
        except:
            return False
    
    def execute_in_persistent_container(self, command: List[str], 
                                       container_name: str = "totalseg-daemon") -> subprocess.CompletedProcess:
        """永続化コンテナ内でコマンド実行"""
        exec_cmd = ["docker", "exec"] + [container_name] + command
        return subprocess.run(exec_cmd, capture_output=True, text=True)
    
    def run_optimized_totalseg(self, ct_path: Path, output_dir: Path, 
                              task: str = "total", fast: bool = False,
                              roi_subset: List[str] = None,
                              container_name: str = "totalseg-daemon") -> Dict[str, Any]:
        """最適化環境でTotalSegmentator実行"""
        self.logger.info(f"⚡ 最適化環境でTotalSegmentator実行: {task}")
        
        # コマンド構築
        cmd = [
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(Path('/home/soya/ctrate_ws'))}",
            "-o", f"/workspace/{output_dir.relative_to(Path('/home/soya/ctrate_ws'))}",
            "--task", task
        ]
        
        if fast and task == "total":  # tissue_typesでは--fast使用不可
            cmd.append("--fast")
        
        if roi_subset and task == "total":
            cmd.extend(["--roi_subset"] + roi_subset)
        
        start_time = time.time()
        result = self.execute_in_persistent_container(cmd, container_name)
        execution_time = time.time() - start_time
        
        return {
            'command': ' '.join(cmd),
            'execution_time': execution_time,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    
    def benchmark_environments(self, ct_path: Path) -> Dict[str, Any]:
        """通常環境 vs 最適化環境のベンチマーク"""
        self.logger.info("📊 環境別ベンチマーク開始")
        
        results = {
            'ct_path': str(ct_path),
            'environments': {}
        }
        
        # 1. 通常Docker環境
        self.logger.info("🐋 通常Docker環境テスト")
        normal_start = time.time()
        normal_cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", "/home/soya/ctrate_ws:/workspace",
            "wasserth/totalsegmentator:2.10.0",
            "TotalSegmentator",
            "-i", f"/workspace/{ct_path.relative_to(Path('/home/soya/ctrate_ws'))}",
            "-o", "/workspace/pipeline/benchmark_test/normal",
            "--task", "total", "--fast"
        ]
        
        normal_result = subprocess.run(normal_cmd, capture_output=True, text=True)
        normal_time = time.time() - normal_start
        
        results['environments']['normal_docker'] = {
            'execution_time': normal_time,
            'success': normal_result.returncode == 0,
            'description': '通常Docker環境（毎回新規起動）'
        }
        
        # 2. 最適化環境（永続化コンテナ）
        if self.start_persistent_container():
            self.logger.info("⚡ 最適化環境テスト")
            optimized_result = self.run_optimized_totalseg(
                ct_path, Path("/home/soya/ctrate_ws/pipeline/benchmark_test/optimized"),
                task="total", fast=True
            )
            
            results['environments']['optimized_persistent'] = {
                'execution_time': optimized_result['execution_time'],
                'success': optimized_result['success'],
                'description': '最適化環境（永続化コンテナ + 事前キャッシュ）'
            }
        
        # 改善率計算
        if 'normal_docker' in results['environments'] and 'optimized_persistent' in results['environments']:
            normal_time = results['environments']['normal_docker']['execution_time']
            optimized_time = results['environments']['optimized_persistent']['execution_time']
            
            if optimized_time > 0:
                improvement_ratio = normal_time / optimized_time
                results['improvement_ratio'] = improvement_ratio
                self.logger.info(f"⚡ 環境最適化による改善: {improvement_ratio:.2f}x 高速化")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="TotalSegmentator Docker環境最適化")
    parser.add_argument("--build-image", action="store_true", help="最適化イメージをビルド")
    parser.add_argument("--start-daemon", action="store_true", help="永続化コンテナを起動")
    parser.add_argument("--stop-daemon", action="store_true", help="永続化コンテナを停止")
    parser.add_argument("--benchmark", help="環境ベンチマーク（CTファイルパス指定）")
    
    args = parser.parse_args()
    
    optimizer = DockerOptimizer()
    
    if args.build_image:
        success = optimizer.build_optimized_image()
        sys.exit(0 if success else 1)
    
    if args.start_daemon:
        success = optimizer.start_persistent_container()
        sys.exit(0 if success else 1)
    
    if args.stop_daemon:
        optimizer.stop_persistent_container()
        sys.exit(0)
    
    if args.benchmark:
        results = optimizer.benchmark_environments(Path(args.benchmark))
        import json
        print(json.dumps(results, indent=2, ensure_ascii=False))
        sys.exit(0)
    
    # デフォルト: ヘルプ表示
    parser.print_help()


if __name__ == "__main__":
    main()