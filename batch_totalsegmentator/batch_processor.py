#!/usr/bin/env python3
"""
TotalSegmentator バッチ処理システム
複数のCTデータに対してTotalSegmentatorを効率的に実行

特徴：
- モデルは一度だけダウンロード（初回のみ）
- 複数のCTファイルを連続処理
- 進捗管理と再開機能
- エラーハンドリングとログ記録
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

class BatchTotalSegmentator:
    """TotalSegmentatorのバッチ処理クラス"""
    
    def __init__(self, config_file: str, output_base_dir: str, 
                 task: str = "total", license_key: str = "aca_DHDKRFJMHUX0PK"):
        """
        Args:
            config_file: 処理するCTファイルのリストを含むJSONファイル
            output_base_dir: 出力ベースディレクトリ
            task: TotalSegmentatorのタスク (total, tissue_types, heartchambers_highres等)
            license_key: TotalSegmentatorのライセンスキー
        """
        self.config_file = Path(config_file)
        self.output_base_dir = Path(output_base_dir)
        self.task = task
        self.license_key = license_key
        
        # 出力ディレクトリを作成
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # ログディレクトリ
        self.log_dir = self.output_base_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # 進捗ファイル
        self.progress_file = self.output_base_dir / "progress.json"
        
        # ログ設定
        self.setup_logging()
        
        # 設定を読み込み
        self.load_config()
        
        # 進捗を読み込み（存在する場合）
        self.load_progress()
        
    def setup_logging(self):
        """ロギングの設定"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # ファイルハンドラ
        log_file = self.log_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # コンソールハンドラ
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def load_config(self):
        """設定ファイルを読み込み"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        
        self.ct_files = self.config.get('ct_files', [])
        self.logger.info(f"設定を読み込みました: {len(self.ct_files)}個のCTファイル")
        
    def load_progress(self):
        """進捗ファイルを読み込み"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
            self.logger.info(f"進捗を読み込みました: {len(self.progress.get('completed', []))}個完了済み")
        else:
            self.progress = {
                'completed': [],
                'failed': [],
                'skipped': [],
                'last_updated': None
            }
    
    def save_progress(self):
        """進捗を保存"""
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)
    
    def get_output_dir_path(self, ct_file: Dict) -> Path:
        """CTファイルパスから出力ディレクトリパスを生成
        
        元のディレクトリ構造を維持して出力ディレクトリを作成
        例: valid_1/valid_1_a/valid_1_a_1.nii.gz 
        -> TotalSegmentator_valid_fixed/valid_1/valid_1_a/valid_1_a_1/
        """
        # 相対パスを取得（設定ファイルに保存されている）
        relative_path = ct_file.get('relative_path')
        if not relative_path:
            # relative_pathがない場合は元のパスから生成
            ct_path = Path(ct_file['path'])
            # valid_fixedからの相対パスを計算
            try:
                relative_path = ct_path.relative_to(Path("/mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed"))
            except:
                # フォールバック
                relative_path = Path(ct_file.get('name', ct_path.stem))
        else:
            relative_path = Path(relative_path)
        
        # ファイル名（拡張子除く）をディレクトリ名として使用
        file_stem = relative_path.stem
        if file_stem.endswith('.nii'):
            file_stem = file_stem[:-4]
        
        # 出力パス: ディレクトリ構造 + ファイル名ディレクトリ
        output_subpath = relative_path.parent / file_stem
        
        return self.output_base_dir / output_subpath
    
    def run_totalsegmentator(self, ct_file: Dict) -> bool:
        """
        単一のCTファイルに対してTotalSegmentatorを実行
        
        Args:
            ct_file: CTファイル情報の辞書 {"path": "...", "name": "..."}
        
        Returns:
            成功した場合True
        """
        ct_path = Path(ct_file['path'])
        
        # 出力ディレクトリを準備（元のディレクトリ構造を維持）
        output_dir = self.get_output_dir_path(ct_file)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"処理開始: {ct_path.name} -> {output_dir}")
        
        # Dockerコマンドを構築
        cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",
            "-v", f"{ct_path.parent.resolve()}:/input",
            "-v", f"{output_dir.resolve()}:/output",
            "-e", f"TOTALSEG_LICENSE_KEY={self.license_key}",
            # メモリ制限を設定（必要に応じて調整）
            "--memory", "16g",
            "--memory-swap", "32g",
            "wasserth/totalsegmentator:2.10.0",
            "TotalSegmentator",
            "-i", f"/input/{ct_path.name}",
            "-o", "/output",
            "--task", self.task,
            # メモリ節約オプション
            "--nr_thr_resamp", "1",  # リサンプリング時のスレッド数を1に
            "--nr_thr_saving", "1"   # 保存時のスレッド数を1に
        ]
        
        # 特定のタスクにはライセンスキーオプションが必要
        if self.task in ["tissue_types", "heartchambers_highres"]:
            cmd.extend(["-l", self.license_key])
        
        # 実行
        start_time = time.time()
        try:
            self.logger.debug(f"コマンド: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            elapsed_time = time.time() - start_time
            
            # 出力ファイル数を確認
            output_files = list(output_dir.glob("*.nii.gz"))
            self.logger.info(f"完了: {ct_path.name} ({elapsed_time:.1f}秒, {len(output_files)}ファイル生成)")
            
            # デバッグ情報
            if result.stdout:
                self.logger.debug(f"stdout: {result.stdout[:500]}...")  # 最初の500文字のみ
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"エラー: {ct_path.name}")
            self.logger.error(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"予期しないエラー: {ct_path.name}: {str(e)}")
            return False
    
    def is_already_processed(self, ct_file: Dict) -> bool:
        """
        CTファイルが既に処理済みかをチェック
        出力ディレクトリに十分な数のファイルが存在するか確認
        """
        output_dir = self.get_output_dir_path(ct_file)
        
        if not output_dir.exists():
            return False
        
        # 出力ファイル数をチェック（totalタスクは通常100以上のファイルを生成）
        output_files = list(output_dir.glob("*.nii.gz"))
        min_expected_files = {
            "total": 100,
            "tissue_types": 2,  # fat, muscle
            "heartchambers_highres": 4,  # 心房・心室
            "body": 1,
            "bones_extremities": 20
        }
        
        expected = min_expected_files.get(self.task, 1)
        return len(output_files) >= expected
    
    def process_all(self, resume: bool = True):
        """
        すべてのCTファイルを処理
        
        Args:
            resume: 前回の続きから再開するか
        """
        self.logger.info("=" * 60)
        self.logger.info("バッチ処理を開始します")
        self.logger.info(f"タスク: {self.task}")
        self.logger.info(f"出力先: {self.output_base_dir}")
        self.logger.info("=" * 60)
        
        total_count = len(self.ct_files)
        processed_count = 0
        success_count = 0
        fail_count = 0
        skip_count = 0
        
        for i, ct_file in enumerate(self.ct_files, 1):
            ct_path = ct_file['path']
            
            # 既に処理済みかチェック（progress.jsonと実際のファイルの両方を確認）
            if resume:
                # 実際のファイルが存在するかチェック
                if self.is_already_processed(ct_file):
                    self.logger.info(f"[{i}/{total_count}] スキップ（ファイル存在）: {Path(ct_path).name}")
                    skip_count += 1
                    # progress.jsonにも記録
                    if ct_path not in self.progress['completed']:
                        self.progress['completed'].append(ct_path)
                        self.save_progress()
                    continue
                # progress.jsonに記録されているが、ファイルが不完全な場合は再処理
                elif ct_path in self.progress['completed']:
                    self.logger.info(f"[{i}/{total_count}] 再処理（ファイル不完全）: {Path(ct_path).name}")
                    self.progress['completed'].remove(ct_path)
            
            # 前回失敗したものも再試行するかの判断
            if resume and ct_path in self.progress['failed']:
                self.logger.info(f"[{i}/{total_count}] 再試行: {ct_path}")
            
            self.logger.info(f"[{i}/{total_count}] 処理中...")
            
            # TotalSegmentator実行
            success = self.run_totalsegmentator(ct_file)
            processed_count += 1
            
            if success:
                success_count += 1
                self.progress['completed'].append(ct_path)
                # 失敗リストから削除（再試行成功の場合）
                if ct_path in self.progress['failed']:
                    self.progress['failed'].remove(ct_path)
            else:
                fail_count += 1
                if ct_path not in self.progress['failed']:
                    self.progress['failed'].append(ct_path)
            
            # 進捗を保存
            self.save_progress()
            
            # 統計を表示
            self.logger.info(f"進捗: 処理済み {processed_count}, 成功 {success_count}, "
                           f"失敗 {fail_count}, スキップ {skip_count}")
            
            # 少し待機（Docker環境への負荷軽減）
            if i < total_count:
                time.sleep(2)
        
        # 最終統計
        self.logger.info("=" * 60)
        self.logger.info("バッチ処理完了")
        self.logger.info(f"総ファイル数: {total_count}")
        self.logger.info(f"処理済み: {processed_count}")
        self.logger.info(f"成功: {success_count}")
        self.logger.info(f"失敗: {fail_count}")
        self.logger.info(f"スキップ: {skip_count}")
        
        if self.progress['failed']:
            self.logger.warning(f"失敗したファイル: {self.progress['failed']}")
        
        self.logger.info("=" * 60)
    
    def generate_report(self):
        """処理結果のレポートを生成"""
        report_file = self.output_base_dir / "report.txt"
        
        with open(report_file, 'w') as f:
            f.write("TotalSegmentator バッチ処理レポート\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"タスク: {self.task}\n")
            f.write(f"出力ディレクトリ: {self.output_base_dir}\n")
            f.write("\n")
            
            f.write("処理結果:\n")
            f.write(f"  総ファイル数: {len(self.ct_files)}\n")
            f.write(f"  完了: {len(self.progress['completed'])}\n")
            f.write(f"  失敗: {len(self.progress['failed'])}\n")
            f.write("\n")
            
            if self.progress['completed']:
                f.write("完了したファイル:\n")
                for ct_path in self.progress['completed']:
                    f.write(f"  ✓ {ct_path}\n")
                f.write("\n")
            
            if self.progress['failed']:
                f.write("失敗したファイル:\n")
                for ct_path in self.progress['failed']:
                    f.write(f"  ✗ {ct_path}\n")
        
        self.logger.info(f"レポートを生成しました: {report_file}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="TotalSegmentator バッチ処理システム"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="処理するCTファイルリストのJSON設定ファイル"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="出力ベースディレクトリ"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="total",
        choices=["total", "tissue_types", "heartchambers_highres", "body", "bones_extremities"],
        help="TotalSegmentatorのタスク [デフォルト: total]"
    )
    parser.add_argument(
        "--license-key", "-l",
        type=str,
        default="aca_DHDKRFJMHUX0PK",
        help="TotalSegmentatorライセンスキー"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="前回の続きから再開しない（最初から実行）"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="レポートのみ生成（処理は実行しない）"
    )
    
    args = parser.parse_args()
    
    # バッチプロセッサを初期化
    processor = BatchTotalSegmentator(
        config_file=args.config,
        output_base_dir=args.output,
        task=args.task,
        license_key=args.license_key
    )
    
    if args.report:
        # レポートのみ生成
        processor.generate_report()
    else:
        # バッチ処理を実行
        processor.process_all(resume=not args.no_resume)
        # レポートも生成
        processor.generate_report()


if __name__ == "__main__":
    main()