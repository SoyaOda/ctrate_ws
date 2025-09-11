#!/usr/bin/env python3
"""
CTスキャン バッチ処理スクリプト
数万例のCTスキャンを効率的に処理するためのバッチシステム

使用方法:
    python batch_process_ct.py <INPUT_DIR> <OUTPUT_DIR> [options]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import multiprocessing

from ct_processing_pipeline import CTProcessingPipeline, ResourceMonitor


class BatchProcessor:
    """バッチ処理メインクラス"""
    
    def __init__(self, 
                 input_dir: Path, 
                 output_dir: Path,
                 max_workers: int = 1,
                 dry_run: bool = False,
                 verbose: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.dry_run = dry_run
        self.verbose = verbose
        
        # ログ設定
        log_level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # リソース監視
        self.monitor = ResourceMonitor()
        
        # 統計
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None,
            'failed_files': []
        }
    
    def find_ct_files(self) -> List[Path]:
        """CTファイルを検索"""
        self.logger.info(f"🔍 CTファイル検索中: {self.input_dir}")
        
        ct_files = []
        patterns = ["*.nii.gz", "*.nii"]
        
        for pattern in patterns:
            ct_files.extend(self.input_dir.rglob(pattern))
        
        # ファイル名でソート（一貫性のため）
        ct_files.sort()
        
        self.logger.info(f"📊 発見したCTファイル数: {len(ct_files)}")
        
        if self.verbose and ct_files:
            self.logger.info("📋 最初の5ファイル:")
            for i, file_path in enumerate(ct_files[:5]):
                self.logger.info(f"  {i+1}. {file_path.relative_to(self.input_dir)}")
        
        return ct_files
    
    def get_output_path(self, ct_file: Path) -> Path:
        """CTファイルに対応する出力パスを生成"""
        # 相対パスを取得し、.nii.gzを除去
        rel_path = ct_file.relative_to(self.input_dir)
        stem = rel_path.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]
        
        return self.output_dir / rel_path.parent / stem
    
    def is_already_processed(self, output_path: Path) -> bool:
        """すでに処理済みかチェック"""
        # 3つのステップがすべて完了していることを確認
        step1_complete = (output_path / "step1_eat_pat" / "masks" / "eat_pat.nii.gz").exists()
        step2_complete = len(list((output_path / "step2_drr").glob("*.png"))) > 0
        step3_complete = len(list((output_path / "step3_mask_projection").glob("*.png"))) > 0
        
        return step1_complete and step2_complete and step3_complete
    
    def process_single_file(self, ct_file: Path) -> Dict[str, Any]:
        """単一ファイルを処理"""
        output_path = self.get_output_path(ct_file)
        
        result = {
            'ct_file': str(ct_file),
            'output_path': str(output_path),
            'success': False,
            'start_time': time.time(),
            'end_time': None,
            'duration': 0,
            'error': None,
            'skipped': False
        }
        
        try:
            # スキップ判定
            if self.is_already_processed(output_path):
                self.logger.info(f"⏭️  スキップ（処理済み）: {ct_file.name}")
                result['skipped'] = True
                result['success'] = True
                result['end_time'] = result['start_time']
                return result
            
            if self.dry_run:
                self.logger.info(f"🧪 ドライラン: {ct_file.name}")
                result['success'] = True
                result['end_time'] = result['start_time']
                return result
            
            # 実際の処理
            self.logger.info(f"🚀 処理開始: {ct_file.name}")
            
            pipeline = CTProcessingPipeline(
                ct_path=ct_file,
                output_dir=output_path,
                verbose=False  # バッチモードでは個別ログを抑制
            )
            
            success = pipeline.run_pipeline()
            
            result['success'] = success
            result['end_time'] = time.time()
            result['duration'] = result['end_time'] - result['start_time']
            
            if success:
                self.logger.info(f"✅ 処理完了: {ct_file.name} ({result['duration']:.1f}秒)")
            else:
                self.logger.error(f"❌ 処理失敗: {ct_file.name}")
                result['error'] = "Pipeline execution failed"
            
        except Exception as e:
            result['error'] = str(e)
            result['end_time'] = time.time()
            result['duration'] = result['end_time'] - result['start_time']
            self.logger.error(f"❌ エラー: {ct_file.name} - {str(e)}")
        
        return result
    
    def save_progress_report(self, results: List[Dict[str, Any]]):
        """進捗レポートを保存"""
        report_file = self.output_dir / "batch_processing_report.json"
        
        # 統計計算
        successful = [r for r in results if r['success'] and not r['skipped']]
        failed = [r for r in results if not r['success']]
        skipped = [r for r in results if r['skipped']]
        
        total_duration = sum(r['duration'] for r in results)
        avg_duration = total_duration / len(results) if results else 0
        processing_duration = sum(r['duration'] for r in successful)
        
        report = {
            'summary': {
                'total_files': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'skipped': len(skipped),
                'success_rate': len(successful) / len(results) * 100 if results else 0,
                'total_duration_seconds': total_duration,
                'average_duration_seconds': avg_duration,
                'processing_duration_seconds': processing_duration,
                'estimated_time_remaining': None
            },
            'system_info': self.monitor.get_system_info(),
            'batch_config': {
                'input_dir': str(self.input_dir),
                'output_dir': str(self.output_dir),
                'max_workers': self.max_workers,
                'dry_run': self.dry_run
            },
            'results': results
        }
        
        # 残り時間推定（実行中の場合）
        if successful and len(results) < self.stats['total_files']:
            avg_processing_time = processing_duration / len(successful)
            remaining_files = self.stats['total_files'] - len(results)
            estimated_remaining = avg_processing_time * remaining_files / self.max_workers
            report['summary']['estimated_time_remaining'] = estimated_remaining
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 進捗レポート保存: {report_file}")
    
    def run_batch(self) -> bool:
        """バッチ処理実行"""
        self.logger.info("🔄 バッチ処理開始")
        self.logger.info(f"📁 入力ディレクトリ: {self.input_dir}")
        self.logger.info(f"📂 出力ディレクトリ: {self.output_dir}")
        self.logger.info(f"⚡ 最大並列数: {self.max_workers}")
        self.logger.info(f"🧪 ドライラン: {'はい' if self.dry_run else 'いいえ'}")
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CTファイル検索
        ct_files = self.find_ct_files()
        if not ct_files:
            self.logger.error("❌ CTファイルが見つかりません")
            return False
        
        self.stats['total_files'] = len(ct_files)
        self.stats['start_time'] = time.time()
        
        # システム情報ログ
        sys_info = self.monitor.get_system_info()
        self.logger.info(f"🖥️  システム: CPU {sys_info['cpu_count']}コア, "
                        f"Memory {sys_info['memory_total_gb']}GB, "
                        f"GPU {sys_info['gpu_count']}台")
        
        # バッチ処理実行
        results = []
        
        try:
            if self.max_workers == 1:
                # シーケンシャル処理
                for i, ct_file in enumerate(ct_files, 1):
                    self.logger.info(f"📋 進捗: {i}/{len(ct_files)} ({i/len(ct_files)*100:.1f}%)")
                    result = self.process_single_file(ct_file)
                    results.append(result)
                    
                    # 定期的にレポート保存
                    if i % 10 == 0:
                        self.save_progress_report(results)
            else:
                # 並列処理
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_file = {executor.submit(self.process_single_file, ct_file): ct_file 
                                      for ct_file in ct_files}
                    
                    for i, future in enumerate(as_completed(future_to_file), 1):
                        try:
                            result = future.result()
                            results.append(result)
                            self.logger.info(f"📋 進捗: {i}/{len(ct_files)} ({i/len(ct_files)*100:.1f}%)")
                            
                            # 定期的にレポート保存
                            if i % 10 == 0:
                                self.save_progress_report(results)
                        except Exception as e:
                            ct_file = future_to_file[future]
                            self.logger.error(f"❌ タスクエラー: {ct_file.name} - {str(e)}")
        
        except KeyboardInterrupt:
            self.logger.warning("⚠️ ユーザーによる中断")
        
        # 最終統計
        self.stats['end_time'] = time.time()
        self.stats['processed'] = len(results)
        self.stats['successful'] = sum(1 for r in results if r['success'] and not r['skipped'])
        self.stats['failed'] = sum(1 for r in results if not r['success'])
        self.stats['skipped'] = sum(1 for r in results if r['skipped'])
        
        # 最終レポート保存
        self.save_progress_report(results)
        
        # 結果サマリ
        total_time = self.stats['end_time'] - self.stats['start_time']
        self.logger.info("="*60)
        self.logger.info("📊 バッチ処理結果")
        self.logger.info(f"  総ファイル数: {self.stats['total_files']}")
        self.logger.info(f"  処理済み: {self.stats['processed']}")
        self.logger.info(f"  成功: {self.stats['successful']}")
        self.logger.info(f"  失敗: {self.stats['failed']}")
        self.logger.info(f"  スキップ: {self.stats['skipped']}")
        self.logger.info(f"  成功率: {self.stats['successful']/self.stats['processed']*100:.1f}%" if self.stats['processed'] > 0 else "  成功率: 0%")
        self.logger.info(f"  総実行時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
        
        return self.stats['failed'] == 0


def main():
    parser = argparse.ArgumentParser(
        description="CTスキャン バッチ処理システム",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "input_dir",
        help="CTファイルが格納されたディレクトリ"
    )
    
    parser.add_argument(
        "output_dir",
        help="出力ディレクトリ"
    )
    
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=1,
        help="最大並列処理数（デフォルト: 1, GPU制約により通常は1を推奨）"
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true", 
        help="ドライラン（実際の処理は行わず、ファイル検索とパス生成のみ）"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細ログ出力"
    )
    
    args = parser.parse_args()
    
    # 並列数の検証
    if args.max_workers > 1:
        print("⚠️ 警告: GPU使用のため、通常はmax-workers=1を推奨")
        print("   複数GPUがある場合のみ並列数を増やしてください")
        response = input("続行しますか？ (y/N): ")
        if response.lower() != 'y':
            print("処理を中断しました")
            return
    
    # バッチ処理実行
    processor = BatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    success = processor.run_batch()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()