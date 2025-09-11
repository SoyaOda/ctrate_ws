#!/usr/bin/env python3
"""
パフォーマンス分析スクリプト
パイプラインの実行ログを解析して数万例処理の実現可能性を評価

使用方法:
    python analyze_performance.py <LOG_DIR> [options]
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

class PerformanceAnalyzer:
    """パフォーマンス分析クラス"""
    
    def __init__(self, log_dir: Path, output_dir: Path = None):
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir) if output_dir else self.log_dir / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ格納
        self.resource_logs = []
        self.performance_data = []
    
    def load_logs(self) -> bool:
        """ログファイルを読み込み"""
        print(f"📁 ログディレクトリ: {self.log_dir}")
        
        # resource_usage.jsonファイルを検索
        resource_files = list(self.log_dir.rglob("resource_usage.json"))
        batch_reports = list(self.log_dir.rglob("batch_processing_report.json"))
        
        print(f"🔍 発見ファイル: リソースログ {len(resource_files)}個, バッチレポート {len(batch_reports)}個")
        
        # リソースログ読み込み
        for file_path in resource_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.resource_logs.append({
                        'file': file_path,
                        'data': data
                    })
            except Exception as e:
                print(f"❌ ログ読み込みエラー: {file_path} - {str(e)}")
        
        # バッチレポート読み込み  
        for file_path in batch_reports:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.performance_data.append({
                        'file': file_path,
                        'data': data
                    })
            except Exception as e:
                print(f"❌ レポート読み込みエラー: {file_path} - {str(e)}")
        
        return len(self.resource_logs) > 0 or len(self.performance_data) > 0
    
    def analyze_single_case_performance(self) -> Dict[str, Any]:
        """単一症例のパフォーマンス分析"""
        if not self.resource_logs:
            return {}
        
        print("📊 単一症例パフォーマンス分析")
        
        analysis = {
            'cases_analyzed': len(self.resource_logs),
            'step_durations': {'step1': [], 'step2': [], 'step3': []},
            'memory_usage': {'cpu': [], 'gpu': []},
            'gpu_utilization': [],
            'total_durations': []
        }
        
        for log_entry in self.resource_logs:
            data = log_entry['data']
            
            # パイプライン総時間
            pipeline_info = data.get('pipeline_info', {})
            total_time = pipeline_info.get('total_time', 0)
            if total_time > 0:
                analysis['total_durations'].append(total_time)
            
            # ステップ別時間分析
            resource_logs = data.get('resource_logs', [])
            step_times = {}
            
            for entry in resource_logs:
                step = entry.get('step', '')
                stage = entry.get('stage', '')
                timestamp = entry.get('timestamp', 0)
                
                if step not in step_times:
                    step_times[step] = {}
                
                step_times[step][stage] = timestamp
                
                # メモリ使用量
                if entry.get('memory_percent'):
                    analysis['memory_usage']['cpu'].append(entry['memory_percent'])
                
                # GPU情報
                gpu_usage = entry.get('gpu_usage', [])
                for gpu in gpu_usage:
                    if gpu.get('memory_percent'):
                        analysis['memory_usage']['gpu'].append(gpu['memory_percent'])
                    if gpu.get('gpu_utilization'):
                        analysis['gpu_utilization'].append(gpu['gpu_utilization'])
            
            # ステップ時間計算
            for step in ['step1', 'step2', 'step3']:
                if step in step_times:
                    times = step_times[step]
                    if 'start' in times and 'complete' in times:
                        duration = times['complete'] - times['start']
                        analysis['step_durations'][step].append(duration)
        
        # 統計計算
        stats = {}
        for key, values in analysis.items():
            if key == 'step_durations':
                stats[key] = {}
                for step, durations in values.items():
                    if durations:
                        stats[key][step] = {
                            'mean': np.mean(durations),
                            'std': np.std(durations),
                            'min': np.min(durations),
                            'max': np.max(durations),
                            'count': len(durations)
                        }
            elif key == 'memory_usage':
                stats[key] = {}
                for mem_type, values_list in values.items():
                    if values_list:
                        stats[key][mem_type] = {
                            'mean': np.mean(values_list),
                            'max': np.max(values_list),
                            'count': len(values_list)
                        }
            elif isinstance(values, list) and values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return stats
    
    def analyze_batch_performance(self) -> Dict[str, Any]:
        """バッチ処理パフォーマンス分析"""
        if not self.performance_data:
            return {}
        
        print("📊 バッチ処理パフォーマンス分析")
        
        analysis = {
            'total_files_processed': 0,
            'success_rates': [],
            'processing_speeds': [],  # files/hour
            'failure_patterns': {},
            'durations': []
        }
        
        for report_entry in self.performance_data:
            data = report_entry['data']
            summary = data.get('summary', {})
            
            # 基本統計
            total_files = summary.get('total_files', 0)
            successful = summary.get('successful', 0)
            processing_duration = summary.get('processing_duration_seconds', 0)
            
            analysis['total_files_processed'] += total_files
            
            if total_files > 0:
                success_rate = successful / total_files * 100
                analysis['success_rates'].append(success_rate)
            
            if processing_duration > 0 and successful > 0:
                files_per_hour = successful / (processing_duration / 3600)
                analysis['processing_speeds'].append(files_per_hour)
            
            # 個別結果分析
            results = data.get('results', [])
            for result in results:
                if result.get('duration'):
                    analysis['durations'].append(result['duration'])
                
                if not result.get('success') and result.get('error'):
                    error = result['error']
                    if error in analysis['failure_patterns']:
                        analysis['failure_patterns'][error] += 1
                    else:
                        analysis['failure_patterns'][error] = 1
        
        return analysis
    
    def estimate_scale_feasibility(self, single_stats: Dict, batch_stats: Dict) -> Dict[str, Any]:
        """数万例処理の実現可能性を推定"""
        print("🔮 スケール実現可能性分析")
        
        # 基準値
        target_files = 30000  # 3万例
        
        estimates = {
            'target_files': target_files,
            'feasibility': 'unknown',
            'bottlenecks': [],
            'recommendations': []
        }
        
        # 単一症例の平均処理時間
        if single_stats.get('total_durations', {}).get('mean'):
            avg_time_per_file = single_stats['total_durations']['mean']
            total_time_hours = (avg_time_per_file * target_files) / 3600
            total_time_days = total_time_hours / 24
            
            estimates['single_case_estimates'] = {
                'avg_time_per_file_seconds': avg_time_per_file,
                'total_time_hours': total_time_hours,
                'total_time_days': total_time_days
            }
            
            # 実現可能性判定
            if total_time_days < 30:  # 1ヶ月以内
                estimates['feasibility'] = 'feasible'
            elif total_time_days < 90:  # 3ヶ月以内
                estimates['feasibility'] = 'challenging'
            else:
                estimates['feasibility'] = 'difficult'
        
        # バッチ処理による改善推定
        if batch_stats.get('processing_speeds'):
            avg_speed = np.mean(batch_stats['processing_speeds'])
            batch_total_hours = target_files / avg_speed
            batch_total_days = batch_total_hours / 24
            
            estimates['batch_estimates'] = {
                'avg_files_per_hour': avg_speed,
                'total_time_hours': batch_total_hours, 
                'total_time_days': batch_total_days
            }
        
        # ボトルネック特定
        if single_stats.get('step_durations'):
            step_times = single_stats['step_durations']
            slowest_step = None
            slowest_time = 0
            
            for step, stats_dict in step_times.items():
                if stats_dict.get('mean', 0) > slowest_time:
                    slowest_time = stats_dict['mean']
                    slowest_step = step
            
            if slowest_step:
                estimates['bottlenecks'].append(f"最遅ステップ: {slowest_step} ({slowest_time:.1f}秒)")
        
        # GPU メモリ使用量チェック
        if single_stats.get('memory_usage', {}).get('gpu', {}).get('max', 0) > 80:
            estimates['bottlenecks'].append("GPU メモリ使用量が高い (>80%)")
        
        # 推奨事項
        if estimates['feasibility'] == 'difficult':
            estimates['recommendations'].extend([
                "処理アルゴリズムの最適化が必要",
                "複数GPU環境の検討",
                "クラウド並列処理の検討"
            ])
        elif estimates['feasibility'] == 'challenging':
            estimates['recommendations'].extend([
                "バッチサイズの最適化",
                "処理優先順位の設定",
                "段階的処理の検討"
            ])
        else:
            estimates['recommendations'].append("現在の構成で処理可能")
        
        # 失敗率が高い場合
        if batch_stats.get('success_rates') and np.mean(batch_stats['success_rates']) < 90:
            estimates['recommendations'].append("エラーハンドリングの改善が必要")
        
        return estimates
    
    def generate_report(self, single_stats: Dict, batch_stats: Dict, estimates: Dict):
        """分析レポートを生成"""
        print("📝 分析レポート生成中...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'log_directory': str(self.log_dir),
            'single_case_analysis': single_stats,
            'batch_analysis': batch_stats,
            'scalability_estimates': estimates
        }
        
        # JSONレポート保存
        report_file = self.output_dir / "performance_analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"💾 分析レポート保存: {report_file}")
        
        # テキストサマリ生成
        self.generate_text_summary(single_stats, batch_stats, estimates)
        
        # 可視化グラフ生成
        self.generate_visualizations(single_stats, batch_stats)
    
    def generate_text_summary(self, single_stats: Dict, batch_stats: Dict, estimates: Dict):
        """テキストサマリ生成"""
        summary_file = self.output_dir / "performance_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# CTスキャン処理パイプライン パフォーマンス分析レポート\\n")
            f.write(f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            # 単一症例分析
            if single_stats:
                f.write("## 単一症例分析\\n")
                if single_stats.get('total_durations'):
                    avg_time = single_stats['total_durations']['mean']
                    f.write(f"平均処理時間: {avg_time:.1f}秒 ({avg_time/60:.1f}分)\\n")
                
                if single_stats.get('step_durations'):
                    f.write("\\nステップ別処理時間:\\n")
                    for step, stats_dict in single_stats['step_durations'].items():
                        if stats_dict:
                            f.write(f"  {step}: {stats_dict['mean']:.1f}±{stats_dict['std']:.1f}秒\\n")
                
                if single_stats.get('memory_usage'):
                    f.write("\\nメモリ使用量:\\n")
                    cpu_mem = single_stats['memory_usage'].get('cpu', {})
                    if cpu_mem:
                        f.write(f"  CPU: 平均{cpu_mem['mean']:.1f}%, 最大{cpu_mem['max']:.1f}%\\n")
                    gpu_mem = single_stats['memory_usage'].get('gpu', {})
                    if gpu_mem:
                        f.write(f"  GPU: 平均{gpu_mem['mean']:.1f}%, 最大{gpu_mem['max']:.1f}%\\n")
                
                f.write("\\n")
            
            # スケーラビリティ推定
            if estimates:
                f.write("## 数万例処理の実現可能性\\n")
                f.write(f"目標ファイル数: {estimates.get('target_files', 'N/A'):,}例\\n")
                f.write(f"実現可能性: {estimates.get('feasibility', 'unknown')}\\n")
                
                if estimates.get('single_case_estimates'):
                    est = estimates['single_case_estimates']
                    f.write(f"推定総処理時間: {est.get('total_time_days', 0):.1f}日\\n")
                
                if estimates.get('bottlenecks'):
                    f.write("\\nボトルネック:\\n")
                    for bottleneck in estimates['bottlenecks']:
                        f.write(f"  - {bottleneck}\\n")
                
                if estimates.get('recommendations'):
                    f.write("\\n推奨事項:\\n")
                    for rec in estimates['recommendations']:
                        f.write(f"  - {rec}\\n")
        
        print(f"📄 サマリ保存: {summary_file}")
    
    def generate_visualizations(self, single_stats: Dict, batch_stats: Dict):
        """可視化グラフ生成"""
        if not single_stats:
            return
        
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        
        # ステップ別処理時間のグラフ
        if single_stats.get('step_durations'):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            steps = []
            means = []
            stds = []
            
            for step, stats_dict in single_stats['step_durations'].items():
                if stats_dict:
                    steps.append(step.replace('step', 'Step '))
                    means.append(stats_dict['mean'])
                    stds.append(stats_dict.get('std', 0))
            
            if steps:
                x = np.arange(len(steps))
                bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
                ax.set_xlabel('処理ステップ')
                ax.set_ylabel('処理時間 (秒)')
                ax.set_title('ステップ別平均処理時間')
                ax.set_xticks(x)
                ax.set_xticklabels(steps)
                
                # 値をバーに表示
                for i, (bar, mean) in enumerate(zip(bars, means)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i],
                           f'{mean:.1f}s', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "step_durations.png", dpi=150)
                plt.close()
        
        print(f"📈 グラフ保存: {self.output_dir}")
    
    def run_analysis(self) -> bool:
        """分析実行"""
        print("🔍 パフォーマンス分析開始")
        
        # ログ読み込み
        if not self.load_logs():
            print("❌ ログファイルが見つかりません")
            return False
        
        # 分析実行
        single_stats = self.analyze_single_case_performance()
        batch_stats = self.analyze_batch_performance()
        estimates = self.estimate_scale_feasibility(single_stats, batch_stats)
        
        # レポート生成
        self.generate_report(single_stats, batch_stats, estimates)
        
        print("✅ 分析完了")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="CTスキャン処理パイプライン パフォーマンス分析",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "log_dir",
        help="ログファイルが格納されたディレクトリ"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        help="分析結果出力ディレクトリ（デフォルト: LOG_DIR/analysis）"
    )
    
    args = parser.parse_args()
    
    # 分析実行
    analyzer = PerformanceAnalyzer(
        log_dir=args.log_dir,
        output_dir=args.output_dir
    )
    
    success = analyzer.run_analysis()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())