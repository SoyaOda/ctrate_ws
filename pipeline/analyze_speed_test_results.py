#!/usr/bin/env python3
"""
TotalSegmentator 高速化テスト結果分析・レポート生成
- 処理時間・品質トレードオフ分析
- 推奨設定の提案
- 数万例処理の実現可能性評価
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import logging


class SpeedTestAnalyzer:
    """高速化テスト結果分析クラス"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # 結果データ読み込み
        self.batch_results = self.load_batch_results()
    
    def load_batch_results(self) -> Dict[str, Any]:
        """バッチテスト結果を読み込み"""
        results_file = self.results_dir / "batch_test_results.json"
        
        if not results_file.exists():
            self.logger.warning(f"結果ファイルが見つかりません: {results_file}")
            return {}
        
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_performance_improvement(self) -> Dict[str, Any]:
        """パフォーマンス改善分析"""
        self.logger.info("📈 パフォーマンス改善分析")
        
        analysis = {
            'time_savings': {},
            'speedup_ratios': [],
            'step_breakdown': {'step1': [], 'step2': []}
        }
        
        for case in self.batch_results.get('test_cases', []):
            case_name = case['case_name']
            baseline = case.get('baseline', {})
            fast = case.get('fast_optimized', {})
            
            if 'total_time' in baseline and 'total_time' in fast:
                baseline_time = baseline['total_time']
                fast_time = fast['total_time']
                
                time_saved = baseline_time - fast_time
                speedup = baseline_time / fast_time if fast_time > 0 else 0
                
                analysis['time_savings'][case_name] = {
                    'baseline_time': baseline_time,
                    'fast_time': fast_time,
                    'time_saved': time_saved,
                    'speedup_ratio': speedup
                }
                
                analysis['speedup_ratios'].append(speedup)
                
                # ステップ別分析
                if 'step1_time' in baseline and 'step1_time' in fast:
                    step1_improvement = baseline['step1_time'] / fast['step1_time'] if fast['step1_time'] > 0 else 0
                    analysis['step_breakdown']['step1'].append(step1_improvement)
                
                if 'step2_time' in baseline and 'step2_time' in fast:
                    step2_improvement = baseline['step2_time'] / fast['step2_time'] if fast['step2_time'] > 0 else 0
                    analysis['step_breakdown']['step2'].append(step2_improvement)
        
        # 統計計算
        if analysis['speedup_ratios']:
            analysis['summary'] = {
                'average_speedup': np.mean(analysis['speedup_ratios']),
                'median_speedup': np.median(analysis['speedup_ratios']),
                'min_speedup': np.min(analysis['speedup_ratios']),
                'max_speedup': np.max(analysis['speedup_ratios']),
                'std_speedup': np.std(analysis['speedup_ratios'])
            }
        
        return analysis
    
    def analyze_quality_impact(self) -> Dict[str, Any]:
        """品質への影響分析"""
        self.logger.info("🎯 品質への影響分析")
        
        analysis = {
            'mask_quality': {},
            'critical_masks': ['heart_myocardium.nii.gz', 'torso_fat.nii.gz'],
            'quality_summary': {}
        }
        
        all_dice_scores = {}
        
        for case in self.batch_results.get('test_cases', []):
            case_name = case['case_name']
            quality_comp = case.get('quality_comparison', {})
            
            analysis['mask_quality'][case_name] = quality_comp
            
            # マスク別Dice係数収集
            for mask_name, dice in quality_comp.items():
                if dice > 0:  # エラー値除外
                    if mask_name not in all_dice_scores:
                        all_dice_scores[mask_name] = []
                    all_dice_scores[mask_name].append(dice)
        
        # マスク別統計
        for mask_name, dice_scores in all_dice_scores.items():
            if dice_scores:
                analysis['quality_summary'][mask_name] = {
                    'mean_dice': np.mean(dice_scores),
                    'min_dice': np.min(dice_scores),
                    'max_dice': np.max(dice_scores),
                    'std_dice': np.std(dice_scores),
                    'count': len(dice_scores),
                    'quality_level': self.assess_quality_level(np.mean(dice_scores))
                }
        
        return analysis
    
    def assess_quality_level(self, dice: float) -> str:
        """Dice係数による品質レベル評価"""
        if dice >= 0.95:
            return "Excellent"
        elif dice >= 0.90:
            return "Very Good"
        elif dice >= 0.85:
            return "Good"
        elif dice >= 0.80:
            return "Acceptable"
        else:
            return "Poor"
    
    def estimate_large_scale_feasibility(self, target_cases: int = 30000) -> Dict[str, Any]:
        """大規模処理の実現可能性評価"""
        self.logger.info(f"🔮 大規模処理実現可能性評価 ({target_cases:,}例)")
        
        performance_analysis = self.analyze_performance_improvement()
        
        if not performance_analysis.get('summary'):
            return {'error': 'パフォーマンス分析データが不足しています'}
        
        avg_baseline_time = np.mean([
            case['baseline']['total_time'] 
            for case in self.batch_results.get('test_cases', [])
            if 'total_time' in case.get('baseline', {})
        ])
        
        avg_fast_time = np.mean([
            case['fast_optimized']['total_time']
            for case in self.batch_results.get('test_cases', [])
            if 'total_time' in case.get('fast_optimized', {})
        ])
        
        if avg_baseline_time == 0 or avg_fast_time == 0:
            return {'error': '処理時間データが不足しています'}
        
        # 時間推定
        baseline_total_hours = (avg_baseline_time * target_cases) / 3600
        fast_total_hours = (avg_fast_time * target_cases) / 3600
        
        baseline_days = baseline_total_hours / 24
        fast_days = fast_total_hours / 24
        
        # 並列処理での改善可能性（仮想的）
        parallel_improvement = 2.0  # 2倍程度の改善を想定
        optimized_days = fast_days / parallel_improvement
        
        feasibility = {
            'target_cases': target_cases,
            'time_estimates': {
                'baseline': {
                    'total_hours': baseline_total_hours,
                    'total_days': baseline_days,
                    'feasibility': 'Impractical' if baseline_days > 365 else 'Challenging'
                },
                'fast_optimized': {
                    'total_hours': fast_total_hours,
                    'total_days': fast_days,
                    'feasibility': self.assess_time_feasibility(fast_days)
                },
                'future_optimized': {
                    'total_hours': fast_total_hours / parallel_improvement,
                    'total_days': optimized_days,
                    'feasibility': self.assess_time_feasibility(optimized_days),
                    'assumptions': 'Docker最適化 + 並列処理改善'
                }
            },
            'recommendations': self.generate_recommendations(fast_days, optimized_days)
        }
        
        return feasibility
    
    def assess_time_feasibility(self, days: float) -> str:
        """処理時間による実現可能性評価"""
        if days <= 7:
            return "Excellent (1週間以内)"
        elif days <= 30:
            return "Very Good (1ヶ月以内)"
        elif days <= 90:
            return "Good (3ヶ月以内)"
        elif days <= 365:
            return "Challenging (1年以内)"
        else:
            return "Impractical (1年超)"
    
    def generate_recommendations(self, fast_days: float, optimized_days: float) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        if fast_days <= 30:
            recommendations.append("現在の高速化設定で実用的な処理時間を達成可能")
        else:
            recommendations.append("追加の最適化が必要")
        
        if optimized_days <= 14:
            recommendations.append("Docker環境最適化により2週間以内の処理が可能")
        
        recommendations.extend([
            "--fast + --roi_subsetオプションの継続使用を推奨",
            "永続化コンテナによるオーバーヘッド削減を実装",
            "品質が許容範囲内であることを定期的に検証"
        ])
        
        return recommendations
    
    def generate_visual_report(self, output_dir: Path):
        """視覚的レポート生成"""
        self.logger.info("📊 視覚的レポート生成")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. パフォーマンス比較グラフ
        self.plot_performance_comparison(output_dir)
        
        # 2. 品質比較グラフ
        self.plot_quality_comparison(output_dir)
        
        # 3. スケーラビリティ予測グラフ
        self.plot_scalability_prediction(output_dir)
    
    def plot_performance_comparison(self, output_dir: Path):
        """パフォーマンス比較グラフ"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 処理時間比較
        case_names = []
        baseline_times = []
        fast_times = []
        
        for case in self.batch_results.get('test_cases', []):
            if 'total_time' in case.get('baseline', {}) and 'total_time' in case.get('fast_optimized', {}):
                case_names.append(case['case_name'])
                baseline_times.append(case['baseline']['total_time'] / 60)  # 分単位
                fast_times.append(case['fast_optimized']['total_time'] / 60)
        
        if case_names:
            x = np.arange(len(case_names))
            width = 0.35
            
            ax1.bar(x - width/2, baseline_times, width, label='通常版', alpha=0.8)
            ax1.bar(x + width/2, fast_times, width, label='高速化版', alpha=0.8)
            ax1.set_xlabel('症例')
            ax1.set_ylabel('処理時間 (分)')
            ax1.set_title('処理時間比較')
            ax1.set_xticks(x)
            ax1.set_xticklabels(case_names, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 高速化率
        speedup_ratios = [b/f for b, f in zip(baseline_times, fast_times) if f > 0]
        if speedup_ratios:
            ax2.bar(case_names, speedup_ratios, alpha=0.8, color='green')
            ax2.set_xlabel('症例')
            ax2.set_ylabel('高速化倍率')
            ax2.set_title('高速化効果')
            ax2.set_xticklabels(case_names, rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 平均線
            avg_speedup = np.mean(speedup_ratios)
            ax2.axhline(y=avg_speedup, color='red', linestyle='--', 
                       label=f'平均: {avg_speedup:.2f}x')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_quality_comparison(self, output_dir: Path):
        """品質比較グラフ"""
        quality_analysis = self.analyze_quality_impact()
        
        mask_names = []
        mean_dice_scores = []
        
        for mask_name, stats in quality_analysis.get('quality_summary', {}).items():
            mask_names.append(mask_name.replace('.nii.gz', ''))
            mean_dice_scores.append(stats['mean_dice'])
        
        if mask_names:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(mask_names, mean_dice_scores, alpha=0.8)
            
            # 品質レベル別色分け
            for bar, score in zip(bars, mean_dice_scores):
                if score >= 0.95:
                    bar.set_color('darkgreen')
                elif score >= 0.90:
                    bar.set_color('green')
                elif score >= 0.85:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            plt.xlabel('マスク種類')
            plt.ylabel('Dice係数')
            plt.title('高速化版 vs 通常版 マスク品質比較')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0.5, 1.0)
            plt.grid(True, alpha=0.3)
            
            # 品質閾値線
            plt.axhline(y=0.95, color='darkgreen', linestyle='--', alpha=0.5, label='Excellent (≥0.95)')
            plt.axhline(y=0.90, color='green', linestyle='--', alpha=0.5, label='Very Good (≥0.90)')
            plt.axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='Good (≥0.85)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / "quality_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    def plot_scalability_prediction(self, output_dir: Path):
        """スケーラビリティ予測グラフ"""
        case_counts = [100, 1000, 5000, 10000, 30000, 50000]
        
        # 平均処理時間取得
        avg_fast_time = np.mean([
            case['fast_optimized']['total_time']
            for case in self.batch_results.get('test_cases', [])
            if 'total_time' in case.get('fast_optimized', {})
        ])
        
        if avg_fast_time > 0:
            # 処理時間予測（日単位）
            current_days = [(count * avg_fast_time) / (3600 * 24) for count in case_counts]
            optimized_days = [days / 2.0 for days in current_days]  # 2倍改善を想定
            
            plt.figure(figsize=(12, 8))
            plt.plot(case_counts, current_days, 'o-', label='現在の高速化版', linewidth=2)
            plt.plot(case_counts, optimized_days, 's-', label='将来の最適化版', linewidth=2)
            
            plt.xlabel('処理症例数')
            plt.ylabel('総処理時間 (日)')
            plt.title('大規模処理時間予測')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 実用性の目安線
            plt.axhline(y=7, color='green', linestyle='--', alpha=0.7, label='1週間')
            plt.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='1ヶ月') 
            plt.axhline(y=365, color='red', linestyle='--', alpha=0.7, label='1年')
            
            plt.tight_layout()
            plt.savefig(output_dir / "scalability_prediction.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    def generate_comprehensive_report(self, output_dir: Path):
        """包括的レポート生成"""
        self.logger.info("📋 包括的レポート生成")
        
        # 分析実行
        performance_analysis = self.analyze_performance_improvement()
        quality_analysis = self.analyze_quality_impact()
        feasibility_analysis = self.estimate_large_scale_feasibility()
        
        # レポート作成
        report = {
            'summary': {
                'test_cases': len(self.batch_results.get('test_cases', [])),
                'analysis_timestamp': self.batch_results.get('timestamp'),
                'average_speedup': performance_analysis.get('summary', {}).get('average_speedup', 0),
                'quality_maintained': self.assess_overall_quality(quality_analysis)
            },
            'performance_analysis': performance_analysis,
            'quality_analysis': quality_analysis,
            'scalability_analysis': feasibility_analysis
        }
        
        # JSON保存
        report_file = output_dir / "comprehensive_analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # テキストサマリ生成
        self.generate_text_summary(report, output_dir)
        
        # 視覚的レポート
        self.generate_visual_report(output_dir)
        
        self.logger.info(f"📊 包括的レポート完成: {output_dir}")
        return report
    
    def assess_overall_quality(self, quality_analysis: Dict[str, Any]) -> bool:
        """全体的な品質維持評価"""
        quality_summary = quality_analysis.get('quality_summary', {})
        
        # 重要マスクの品質チェック
        critical_masks = ['heart_myocardium.nii.gz', 'torso_fat.nii.gz']
        
        for mask_name in critical_masks:
            if mask_name in quality_summary:
                mean_dice = quality_summary[mask_name]['mean_dice']
                if mean_dice < 0.85:  # 品質閾値
                    return False
        
        return True
    
    def generate_text_summary(self, report: Dict[str, Any], output_dir: Path):
        """テキストサマリ生成"""
        summary_file = output_dir / "speed_optimization_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# TotalSegmentator 高速化テスト - 分析サマリ\n")
            f.write(f"分析日時: {report['summary']['analysis_timestamp']}\n\n")
            
            # パフォーマンス
            f.write("## パフォーマンス改善\n")
            perf = report.get('performance_analysis', {}).get('summary', {})
            if perf:
                f.write(f"平均高速化: {perf['average_speedup']:.2f}x\n")
                f.write(f"最大高速化: {perf['max_speedup']:.2f}x\n")
                f.write(f"最小高速化: {perf['min_speedup']:.2f}x\n\n")
            
            # 品質
            f.write("## 品質への影響\n")
            quality_maintained = report['summary']['quality_maintained']
            f.write(f"品質維持: {'✅ 良好' if quality_maintained else '⚠️ 要注意'}\n")
            
            quality_summary = report.get('quality_analysis', {}).get('quality_summary', {})
            for mask_name, stats in quality_summary.items():
                f.write(f"{mask_name}: Dice = {stats['mean_dice']:.3f} ({stats['quality_level']})\n")
            f.write("\n")
            
            # スケーラビリティ
            f.write("## 大規模処理実現可能性\n")
            feasibility = report.get('scalability_analysis', {})
            if 'time_estimates' in feasibility:
                fast_est = feasibility['time_estimates']['fast_optimized']
                f.write(f"3万例処理予想時間: {fast_est['total_days']:.1f}日\n")
                f.write(f"実現可能性: {fast_est['feasibility']}\n\n")
            
            # 推奨事項
            recommendations = feasibility.get('recommendations', [])
            if recommendations:
                f.write("## 推奨事項\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")


def main():
    parser = argparse.ArgumentParser(description="TotalSegmentator高速化テスト結果分析")
    parser.add_argument("--results-dir", "-r", 
                       default="/home/soya/ctrate_ws/pipeline/totalseg_speed_test_results",
                       help="テスト結果ディレクトリ")
    parser.add_argument("--output-dir", "-o",
                       default="/home/soya/ctrate_ws/pipeline/analysis_reports",
                       help="分析レポート出力ディレクトリ")
    
    args = parser.parse_args()
    
    analyzer = SpeedTestAnalyzer(args.results_dir)
    report = analyzer.generate_comprehensive_report(Path(args.output_dir))
    
    print("分析レポート生成完了!")
    print(f"出力先: {args.output_dir}")


if __name__ == "__main__":
    main()