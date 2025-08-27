#!/usr/bin/env python3
"""
EDTベースと反復膨張の比較
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# データ読み込み
with open('/home/soya/ctrate_ws/outputs/eat_pat_edt/statistics/eat_pat_analysis.json', 'r') as f:
    edt_stats = json.load(f)

with open('/home/soya/ctrate_ws/outputs/eat_pat_integrated/statistics/eat_pat_analysis.json', 'r') as f:
    iter_stats = json.load(f)

# 比較表の作成
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 体積比較
categories = ['Heart', 'Shell', 'EAT+PAT', 'Visceral Fat']
edt_volumes = [
    edt_stats['volumes_ml']['heart'],
    edt_stats['volumes_ml']['shell'],
    edt_stats['volumes_ml']['eat_pat'],
    edt_stats['volumes_ml']['visceral_fat']
]
iter_volumes = [
    iter_stats['volumes_ml']['heart'],
    iter_stats['volumes_ml']['shell'],
    iter_stats['volumes_ml']['eat_pat'],
    iter_stats['volumes_ml']['visceral_fat']
]

x = range(len(categories))
width = 0.35

bars1 = ax1.bar([i - width/2 for i in x], edt_volumes, width, label='EDT-based (新)', color='skyblue')
bars2 = ax1.bar([i + width/2 for i in x], iter_volumes, width, label='Iteration-based (旧)', color='lightcoral')

ax1.set_xlabel('Component', fontsize=12)
ax1.set_ylabel('Volume (ml)', fontsize=12)
ax1.set_title('Volume Comparison: EDT vs Iteration-based Dilation', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 値をバーの上に表示
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

# 比率比較
ratios_labels = ['Fat fraction\nin shell (%)', 'EAT+PAT /\nVisceral (%)', 'EAT+PAT /\nHeart (%)']
edt_ratios = [
    edt_stats['ratios_percent']['fat_fraction_in_shell'],
    edt_stats['ratios_percent']['eat_pat_to_visceral'],
    edt_stats['ratios_percent']['eat_pat_to_heart']
]
iter_ratios = [
    iter_stats['ratios_percent']['fat_fraction_in_shell'],
    iter_stats['ratios_percent']['eat_pat_to_visceral'],
    iter_stats['ratios_percent']['eat_pat_to_heart']
]

x2 = range(len(ratios_labels))
bars3 = ax2.bar([i - width/2 for i in x2], edt_ratios, width, label='EDT-based (新)', color='skyblue')
bars4 = ax2.bar([i + width/2 for i in x2], iter_ratios, width, label='Iteration-based (旧)', color='lightcoral')

ax2.set_xlabel('Metric', fontsize=12)
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_title('Ratio Comparison: EDT vs Iteration-based', fontsize=14, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(ratios_labels)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 値をバーの上に表示
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

# 重要な改善点のテキスト追加
fig.text(0.5, 0.02, 
         'EDT改善点: (1) Shell体積が669ml (旧1266ml)に減少 → より正確な15mm膨張\n'
         '(2) 脂肪割合が12.5% (旧7.3%)に増加 → Shell縮小による相対的増加\n'
         '(3) 物理的距離での等方的膨張により、Z方向の歪みを解消',
         ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('/home/soya/ctrate_ws/outputs/edt_vs_iteration_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== 比較結果 ===")
print(f"Shell体積削減: {iter_stats['volumes_ml']['shell']:.1f} ml → {edt_stats['volumes_ml']['shell']:.1f} ml "
      f"({(1 - edt_stats['volumes_ml']['shell']/iter_stats['volumes_ml']['shell'])*100:.1f}%減少)")
print(f"EAT+PAT体積: {iter_stats['volumes_ml']['eat_pat']:.1f} ml → {edt_stats['volumes_ml']['eat_pat']:.1f} ml")
print(f"Shell内脂肪割合: {iter_stats['ratios_percent']['fat_fraction_in_shell']:.1f}% → "
      f"{edt_stats['ratios_percent']['fat_fraction_in_shell']:.1f}%")
print("\n図を保存: /home/soya/ctrate_ws/outputs/edt_vs_iteration_comparison.png")