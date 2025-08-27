#!/usr/bin/env python3
"""
複数のEAT+PAT抽出結果を比較
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 結果ディレクトリ
results = {
    'valid_2_a_1 (EDT)': '/home/soya/ctrate_ws/outputs/eat_pat_edt/statistics/eat_pat_analysis.json',
    'valid_2_a_1 (Iter)': '/home/soya/ctrate_ws/outputs/eat_pat_integrated/statistics/eat_pat_analysis.json',
    'valid_1_a_1': '/home/soya/ctrate_ws/outputs/eat_pat_valid_1_a_1/statistics/eat_pat_analysis.json',
    'valid_1_a_2': '/home/soya/ctrate_ws/outputs/eat_pat_valid_1_a_2/statistics/eat_pat_analysis.json'
}

# データ読み込み
data = {}
for name, path in results.items():
    if Path(path).exists():
        with open(path, 'r') as f:
            data[name] = json.load(f)
    else:
        print(f"Warning: {path} not found")

# 比較表の作成
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. 体積比較
names = list(data.keys())
colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow']

# Heart volumes
heart_volumes = [data[n]['volumes_ml']['heart'] for n in names]
ax1.bar(names, heart_volumes, color=colors[0])
ax1.set_ylabel('Volume (ml)', fontsize=12)
ax1.set_title('Heart Volume', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)
ax1.set_xticklabels(names, rotation=45, ha='right')
for i, v in enumerate(heart_volumes):
    ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom')

# Shell volumes
shell_volumes = [data[n]['volumes_ml']['shell'] for n in names]
ax2.bar(names, shell_volumes, color=colors[1])
ax2.set_ylabel('Volume (ml)', fontsize=12)
ax2.set_title('Shell Volume', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)
ax2.set_xticklabels(names, rotation=45, ha='right')
for i, v in enumerate(shell_volumes):
    ax2.text(i, v, f'{v:.1f}', ha='center', va='bottom')

# EAT+PAT volumes
eat_pat_volumes = [data[n]['volumes_ml']['eat_pat'] for n in names]
ax3.bar(names, eat_pat_volumes, color=colors[2])
ax3.set_ylabel('Volume (ml)', fontsize=12)
ax3.set_title('EAT+PAT Volume', fontsize=14, fontweight='bold')
ax3.axhline(y=50, color='green', linestyle='--', label='Normal range')
ax3.axhline(y=200, color='red', linestyle='--')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.tick_params(axis='x', rotation=45)
ax3.set_xticklabels(names, rotation=45, ha='right')
for i, v in enumerate(eat_pat_volumes):
    ax3.text(i, v, f'{v:.1f}', ha='center', va='bottom')

# Fat fraction in shell
fat_fractions = [data[n]['ratios_percent']['fat_fraction_in_shell'] for n in names]
ax4.bar(names, fat_fractions, color=colors[3])
ax4.set_ylabel('Percentage (%)', fontsize=12)
ax4.set_title('Fat Fraction in Shell', fontsize=14, fontweight='bold')
ax4.axhline(y=30, color='red', linestyle='--', label='Abnormal threshold')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.tick_params(axis='x', rotation=45)
ax4.set_xticklabels(names, rotation=45, ha='right')
for i, v in enumerate(fat_fractions):
    ax4.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

plt.suptitle('EAT+PAT Extraction Results Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/soya/ctrate_ws/outputs/eat_pat_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 統計サマリーテーブル
print("\n" + "="*80)
print("EAT+PAT Extraction Results Summary")
print("="*80)
print(f"{'Dataset':<20} {'Heart (ml)':<12} {'Shell (ml)':<12} {'EAT+PAT (ml)':<15} {'Fat % in Shell':<15}")
print("-"*80)

for name in names:
    print(f"{name:<20} "
          f"{data[name]['volumes_ml']['heart']:<12.2f} "
          f"{data[name]['volumes_ml']['shell']:<12.2f} "
          f"{data[name]['volumes_ml']['eat_pat']:<15.2f} "
          f"{data[name]['ratios_percent']['fat_fraction_in_shell']:<15.2f}")

print("-"*80)

# 手法による違いの分析（valid_2_a_1のEDT vs Iteration）
if 'valid_2_a_1 (EDT)' in data and 'valid_2_a_1 (Iter)' in data:
    print("\n[Method Comparison for valid_2_a_1]")
    edt = data['valid_2_a_1 (EDT)']
    iter_data = data['valid_2_a_1 (Iter)']
    
    shell_diff = (edt['volumes_ml']['shell'] - iter_data['volumes_ml']['shell']) / iter_data['volumes_ml']['shell'] * 100
    eat_diff = (edt['volumes_ml']['eat_pat'] - iter_data['volumes_ml']['eat_pat']) / iter_data['volumes_ml']['eat_pat'] * 100
    
    print(f"  Shell volume change: {shell_diff:.1f}% (EDT creates smaller shell)")
    print(f"  EAT+PAT volume change: {eat_diff:.1f}%")
    print(f"  EDT method: More physically accurate 15mm expansion")

print("\n[Clinical Interpretation]")
for name in names:
    eat_vol = data[name]['volumes_ml']['eat_pat']
    fat_frac = data[name]['ratios_percent']['fat_fraction_in_shell']
    
    status = []
    if 50 <= eat_vol <= 200:
        status.append("Normal EAT+PAT volume")
    elif eat_vol > 200:
        status.append("Elevated EAT+PAT (>200ml)")
    else:
        status.append("Low EAT+PAT (<50ml)")
    
    if fat_frac < 30:
        status.append("Normal fat fraction")
    else:
        status.append("High fat fraction (>30%)")
    
    print(f"{name}: {', '.join(status)}")

print("\nFigure saved: /home/soya/ctrate_ws/outputs/eat_pat_comparison.png")
print("="*80)