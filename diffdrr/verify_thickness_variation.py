#!/usr/bin/env python3
"""
マスク投影の厚み変化が正しく反映されているか検証
異なる角度からの投影と、Z軸に沿った厚みの理論値との比較
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile


def analyze_thickness_variation():
    """
    マスクの厚みの変化を詳細に解析
    """
    print("="*60)
    print("Thickness Variation Analysis")
    print("="*60)
    
    # パス設定
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = Path("./thickness_verification_output")
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========================================
    # 1. マスクデータの解析
    # ========================================
    print("\n1. Analyzing mask data...")
    print("-"*40)
    
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    mask_binary = (mask_data > 0).astype(np.float32)
    voxel_dims = mask_nifti.header.get_zooms()[:3]
    
    print(f"Mask shape: {mask_data.shape}")
    print(f"Voxel dimensions: {voxel_dims} mm")
    
    # Z軸方向の理論的な厚み（単純な積算）
    # 各XY位置でZ方向にマスクが何ボクセルあるか
    theoretical_thickness_z = np.sum(mask_binary, axis=2) * voxel_dims[2]
    print(f"\nTheoretical Z-thickness stats:")
    print(f"  Max: {theoretical_thickness_z.max():.2f} mm")
    print(f"  Mean (non-zero): {theoretical_thickness_z[theoretical_thickness_z > 0].mean():.2f} mm")
    print(f"  Min (non-zero): {theoretical_thickness_z[theoretical_thickness_z > 0].min():.2f} mm")
    
    # 厚みのヒストグラム
    thickness_values = theoretical_thickness_z[theoretical_thickness_z > 0].flatten()
    print(f"  Unique thickness values: {len(np.unique(thickness_values))}")
    
    # ========================================
    # 2. DiffDRRでの投影（複数角度）
    # ========================================
    print("\n2. DiffDRR projection analysis...")
    print("-"*40)
    
    # マスクを一時ファイルとして保存
    temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    mask_nifti_binary = nib.Nifti1Image(mask_binary, mask_nifti.affine)
    nib.save(mask_nifti_binary, temp_file.name)
    
    # DiffDRRで読み込み
    mask_subject = read(temp_file.name)
    
    # 異なる投影角度でテスト
    test_cases = [
        {
            'name': 'PA_view',
            'rotations': [0.0, 0.0, 0.0],
            'translations': [0.0, 850.0, 0.0],
            'description': 'Standard PA view'
        },
        {
            'name': 'Lateral_view',
            'rotations': [0.0, 90.0, 0.0],
            'translations': [850.0, 0.0, 0.0],
            'description': 'Lateral view (90° rotation)'
        },
        {
            'name': 'Oblique_view',
            'rotations': [30.0, 0.0, 0.0],
            'translations': [0.0, 850.0, 0.0],
            'description': 'Oblique view (30° rotation)'
        }
    ]
    
    results = {}
    
    for case in test_cases:
        print(f"\nProcessing {case['name']}...")
        
        # DRRモジュール作成
        drr = DRR(
            mask_subject,
            sdd=1800.0,
            height=512,
            width=512,
            delx=0.7,
            dely=0.7,
        ).to(device)
        
        # 投影
        rotations = torch.tensor([case['rotations']], device=device)
        translations = torch.tensor([case['translations']], device=device)
        
        transmittance = drr(
            rotations,
            translations,
            parameterization="euler_angles",
            convention="ZXY"
        )
        
        # 厚みに変換
        thickness = -torch.log(torch.clamp(transmittance, min=1e-9))
        thickness_np = thickness.squeeze().cpu().numpy()
        
        # キャリブレーション（体積保存のため）
        voxel_volume = np.prod(voxel_dims)
        true_volume = np.sum(mask_binary) * voxel_volume
        pixel_area = 0.7 * 0.7
        uncalibrated_volume = np.sum(thickness_np) * pixel_area
        
        if uncalibrated_volume > 0:
            calibration_factor = true_volume / uncalibrated_volume
            calibrated_thickness = thickness_np * calibration_factor
        else:
            calibrated_thickness = thickness_np
            calibration_factor = 0
        
        # 統計
        positive_thickness = calibrated_thickness[calibrated_thickness > 0.1]
        if len(positive_thickness) > 0:
            stats = {
                'max': positive_thickness.max(),
                'min': positive_thickness.min(),
                'mean': positive_thickness.mean(),
                'std': positive_thickness.std(),
                'unique_count': len(np.unique(np.round(positive_thickness, 1)))
            }
        else:
            stats = {'max': 0, 'min': 0, 'mean': 0, 'std': 0, 'unique_count': 0}
        
        results[case['name']] = {
            'thickness': calibrated_thickness,
            'stats': stats,
            'calibration_factor': calibration_factor,
            'description': case['description']
        }
        
        print(f"  Max thickness: {stats['max']:.2f} mm")
        print(f"  Min thickness: {stats['min']:.2f} mm")
        print(f"  Mean thickness: {stats['mean']:.2f} mm")
        print(f"  Std deviation: {stats['std']:.2f} mm")
        print(f"  Unique values (0.1mm bins): {stats['unique_count']}")
    
    # ========================================
    # 3. 詳細な可視化
    # ========================================
    print("\n3. Creating detailed visualizations...")
    print("-"*40)
    
    fig = plt.figure(figsize=(20, 14))
    
    # 理論的な厚み（Z軸積算）
    ax1 = plt.subplot(3, 4, 1)
    im = ax1.imshow(theoretical_thickness_z.T, cmap='jet', origin='lower', vmin=0)
    ax1.set_title(f"Theoretical Z-thickness\nMax: {theoretical_thickness_z.max():.1f} mm")
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, fraction=0.046)
    
    # 各投影の結果
    for idx, (name, result) in enumerate(results.items()):
        # 厚みマップ
        ax = plt.subplot(3, 4, idx + 2)
        im = ax.imshow(result['thickness'], cmap='jet', vmin=0)
        ax.set_title(f"{result['description']}\nMax: {result['stats']['max']:.1f} mm")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # ヒストグラム
        ax_hist = plt.subplot(3, 4, idx + 5)
        positive = result['thickness'][result['thickness'] > 0.1]
        if len(positive) > 0:
            ax_hist.hist(positive.flatten(), bins=30, edgecolor='black', alpha=0.7)
            ax_hist.set_xlabel("Thickness (mm)")
            ax_hist.set_ylabel("Pixel count")
            ax_hist.set_title(f"{name} Distribution")
            ax_hist.grid(True, alpha=0.3)
            
            # 統計情報を追加
            ax_hist.axvline(result['stats']['mean'], color='red', linestyle='--', label=f"Mean: {result['stats']['mean']:.2f}")
            ax_hist.axvline(result['stats']['max'], color='green', linestyle='--', label=f"Max: {result['stats']['max']:.2f}")
            ax_hist.legend(fontsize=8)
    
    # 厚み変化の断面プロット（PA viewの中心線）
    ax_profile = plt.subplot(3, 4, 9)
    pa_thickness = results['PA_view']['thickness']
    center_y = pa_thickness.shape[0] // 2
    profile = pa_thickness[center_y, :]
    
    x_positions = np.arange(len(profile)) * 0.7  # mm単位に変換
    ax_profile.plot(x_positions, profile, 'b-', linewidth=2)
    ax_profile.fill_between(x_positions, 0, profile, alpha=0.3)
    ax_profile.set_xlabel("X Position (mm)")
    ax_profile.set_ylabel("Thickness (mm)")
    ax_profile.set_title("PA View Center Line Profile")
    ax_profile.grid(True, alpha=0.3)
    ax_profile.set_ylim(bottom=0)
    
    # 厚み変化の2D比較
    ax_2d = plt.subplot(3, 4, 10)
    # PA viewの厚みマップから小領域を抽出
    roi = pa_thickness[200:300, 200:300]
    im = ax_2d.imshow(roi, cmap='jet', interpolation='nearest')
    ax_2d.set_title(f"PA View ROI (100x100)\nMin: {roi[roi>0].min():.2f}, Max: {roi.max():.2f} mm")
    plt.colorbar(im, ax=ax_2d, fraction=0.046)
    
    # 統計サマリー
    ax_summary = plt.subplot(3, 4, 11)
    ax_summary.axis('off')
    summary_text = "Thickness Variation Summary\n" + "="*30 + "\n\n"
    summary_text += f"Theoretical Z-thickness:\n"
    summary_text += f"  Range: {theoretical_thickness_z[theoretical_thickness_z>0].min():.1f} - {theoretical_thickness_z.max():.1f} mm\n"
    summary_text += f"  Unique values: {len(np.unique(theoretical_thickness_z[theoretical_thickness_z>0]))}\n\n"
    
    for name, result in results.items():
        summary_text += f"{name}:\n"
        summary_text += f"  Range: {result['stats']['min']:.2f} - {result['stats']['max']:.2f} mm\n"
        summary_text += f"  Mean ± Std: {result['stats']['mean']:.2f} ± {result['stats']['std']:.2f} mm\n"
        summary_text += f"  Unique values: {result['stats']['unique_count']}\n\n"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=10, verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # 差分解析
    ax_diff = plt.subplot(3, 4, 12)
    if 'PA_view' in results and 'Lateral_view' in results:
        pa = results['PA_view']['thickness']
        lateral = results['Lateral_view']['thickness']
        
        # サイズを合わせる（必要に応じて）
        min_size = min(pa.shape[0], lateral.shape[0]), min(pa.shape[1], lateral.shape[1])
        pa_crop = pa[:min_size[0], :min_size[1]]
        lateral_crop = lateral[:min_size[0], :min_size[1]]
        
        # 両方で非ゼロの部分のみ比較
        mask = (pa_crop > 0.1) & (lateral_crop > 0.1)
        if np.any(mask):
            ratio = np.zeros_like(pa_crop)
            ratio[mask] = pa_crop[mask] / lateral_crop[mask]
            im = ax_diff.imshow(ratio, cmap='RdBu_r', vmin=0.5, vmax=2.0)
            ax_diff.set_title("PA / Lateral Thickness Ratio")
            plt.colorbar(im, ax=ax_diff, fraction=0.046)
        else:
            ax_diff.text(0.5, 0.5, "No overlap", ha='center', va='center')
            ax_diff.set_title("PA / Lateral Ratio")
    ax_diff.axis('off')
    
    plt.suptitle("Thickness Variation Analysis", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "thickness_variation_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/thickness_variation_analysis.png")
    
    # クリーンアップ
    import os
    os.unlink(temp_file.name)
    
    # ========================================
    # 4. 結論
    # ========================================
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    if 'PA_view' in results:
        pa_stats = results['PA_view']['stats']
        print(f"\nPA View Thickness Analysis:")
        print(f"  - Range: {pa_stats['min']:.2f} - {pa_stats['max']:.2f} mm")
        print(f"  - Standard deviation: {pa_stats['std']:.2f} mm")
        print(f"  - Unique values: {pa_stats['unique_count']}")
        
        if pa_stats['std'] < 0.01:
            print("\n⚠️  WARNING: Very low standard deviation detected!")
            print("    The thickness appears to be nearly uniform.")
        elif pa_stats['std'] < 0.1:
            print("\n⚠️  CAUTION: Low thickness variation detected.")
            print("    The projection may not fully capture thickness variations.")
        else:
            print("\n✓  Good thickness variation detected.")
            print("    The projection captures spatial thickness variations.")
    
    print("\n" + "="*60)
    
    return results


if __name__ == "__main__":
    results = analyze_thickness_variation()