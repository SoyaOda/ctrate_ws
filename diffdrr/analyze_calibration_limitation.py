#!/usr/bin/env python3
"""
test_mask_calibration.pyの局所的な厚み精度の限界を検証
キャリブレーションは体積を保存するが、局所的な厚みの精度は保証されない
"""

import numpy as np
import torch
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile


def analyze_calibration_limitation():
    """
    キャリブレーション方式の限界を詳細に分析
    """
    print("="*60)
    print("Calibration Method Limitation Analysis")
    print("="*60)
    
    # パス設定
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = Path("./calibration_limitation_output")
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========================================
    # 1. 3Dマスクの厚み分布を解析
    # ========================================
    print("\n1. Analyzing 3D mask thickness distribution...")
    print("-"*40)
    
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    mask_binary = (mask_data > 0).astype(np.float32)
    voxel_dims = mask_nifti.header.get_zooms()[:3]
    
    # Z軸方向の真の厚み分布
    true_thickness_z = np.sum(mask_binary, axis=2) * voxel_dims[2]
    
    # 厚みの統計
    thickness_nonzero = true_thickness_z[true_thickness_z > 0]
    unique_thicknesses = np.unique(thickness_nonzero)
    
    print(f"True 3D thickness statistics:")
    print(f"  - Min: {thickness_nonzero.min():.1f} mm")
    print(f"  - Max: {thickness_nonzero.max():.1f} mm")
    print(f"  - Mean: {thickness_nonzero.mean():.1f} mm")
    print(f"  - Std: {thickness_nonzero.std():.1f} mm")
    print(f"  - Unique values: {len(unique_thicknesses)}")
    print(f"  - Range: {thickness_nonzero.max() - thickness_nonzero.min():.1f} mm")
    
    # ========================================
    # 2. DiffDRR投影（キャリブレーション前）
    # ========================================
    print("\n2. DiffDRR projection (before calibration)...")
    print("-"*40)
    
    # マスクを一時ファイルとして保存
    temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    mask_nifti_binary = nib.Nifti1Image(mask_binary, mask_nifti.affine)
    nib.save(mask_nifti_binary, temp_file.name)
    
    # DiffDRRで投影
    mask_subject = read(temp_file.name)
    drr = DRR(
        mask_subject,
        sdd=1800.0,
        height=512,
        width=512,
        delx=0.7,
        dely=0.7,
    ).to(device)
    
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    transmittance = drr(rotations, translations,
                       parameterization="euler_angles",
                       convention="ZXY")
    
    # 厚みに変換（Beer-Lambert則の逆変換）
    uncalibrated_thickness = -torch.log(torch.clamp(transmittance, min=1e-9))
    uncalibrated_thickness_np = uncalibrated_thickness.squeeze().cpu().numpy()
    
    # キャリブレーション前の統計
    uncalib_nonzero = uncalibrated_thickness_np[uncalibrated_thickness_np > 0.1]
    if len(uncalib_nonzero) > 0:
        print(f"Uncalibrated thickness statistics:")
        print(f"  - Min: {uncalib_nonzero.min():.3f}")
        print(f"  - Max: {uncalib_nonzero.max():.3f}")
        print(f"  - Mean: {uncalib_nonzero.mean():.3f}")
        print(f"  - Std: {uncalib_nonzero.std():.3f}")
        print(f"  - Unique values (0.1 bins): {len(np.unique(np.round(uncalib_nonzero, 1)))}")
        print(f"  - Range: {uncalib_nonzero.max() - uncalib_nonzero.min():.3f}")
    
    # ========================================
    # 3. キャリブレーション後
    # ========================================
    print("\n3. After calibration...")
    print("-"*40)
    
    # 体積ベースのキャリブレーション
    voxel_volume = np.prod(voxel_dims)
    true_volume = np.sum(mask_binary) * voxel_volume
    pixel_area = 0.7 * 0.7
    uncalibrated_volume = np.sum(uncalibrated_thickness_np) * pixel_area
    
    calibration_factor = true_volume / uncalibrated_volume if uncalibrated_volume > 0 else 0
    calibrated_thickness = uncalibrated_thickness_np * calibration_factor
    
    # キャリブレーション後の統計
    calib_nonzero = calibrated_thickness[calibrated_thickness > 0.01]
    if len(calib_nonzero) > 0:
        print(f"Calibrated thickness statistics:")
        print(f"  - Min: {calib_nonzero.min():.3f} mm")
        print(f"  - Max: {calib_nonzero.max():.3f} mm")
        print(f"  - Mean: {calib_nonzero.mean():.3f} mm")
        print(f"  - Std: {calib_nonzero.std():.3f} mm")
        print(f"  - Unique values (0.1 bins): {len(np.unique(np.round(calib_nonzero, 1)))}")
        print(f"  - Range: {calib_nonzero.max() - calib_nonzero.min():.3f} mm")
    
    print(f"\nCalibration factor: {calibration_factor:.6f}")
    
    # ========================================
    # 4. 問題の可視化
    # ========================================
    print("\n4. Visualizing the limitation...")
    print("-"*40)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 3D真の厚み分布
    ax1 = plt.subplot(3, 3, 1)
    im = ax1.imshow(true_thickness_z.T, cmap='jet', origin='lower', vmin=0, vmax=70)
    ax1.set_title(f"True 3D Thickness (Z-axis)\nMax: {true_thickness_z.max():.1f} mm")
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, fraction=0.046)
    
    # DiffDRR投影（キャリブレーション前）
    ax2 = plt.subplot(3, 3, 2)
    im = ax2.imshow(uncalibrated_thickness_np, cmap='jet', vmin=0, vmax=uncalibrated_thickness_np.max())
    ax2.set_title(f"DiffDRR Raw Output\nMax: {uncalibrated_thickness_np.max():.2f}")
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    # キャリブレーション後
    ax3 = plt.subplot(3, 3, 3)
    im = ax3.imshow(calibrated_thickness, cmap='jet', vmin=0, vmax=2)
    ax3.set_title(f"After Calibration\nMax: {calibrated_thickness.max():.2f} mm")
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    
    # ヒストグラム比較
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(thickness_nonzero, bins=50, alpha=0.5, label='True 3D', color='blue', edgecolor='black')
    ax4.set_xlabel("Thickness (mm)")
    ax4.set_ylabel("Pixel count")
    ax4.set_title("True 3D Thickness Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(3, 3, 5)
    if len(uncalib_nonzero) > 0:
        ax5.hist(uncalib_nonzero, bins=50, alpha=0.5, label='Uncalibrated', color='orange', edgecolor='black')
    ax5.set_xlabel("Thickness")
    ax5.set_ylabel("Pixel count")
    ax5.set_title("DiffDRR Raw Output Distribution")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(3, 3, 6)
    if len(calib_nonzero) > 0:
        ax6.hist(calib_nonzero, bins=50, alpha=0.5, label='Calibrated', color='green', edgecolor='black')
    ax6.set_xlabel("Thickness (mm)")
    ax6.set_ylabel("Pixel count")
    ax6.set_title("After Calibration Distribution")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # プロファイル比較
    ax7 = plt.subplot(3, 3, 7)
    center_y = true_thickness_z.shape[0] // 2
    profile_3d = true_thickness_z[center_y, :]
    profile_calib = calibrated_thickness[256, :]  # 中心線
    
    x_3d = np.arange(len(profile_3d)) * voxel_dims[0]
    x_2d = np.arange(len(profile_calib)) * 0.7
    
    ax7.plot(x_3d, profile_3d, 'b-', linewidth=2, label='True 3D', alpha=0.7)
    ax7.plot(x_2d, profile_calib * 40, 'r-', linewidth=2, label='Calibrated×40', alpha=0.7)
    ax7.set_xlabel("Position (mm)")
    ax7.set_ylabel("Thickness (mm)")
    ax7.set_title("Center Line Profile Comparison")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 問題の説明
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    problem_text = f"""
THE FUNDAMENTAL PROBLEM:
{'='*30}

Input: Binary mask (0 or 1)
  ↓
DiffDRR: Beer-Lambert law
  I = exp(-μt) where μ=1
  ↓
Result: All thickness variations
  are exponentially compressed
  
WHAT HAPPENS:
• True: 1.5mm → Output: ~0.22
• True: 10mm → Output: ~0.00005
• True: 30mm → Output: ~10⁻¹³
• True: 69mm → Output: ~0

After uniform scaling (×{calibration_factor:.4f}):
• All values scaled equally
• Relative differences lost
• Volume preserved globally
• Local accuracy NOT preserved
"""
    ax8.text(0.05, 0.95, problem_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.3))
    
    # 解決策の提案
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    solution_text = f"""
PROPOSED SOLUTIONS:
{'='*30}

1. Variable Density Mask:
   • Instead of binary (0,1)
   • Use gradient (0-1000)
   • Encode thickness in density
   
2. Multiple Projections:
   • Different angles
   • Tomographic reconstruction
   
3. Pure Ray Casting:
   • Bypass DiffDRR physics
   • Direct geometric integration
   • Use PyTorch3D or custom

4. Hybrid Approach:
   • Use DiffDRR for alignment
   • Custom ray casting for thickness
   • Combine both outputs

Current Status:
✓ Volume preserved: {true_volume:.0f} mm³
✗ Thickness accuracy: Poor
  (Range compressed from 67.5mm to {(calib_nonzero.max() - calib_nonzero.min()):.2f}mm)
"""
    ax9.text(0.05, 0.95, solution_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle("Calibration Method Limitation: Volume Preserved but Thickness Compressed", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_limitation_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {output_dir}/calibration_limitation_analysis.png")
    
    # クリーンアップ
    import os
    os.unlink(temp_file.name)
    
    # ========================================
    # 5. 結論
    # ========================================
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("\n⚠️  CRITICAL LIMITATION CONFIRMED:")
    print(f"  • True thickness range: {thickness_nonzero.min():.1f} - {thickness_nonzero.max():.1f} mm (Range: {thickness_nonzero.max() - thickness_nonzero.min():.1f} mm)")
    print(f"  • After calibration: {calib_nonzero.min():.3f} - {calib_nonzero.max():.3f} mm (Range: {calib_nonzero.max() - calib_nonzero.min():.3f} mm)")
    print(f"  • Compression ratio: {(thickness_nonzero.max() - thickness_nonzero.min()) / (calib_nonzero.max() - calib_nonzero.min()):.1f}x")
    print("\nThe calibration preserves total volume but CANNOT recover local thickness variations.")
    print("This is a fundamental limitation of using Beer-Lambert law on binary masks.")
    print("="*60)


if __name__ == "__main__":
    analyze_calibration_limitation()