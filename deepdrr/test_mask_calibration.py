#!/usr/bin/env python3
"""
マスク投影の体積保存キャリブレーション実装
DiffDRRの出力を後処理で正規化し、総体積が一致するように調整
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile


def project_mask_with_volume_calibration(
    mask_path: str,
    ct_path: str,
    voxel_spacing: tuple,
    pixel_spacing: tuple,
    sdd: float,
    height: int,
    width: int,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    DiffDRRでマスクを投影し、総体積が保存されるように後処理でキャリブレーション
    
    Args:
        mask_path: 3DマスクNIfTIファイルパス
        ct_path: CTファイルパス（比較用）
        voxel_spacing: ボクセルサイズ (x, y, z) mm
        pixel_spacing: ピクセルサイズ (delx, dely) mm
        sdd: Source-to-Detector Distance (mm)
        height: 出力画像の高さ
        width: 出力画像の幅
        rotations: 回転パラメータ
        translations: 平行移動パラメータ
        device: 計算デバイス
    
    Returns:
        結果の辞書（volume_map, thickness_map, stats）
    """
    
    print("="*60)
    print("Volume-Preserving Mask Projection with Calibration")
    print("="*60)
    
    # ========================================
    # ステップ1: 元の3D体積を計算
    # ========================================
    print("\nStep 1: Calculate true 3D volume")
    print("-"*40)
    
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata().astype(np.float32)
    
    # バイナリマスクとして処理
    mask_binary = (mask_data > 0).astype(np.float32)
    
    voxel_volume = np.prod(voxel_spacing)
    true_3d_volume = np.sum(mask_binary) * voxel_volume
    
    print(f"Mask shape: {mask_data.shape}")
    print(f"Non-zero voxels: {np.sum(mask_binary):.0f}")
    print(f"Voxel size: {voxel_spacing} mm")
    print(f"Voxel volume: {voxel_volume:.6f} mm³")
    print(f"True 3D volume: {true_3d_volume:.2f} mm³")
    
    # ========================================
    # ステップ2: DiffDRRで相対的な厚みマップを生成
    # ========================================
    print("\nStep 2: Generate relative thickness map with DiffDRR")
    print("-"*40)
    
    # マスクを一時ファイルとして保存（DiffDRRのread用）
    temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    mask_nifti_binary = nib.Nifti1Image(mask_binary, mask_nifti.affine)
    nib.save(mask_nifti_binary, temp_file.name)
    
    # DiffDRRで読み込み
    mask_subject = read(temp_file.name)
    
    # DRRモジュールを作成
    drr = DRR(
        mask_subject,
        sdd=sdd,
        height=height,
        width=width,
        delx=pixel_spacing[0],
        dely=pixel_spacing[1],
    ).to(device)
    
    # マスクを投影（透過率として）
    transmittance = drr(
        rotations,
        translations,
        parameterization="euler_angles",
        convention="ZXY"
    )
    
    # 透過率から相対的な厚みを計算
    # I = exp(-∫μ dx) → -log(I) = ∫μ dx = thickness (when μ=1)
    relative_thickness = -torch.log(torch.clamp(transmittance, min=1e-9))
    relative_thickness_np = relative_thickness.squeeze().cpu().numpy()
    
    print(f"DiffDRR output range: [{transmittance.min():.6f}, {transmittance.max():.6f}]")
    print(f"Relative thickness range: [{relative_thickness_np.min():.3f}, {relative_thickness_np.max():.3f}]")
    
    # ========================================
    # ステップ3: キャリブレーション係数を計算・適用
    # ========================================
    print("\nStep 3: Calculate and apply calibration factor")
    print("-"*40)
    
    pixel_area = pixel_spacing[0] * pixel_spacing[1]
    print(f"Pixel area: {pixel_area:.3f} mm²")
    
    # 投影された相対的な体積（未調整）
    uncalibrated_volume = np.sum(relative_thickness_np) * pixel_area
    print(f"Uncalibrated projected volume: {uncalibrated_volume:.2f} mm³")
    
    # キャリブレーション係数
    if uncalibrated_volume > 1e-6:
        calibration_factor = true_3d_volume / uncalibrated_volume
    else:
        calibration_factor = 0.0
        print("Warning: Uncalibrated volume is too small!")
    
    print(f"Calibration factor: {calibration_factor:.6f}")
    
    # キャリブレーションを適用
    calibrated_thickness = relative_thickness_np * calibration_factor
    volume_map = calibrated_thickness * pixel_area
    
    # ========================================
    # ステップ4: 検証
    # ========================================
    print("\nStep 4: Verification")
    print("-"*40)
    
    calibrated_volume = np.sum(volume_map)
    relative_error = abs(true_3d_volume - calibrated_volume) / true_3d_volume * 100
    
    print(f"Calibrated projected volume: {calibrated_volume:.2f} mm³")
    print(f"Volume difference: {abs(true_3d_volume - calibrated_volume):.2f} mm³")
    print(f"Relative error: {relative_error:.4f}%")
    
    # 統計情報
    stats = {
        'true_3d_volume': true_3d_volume,
        'uncalibrated_volume': uncalibrated_volume,
        'calibrated_volume': calibrated_volume,
        'calibration_factor': calibration_factor,
        'relative_error': relative_error,
        'max_thickness': calibrated_thickness.max(),
        'mean_thickness': calibrated_thickness[calibrated_thickness > 0.1].mean() if np.any(calibrated_thickness > 0.1) else 0,
        'non_zero_pixels': np.sum(calibrated_thickness > 0.1)
    }
    
    # 一時ファイル削除
    import os
    os.unlink(temp_file.name)
    
    return {
        'volume_map': volume_map,
        'thickness_map': calibrated_thickness,
        'relative_thickness': relative_thickness_np,
        'stats': stats
    }


def visualize_results(ct_path, mask_results, output_dir):
    """
    結果の可視化
    """
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)
    
    # CTのDRRも生成（比較用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ct_subject = read(ct_path)
    
    drr = DRR(
        ct_subject,
        sdd=1800.0,
        height=512,
        width=512,
        delx=0.7,
        dely=0.7,
    ).to(device)
    
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    ct_drr = drr(rotations, translations,
                 parameterization="euler_angles",
                 convention="ZXY")
    
    ct_drr_np = ct_drr.squeeze().cpu().numpy()
    ct_log = -np.log(np.maximum(ct_drr_np, 1e-6))
    ct_log_norm = (ct_log - ct_log.min()) / (ct_log.max() - ct_log.min())
    
    # 図の作成
    fig = plt.figure(figsize=(20, 12))
    
    # 1. CT DRR
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(ct_log_norm, cmap='gray')
    ax1.set_title("CT DRR", fontsize=12)
    ax1.axis('off')
    
    # 2. 相対的な厚み（キャリブレーション前）
    ax2 = plt.subplot(2, 4, 2)
    im = ax2.imshow(mask_results['relative_thickness'], cmap='hot', vmin=0)
    ax2.set_title(f"Relative Thickness\n(Before Calibration)", fontsize=12)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    # 3. キャリブレーション後の厚み
    ax3 = plt.subplot(2, 4, 3)
    im = ax3.imshow(mask_results['thickness_map'], cmap='hot', vmin=0)
    ax3.set_title(f"Calibrated Thickness\nMax: {mask_results['stats']['max_thickness']:.1f} mm", fontsize=12)
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    
    # 4. 体積マップ
    ax4 = plt.subplot(2, 4, 4)
    im = ax4.imshow(mask_results['volume_map'], cmap='viridis', vmin=0)
    ax4.set_title(f"Volume Map\nTotal: {mask_results['stats']['calibrated_volume']:.1f} mm³", fontsize=12)
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # 5. オーバーレイ
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(ct_log_norm, cmap='gray')
    mask_overlay = np.ma.masked_where(mask_results['thickness_map'] < 0.1, mask_results['thickness_map'])
    ax5.imshow(mask_overlay, cmap='Reds', alpha=0.6)
    ax5.set_title("CT + Mask Overlay", fontsize=12)
    ax5.axis('off')
    
    # 6. 統計情報
    ax6 = plt.subplot(2, 4, 6)
    ax6.axis('off')
    stats_text = f"""
Volume Preservation Analysis
{'='*30}
Original 3D Volume:
  {mask_results['stats']['true_3d_volume']:.2f} mm³

Before Calibration:
  {mask_results['stats']['uncalibrated_volume']:.2f} mm³

After Calibration:
  {mask_results['stats']['calibrated_volume']:.2f} mm³
  
Calibration Factor:
  {mask_results['stats']['calibration_factor']:.6f}

Final Error:
  {mask_results['stats']['relative_error']:.4f}%

Thickness Statistics:
  Max: {mask_results['stats']['max_thickness']:.2f} mm
  Mean: {mask_results['stats']['mean_thickness']:.2f} mm
  Non-zero pixels: {mask_results['stats']['non_zero_pixels']:.0f}
"""
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    # 7. ヒストグラム（厚み分布）
    ax7 = plt.subplot(2, 4, 7)
    thickness_positive = mask_results['thickness_map'][mask_results['thickness_map'] > 0.1]
    if len(thickness_positive) > 0:
        ax7.hist(thickness_positive.flatten(), bins=50, edgecolor='black', alpha=0.7)
        ax7.set_xlabel("Thickness (mm)")
        ax7.set_ylabel("Pixel count")
        ax7.set_title("Calibrated Thickness Distribution")
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, "No positive thickness values", 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title("Thickness Distribution")
    
    # 8. キャリブレーション比較
    ax8 = plt.subplot(2, 4, 8)
    x = np.arange(2)
    volumes = [mask_results['stats']['true_3d_volume'], 
               mask_results['stats']['uncalibrated_volume'],
               mask_results['stats']['calibrated_volume']]
    labels = ['3D Original', 'Uncalibrated', 'Calibrated']
    colors = ['blue', 'orange', 'green']
    
    bars = ax8.bar(np.arange(3), volumes, color=colors, alpha=0.7, edgecolor='black')
    ax8.set_ylabel('Volume (mm³)')
    ax8.set_title('Volume Comparison')
    ax8.set_xticks(np.arange(3))
    ax8.set_xticklabels(labels, rotation=45, ha='right')
    ax8.grid(axis='y', alpha=0.3)
    
    # バーの上に値を表示
    for bar, volume in zip(bars, volumes):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{volume:.0f}',
                ha='center', va='bottom')
    
    # 目標ラインを追加
    ax8.axhline(y=mask_results['stats']['true_3d_volume'], 
                color='red', linestyle='--', alpha=0.5, label='Target')
    ax8.legend()
    
    plt.suptitle("Volume-Preserving Mask Projection with Calibration", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_dir}/calibration_analysis.png")


def main():
    """
    メイン実行関数
    """
    # パス設定
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = Path("./drr_calibration_output")
    output_dir.mkdir(exist_ok=True)
    
    # パラメータ設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ボクセルサイズ（NIfTIヘッダーから取得）
    mask_nifti = nib.load(mask_path)
    voxel_spacing = mask_nifti.header.get_zooms()[:3]
    
    # 投影実行
    results = project_mask_with_volume_calibration(
        mask_path=mask_path,
        ct_path=ct_path,
        voxel_spacing=voxel_spacing,
        pixel_spacing=(0.7, 0.7),
        sdd=1800.0,
        height=512,
        width=512,
        rotations=torch.tensor([[0.0, 0.0, 0.0]], device=device),
        translations=torch.tensor([[0.0, 850.0, 0.0]], device=device),
        device=device,
    )
    
    # 可視化
    visualize_results(ct_path, results, output_dir)
    
    # 結果の保存
    np.save(output_dir / "volume_map.npy", results['volume_map'])
    np.save(output_dir / "thickness_map.npy", results['thickness_map'])
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"✓ Original 3D Volume: {results['stats']['true_3d_volume']:.2f} mm³")
    print(f"✓ Calibrated 2D Volume: {results['stats']['calibrated_volume']:.2f} mm³")
    print(f"✓ Relative Error: {results['stats']['relative_error']:.4f}%")
    print(f"✓ Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()