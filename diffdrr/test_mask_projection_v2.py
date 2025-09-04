#!/usr/bin/env python3
"""
CTとマスクの2D投影テスト Version 2
マスクの投影を物理的な減衰ではなく、純粋な幾何学的投影として扱う
"""

import matplotlib.pyplot as plt
import torch
from diffdrr.drr import DRR
from diffdrr.data import read
from pathlib import Path
import numpy as np
import nibabel as nib
import tempfile


def compute_mask_projection_custom(mask_volume, camera_params):
    """
    カスタム投影関数：マスクのレイキャスティングを行い、
    各レイが通過するマスクボクセルの合計厚みを計算
    """
    # TODO: より正確なレイキャスティング実装
    pass


def generate_ct_and_mask_drrs_v2():
    """
    改良版：マスク投影の処理を修正
    """
    # Paths
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = Path("./drr_mask_test_v2_output")
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("CT and Mask Projection Test V2")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========================================
    # 1. CTのDRR生成
    # ========================================
    print("\n" + "="*40)
    print("1. CT DRR Generation")
    print("="*40)
    
    ct_subject = read(ct_path)
    print(f"CT volume shape: {ct_subject.volume.data.shape}")
    
    drr = DRR(
        ct_subject,
        sdd=1800.0,
        height=512,
        width=512,
        delx=0.7,
        dely=0.7,
    ).to(device)
    
    # カメラパラメータ
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    ct_drr = drr(rotations, translations,
                 parameterization="euler_angles",
                 convention="ZXY")
    
    ct_drr_np = ct_drr.squeeze().cpu().numpy()
    ct_log = -np.log(np.maximum(ct_drr_np, 1e-6))
    ct_log_norm = (ct_log - ct_log.min()) / (ct_log.max() - ct_log.min())
    
    print(f"CT DRR generated: range [{ct_drr_np.min():.3f}, {ct_drr_np.max():.3f}]")
    
    # ========================================
    # 2. マスクの処理（異なるアプローチ）
    # ========================================
    print("\n" + "="*40)
    print("2. Mask Processing (Alternative Approach)")
    print("="*40)
    
    # マスクデータを読み込み
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    voxel_dims = mask_nifti.header.get_zooms()[:3]
    voxel_volume = np.prod(voxel_dims)
    
    print(f"Mask shape: {mask_data.shape}")
    print(f"Voxel dimensions: {voxel_dims} mm")
    print(f"Non-zero voxels: {np.sum(mask_data > 0)}")
    
    # 3Dマスクの総体積
    original_volume_3d = np.sum(mask_data > 0) * voxel_volume
    print(f"Original 3D volume: {original_volume_3d:.2f} mm³")
    
    # マスクを固定値（例：1000）でスケール
    # DiffDRRが物理的な減衰として処理しないように
    mask_scaled = mask_data.astype(np.float32) * 1000.0
    
    # 一時ファイルに保存
    temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    mask_scaled_nifti = nib.Nifti1Image(mask_scaled, mask_nifti.affine)
    nib.save(mask_scaled_nifti, temp_file.name)
    
    # DiffDRRで読み込み
    mask_subject = read(temp_file.name)
    
    # マスク用のDRRモジュール
    mask_drr = DRR(
        mask_subject,
        sdd=1800.0,
        height=512,
        width=512,
        delx=0.7,
        dely=0.7,
    ).to(device)
    
    # マスクを投影
    mask_projection = mask_drr(rotations, translations,
                               parameterization="euler_angles",
                               convention="ZXY")
    
    mask_proj_np = mask_projection.squeeze().cpu().numpy()
    
    # スケーリングを元に戻す（厚みに変換）
    # DiffDRRの出力は減衰後の強度なので、対数変換を逆算
    mask_thickness_raw = -np.log(np.maximum(mask_proj_np, 1e-6))
    
    # 1000でスケールしたので、その分を調整
    mask_thickness = mask_thickness_raw / 1000.0 * np.mean(voxel_dims[:2])
    
    print(f"\nMask projection range: [{mask_proj_np.min():.3f}, {mask_proj_np.max():.3f}]")
    print(f"Thickness map range: [{mask_thickness.min():.3f}, {mask_thickness.max():.3f}] mm")
    
    # ========================================
    # 3. 別の方法：単純な最大値投影
    # ========================================
    print("\n" + "="*40)
    print("3. Maximum Intensity Projection (MIP) Test")
    print("="*40)
    
    # Z軸に沿った最大値投影（簡単な検証用）
    mip_z = np.max(mask_data, axis=2)
    mip_volume = np.sum(mip_z > 0) * voxel_dims[0] * voxel_dims[1] * mask_data.shape[2] * voxel_dims[2]
    print(f"MIP non-zero pixels: {np.sum(mip_z > 0)}")
    
    # ========================================
    # 4. 可視化
    # ========================================
    print("\n" + "="*40)
    print("4. Creating Visualizations")
    print("="*40)
    
    fig = plt.figure(figsize=(18, 12))
    
    # CT DRR
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(ct_log_norm, cmap='gray')
    ax1.set_title("CT DRR")
    ax1.axis('off')
    
    # マスク厚み（DiffDRR）
    ax2 = plt.subplot(2, 4, 2)
    im = ax2.imshow(mask_thickness, cmap='hot', vmin=0)
    ax2.set_title(f"Mask Thickness\nMax: {mask_thickness.max():.1f} mm")
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    # オーバーレイ
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(ct_log_norm, cmap='gray')
    mask_overlay = np.ma.masked_where(mask_thickness < 0.1, mask_thickness)
    ax3.imshow(mask_overlay, cmap='Reds', alpha=0.6)
    ax3.set_title("Overlay")
    ax3.axis('off')
    
    # MIP比較
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(mip_z.T, cmap='hot', origin='lower')
    ax4.set_title(f"Z-axis MIP\n(for comparison)")
    ax4.axis('off')
    
    # 体積解析
    pixel_area = 0.7 * 0.7  # mm²
    volume_map = mask_thickness * pixel_area
    projected_volume = np.sum(volume_map)
    
    # 統計表示
    ax5 = plt.subplot(2, 4, 5)
    ax5.axis('off')
    stats1_text = f"""
    Volume Analysis:
    ━━━━━━━━━━━━━━━━━━
    Original 3D: {original_volume_3d:.1f} mm³
    Projected 2D: {projected_volume:.1f} mm³
    Difference: {abs(original_volume_3d - projected_volume):.1f} mm³
    Error: {abs(original_volume_3d - projected_volume) / original_volume_3d * 100:.1f}%
    
    Thickness Stats:
    ━━━━━━━━━━━━━━━━━━
    Max: {mask_thickness.max():.2f} mm
    Mean: {mask_thickness[mask_thickness > 0.1].mean():.2f} mm
    Non-zero pixels: {np.sum(mask_thickness > 0.1)}
    """
    ax5.text(0.1, 0.9, stats1_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # ヒストグラム
    ax6 = plt.subplot(2, 4, 6)
    ax6.hist(mask_thickness[mask_thickness > 0.1].flatten(), bins=50, edgecolor='black')
    ax6.set_xlabel("Thickness (mm)")
    ax6.set_ylabel("Pixel count")
    ax6.set_title("Thickness Distribution")
    ax6.grid(True, alpha=0.3)
    
    # 体積マップ
    ax7 = plt.subplot(2, 4, 7)
    im = ax7.imshow(volume_map, cmap='viridis', vmin=0)
    ax7.set_title(f"Volume Map\nTotal: {projected_volume:.1f} mm³")
    ax7.axis('off')
    plt.colorbar(im, ax=ax7, fraction=0.046)
    
    # 差分解析
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    debug_text = f"""
    Debug Info:
    ━━━━━━━━━━━━━━━━━━
    CT shape: {ct_subject.volume.data.shape}
    Mask shape: {mask_data.shape}
    Voxel size: {voxel_dims}
    
    DRR Params:
    ━━━━━━━━━━━━━━━━━━
    Resolution: 512×512
    Pixel spacing: 0.7×0.7 mm
    SDD: 1800 mm
    Translation Y: 850 mm
    
    Processing:
    ━━━━━━━━━━━━━━━━━━
    Mask scaling: ×1000
    Output range: [{mask_proj_np.min():.3f}, {mask_proj_np.max():.3f}]
    """
    ax8.text(0.1, 0.9, debug_text, transform=ax8.transAxes,
            fontsize=9, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle("CT and Mask Projection Analysis V2", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "analysis_v2.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/analysis_v2.png")
    
    # データを保存
    np.save(output_dir / "ct_drr.npy", ct_log_norm)
    np.save(output_dir / "mask_thickness.npy", mask_thickness)
    np.save(output_dir / "volume_map.npy", volume_map)
    
    # クリーンアップ
    import os
    os.unlink(temp_file.name)
    
    print("\n" + "="*60)
    print("✓ Analysis completed!")
    print(f"✓ Results saved in: {output_dir}")
    print("="*60)
    
    return {
        'ct_drr': ct_log_norm,
        'mask_thickness': mask_thickness,
        'volume_map': volume_map,
        'original_volume': original_volume_3d,
        'projected_volume': projected_volume
    }


if __name__ == "__main__":
    results = generate_ct_and_mask_drrs_v2()
    
    # 結果サマリー
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original 3D Volume: {results['original_volume']:.2f} mm³")
    print(f"Projected 2D Volume: {results['projected_volume']:.2f} mm³")
    print(f"Relative Error: {abs(results['original_volume'] - results['projected_volume']) / results['original_volume'] * 100:.2f}%")
    print("="*60)