#!/usr/bin/env python3
"""
CTとマスクの2D投影テスト
generate_drrs_working.pyの成功手法を使用して、
CTとEAT+PATマスクを同じカメラパラメータで投影し、可視化する
"""

import matplotlib.pyplot as plt
import torch
from diffdrr.drr import DRR
from diffdrr.data import read
from pathlib import Path
import numpy as np
import nibabel as nib
from matplotlib.patches import Rectangle


def load_mask_volume(mask_path: str) -> np.ndarray:
    """
    NIfTIマスクファイルを読み込み、numpy配列として返す
    """
    print(f"Loading mask from: {mask_path}")
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    
    # バイナリマスクとして処理（0以外を1に）
    mask_binary = (mask_data > 0).astype(np.float32)
    
    print(f"Mask shape: {mask_data.shape}")
    print(f"Mask range: [{mask_data.min():.1f}, {mask_data.max():.1f}]")
    print(f"Non-zero voxels: {np.sum(mask_binary):.0f}")
    
    # ボクセルサイズを取得して3D体積を計算
    voxel_dims = mask_nifti.header.get_zooms()[:3]
    voxel_volume = np.prod(voxel_dims)
    total_volume_3d = np.sum(mask_binary) * voxel_volume
    print(f"Voxel dimensions (mm): {voxel_dims}")
    print(f"Original 3D volume: {total_volume_3d:.2f} mm³")
    
    return mask_binary, voxel_dims


def generate_ct_and_mask_drrs():
    """
    CTとマスクを同じパラメータで投影し、定量的な体積情報を維持
    """
    # Paths
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = Path("./drr_mask_test_output")
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("CT and Mask Projection Test")
    print("="*60)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========================================
    # 1. CTボリュームの読み込みとDRR生成
    # ========================================
    print("\n" + "="*40)
    print("1. Loading CT Volume")
    print("="*40)
    
    # DiffDRRで直接読み込み（ToCanonicalなし）
    ct_subject = read(ct_path)
    print(f"CT volume shape: {ct_subject.volume.data.shape}")
    print(f"CT range: [{ct_subject.volume.data.min():.1f}, {ct_subject.volume.data.max():.1f}] HU")
    
    # DRRモジュールを初期化（CTとマスクで共通使用）
    drr = DRR(
        ct_subject,
        sdd=1800.0,    # 標準的な胸部X線距離
        height=512,    # 高解像度
        width=512,
        delx=0.7,      # ピクセル間隔 (mm)
        dely=0.7,
    ).to(device)
    
    print(f"\nDRR configuration:")
    print(f"  - Image size: 512x512")
    print(f"  - Pixel spacing: 0.7x0.7 mm")
    print(f"  - Pixel area: {0.7 * 0.7:.3f} mm²")
    
    # カメラポーズを設定（PA view）
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    # CT DRRを生成
    print("\nGenerating CT DRR...")
    ct_drr = drr(rotations, translations,
                 parameterization="euler_angles",
                 convention="ZXY")
    
    ct_drr_np = ct_drr.squeeze().cpu().numpy()
    print(f"CT DRR range: [{ct_drr_np.min():.4f}, {ct_drr_np.max():.4f}]")
    
    # ========================================
    # 2. マスクボリュームの読み込みと処理
    # ========================================
    print("\n" + "="*40)
    print("2. Loading Mask Volume")
    print("="*40)
    
    mask_data, voxel_dims = load_mask_volume(mask_path)
    
    # マスクデータを一時的なNIfTIファイルとして保存
    # （DiffDRRのread()で読み込むため）
    import tempfile
    temp_mask_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    
    # CTと同じアフィン行列を使用してマスクを保存
    ct_nifti = nib.load(ct_path)
    mask_nifti = nib.Nifti1Image(mask_data, ct_nifti.affine)
    nib.save(mask_nifti, temp_mask_file.name)
    
    # DiffDRRでマスクを読み込み
    print("\nLoading mask with DiffDRR...")
    mask_subject = read(temp_mask_file.name)
    
    # マスク用のDRRモジュールを作成（同じパラメータ）
    mask_drr = DRR(
        mask_subject,
        sdd=1800.0,
        height=512,
        width=512,
        delx=0.7,
        dely=0.7,
    ).to(device)
    
    # ========================================
    # 3. マスクのDRR生成（厚みマップ）
    # ========================================
    print("\n" + "="*40)
    print("3. Generating Mask DRR (Thickness Map)")
    print("="*40)
    
    # 同じカメラポーズでマスクを投影
    mask_thickness = mask_drr(rotations, translations,
                              parameterization="euler_angles",
                              convention="ZXY")
    
    mask_thickness_np = mask_thickness.squeeze().cpu().numpy()
    print(f"Thickness map range: [{mask_thickness_np.min():.4f}, {mask_thickness_np.max():.4f}] mm")
    
    # ========================================
    # 4. 定量的スケーリング（厚み→体積）
    # ========================================
    print("\n" + "="*40)
    print("4. Quantitative Scaling")
    print("="*40)
    
    # ピクセル面積を計算
    pixel_area = 0.7 * 0.7
    print(f"Pixel area: {pixel_area:.3f} mm²")
    
    # 体積マップに変換（各ピクセルの厚み × ピクセル面積）
    volume_map = mask_thickness_np * pixel_area
    
    # 体積の検証
    total_volume_2d = np.sum(volume_map)
    voxel_volume = np.prod(voxel_dims)
    total_volume_3d = np.sum(mask_data) * voxel_volume
    
    print(f"\nVolume Validation:")
    print(f"  - Original 3D volume: {total_volume_3d:.2f} mm³")
    print(f"  - Projected 2D volume sum: {total_volume_2d:.2f} mm³")
    print(f"  - Difference: {abs(total_volume_3d - total_volume_2d):.2f} mm³")
    print(f"  - Relative error: {abs(total_volume_3d - total_volume_2d) / total_volume_3d * 100:.2f}%")
    
    # ========================================
    # 5. 可視化
    # ========================================
    print("\n" + "="*40)
    print("5. Creating Visualizations")
    print("="*40)
    
    # 図1: CT DRRと処理版
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Raw CT DRR
    axes[0].imshow(ct_drr_np, cmap='gray')
    axes[0].set_title(f"CT DRR (Raw)\nRange: [{ct_drr_np.min():.3f}, {ct_drr_np.max():.3f}]")
    axes[0].axis('off')
    
    # Log transformed CT DRR
    ct_log = -np.log(np.maximum(ct_drr_np, 1e-6))
    ct_log_norm = (ct_log - ct_log.min()) / (ct_log.max() - ct_log.min())
    axes[1].imshow(ct_log_norm, cmap='gray')
    axes[1].set_title("CT DRR (Log transformed)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "ct_drr_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/ct_drr_comparison.png")
    
    # 図2: マスク投影結果
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 厚みマップ
    im1 = axes[0, 0].imshow(mask_thickness_np, cmap='hot', vmin=0)
    axes[0, 0].set_title(f"Thickness Map\nRange: [0, {mask_thickness_np.max():.1f}] mm")
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # 体積マップ
    im2 = axes[0, 1].imshow(volume_map, cmap='viridis', vmin=0)
    axes[0, 1].set_title(f"Volume Map\nTotal: {total_volume_2d:.1f} mm³")
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # オーバーレイ（CTとマスク）
    axes[1, 0].imshow(ct_log_norm, cmap='gray')
    mask_overlay = np.ma.masked_where(mask_thickness_np == 0, mask_thickness_np)
    axes[1, 0].imshow(mask_overlay, cmap='Reds', alpha=0.5)
    axes[1, 0].set_title("CT + Mask Overlay")
    axes[1, 0].axis('off')
    
    # ヒストグラム
    axes[1, 1].hist(volume_map[volume_map > 0].flatten(), bins=50, edgecolor='black')
    axes[1, 1].set_xlabel("Volume per pixel (mm³)")
    axes[1, 1].set_ylabel("Pixel count")
    axes[1, 1].set_title(f"Volume Distribution\nNon-zero pixels: {np.sum(volume_map > 0)}")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "mask_projection_analysis.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/mask_projection_analysis.png")
    
    # 図3: 統合比較図
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # CT DRR（大きく表示）
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.imshow(ct_log_norm, cmap='gray')
    ax1.set_title("CT DRR (Chest X-ray)", fontsize=14)
    ax1.axis('off')
    
    # マスク厚みマップ
    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.imshow(mask_thickness_np, cmap='hot')
    ax2.set_title("EAT+PAT Thickness", fontsize=10)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    # マスク体積マップ
    ax3 = fig.add_subplot(gs[1, 2])
    im = ax3.imshow(volume_map, cmap='viridis')
    ax3.set_title("Volume Map", fontsize=10)
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    
    # オーバーレイ
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.imshow(ct_log_norm, cmap='gray')
    ax4.imshow(mask_overlay, cmap='Reds', alpha=0.6)
    ax4.set_title("Overlay", fontsize=10)
    ax4.axis('off')
    
    # 統計情報
    ax5 = fig.add_subplot(gs[2, 1:3])
    ax5.axis('off')
    stats_text = f"""
    Projection Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • CT Volume Shape: {ct_subject.volume.data.shape}
    • DRR Resolution: 512×512 pixels
    • Pixel Spacing: 0.7×0.7 mm
    • Camera Distance: 1800.0 mm
    
    EAT+PAT Volume Analysis:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Original 3D Volume: {total_volume_3d:.1f} mm³
    • Projected 2D Sum: {total_volume_2d:.1f} mm³
    • Relative Error: {abs(total_volume_3d - total_volume_2d) / total_volume_3d * 100:.2f}%
    • Non-zero Pixels: {np.sum(volume_map > 0)}
    • Max Thickness: {mask_thickness_np.max():.1f} mm
    • Mean Thickness: {mask_thickness_np[mask_thickness_np > 0].mean():.2f} mm
    """
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
            fontsize=9, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle("CT and EAT+PAT Mask Projection Analysis", fontsize=16, y=0.98)
    plt.savefig(output_dir / "complete_analysis.png", dpi=200)
    plt.close()
    print(f"Saved: {output_dir}/complete_analysis.png")
    
    # データを保存
    np.save(output_dir / "ct_drr.npy", ct_log_norm)
    np.save(output_dir / "mask_thickness.npy", mask_thickness_np)
    np.save(output_dir / "mask_volume.npy", volume_map)
    print(f"\nNumpy arrays saved to {output_dir}")
    
    # 一時ファイルを削除
    import os
    os.unlink(temp_mask_file.name)
    
    print("\n" + "="*60)
    print("✓ CT and Mask projection completed successfully!")
    print(f"✓ All results saved in: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    generate_ct_and_mask_drrs()