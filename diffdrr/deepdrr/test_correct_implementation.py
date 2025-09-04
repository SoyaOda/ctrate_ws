#!/usr/bin/env python3
"""
正しいDiffDRR実装によるマスク厚み投影
Beer-Lambert則なしの純粋な線積分を取得
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert
import matplotlib.pyplot as plt
from pathlib import Path

def test_correct_mask_projection():
    """
    正しいAPI（DRR.render）を使った厚み投影の実装
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = Path("/home/soya/ctrate_ws/deepdrr/deepdrr/correct_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) CTとマスクを同一Subjectに読む
    print("=== Step 1: Loading Data ===")
    subject = read(
        volume=ct_path,
        labelmap=mask_path,   # マスクはlabelmapとして
        orientation="PA",     # PAビューの幾何に合わせて
        center_volume=True,
    )
    print(f"Subject loaded: volume shape {subject.volume.shape}, mask shape {subject.mask.shape}")
    
    # マスクデータの確認
    mask_check = subject.mask.data.float()
    print(f"Mask data range: {mask_check.min():.3f} - {mask_check.max():.3f}")
    print(f"Non-zero voxels in mask: {(mask_check > 0).sum().item()}")
    
    # 2) DRR（Siddon）を "線積分" 設定で作る
    print("\n=== Step 2: Initializing DRR with Correct Settings ===")
    drr = DRR(
        subject,
        sdd=1800.0, 
        height=512, 
        width=512,
        delx=0.7,
        renderer="siddon",
        mode="nearest",     # 補間でラベルが溶けるのを防ぐ
        reducefn="sum",     # 線積分（合算）
    ).to(device)
    print("DRR initialized with Siddon renderer (mode='nearest', reducefn='sum')")
    
    # 3) ポーズ（PA view）
    print("\n=== Step 3: Setting Pose ===")
    rot = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    tra = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    pose = convert(rot, tra, parameterization="euler_angles", convention="ZXY")
    print(f"Pose set: rotation {rot}, translation {tra}")
    
    # 4) マスクをdensityとしてレンダ（Beer–Lambertはかからない）
    print("\n=== Step 4: Rendering Mask Thickness ===")
    
    # マスクデータを準備（0/1のバイナリ）
    mask_density = subject.mask.data.float().squeeze().to(device)
    print(f"Mask density shape: {mask_density.shape}")
    print(f"Mask density range: {mask_density.min():.3f} - {mask_density.max():.3f}")
    
    # DRR.renderを使う（これが正しい方法）
    src, tgt = drr.detector(pose)  # ワールド座標のsource/targetを取得
    print(f"Source shape: {src.shape}, Target shape: {tgt.shape}")
    
    thick_vec = drr.render(mask_density, src, tgt)  # 形状: (B, 1, H*W)相当
    print(f"Thick_vec shape: {thick_vec.shape}")
    
    thickness = drr.reshape_transform(thick_vec, batch_size=len(pose))  # (B,1,H,W)に整形
    print(f"Thickness shape after reshape: {thickness.shape}")
    
    # numpy配列に変換
    thickness_np = thickness.squeeze().detach().cpu().numpy()
    print(f"Final thickness shape: {thickness_np.shape}")
    print(f"Thickness range: {thickness_np.min():.6f} - {thickness_np.max():.6f}")
    print(f"Non-zero pixels: {(thickness_np > 1e-6).sum()}")
    
    # 5) 3D体積の計算
    print("\n=== Step 5: Volume Calculation ===")
    nii = nib.load(mask_path)
    voxel_dims = nii.header.get_zooms()[:3]  # (dx,dy,dz) [mm]
    voxel_vol = np.prod(voxel_dims)
    mask_data = nii.get_fdata()
    true_vol_3d = float((mask_data > 0).sum()) * voxel_vol
    
    print(f"Voxel dimensions: {voxel_dims} mm")
    print(f"Voxel volume: {voxel_vol:.6f} mm³")
    print(f"True 3D volume: {true_vol_3d:.2f} mm³")
    
    # 2D投影の体積
    pixel_area = 0.7 * 0.7  # mm²
    vol_2d_raw = thickness_np.sum() * pixel_area
    
    print(f"\n2D Volume (raw): {vol_2d_raw:.2f} mm³")
    print(f"Ratio (2D/3D): {vol_2d_raw / true_vol_3d:.6f}")
    
    # 6) 単一スカラ係数による体積キャリブレーション
    print("\n=== Step 6: Volume Calibration ===")
    if vol_2d_raw > 1e-6:
        k = true_vol_3d / vol_2d_raw
        thickness_calibrated = thickness_np * k
        vol_2d_calibrated = thickness_calibrated.sum() * pixel_area
        
        print(f"Calibration factor k: {k:.6f}")
        print(f"2D Volume (calibrated): {vol_2d_calibrated:.2f} mm³")
        print(f"Volume error: {abs(vol_2d_calibrated - true_vol_3d) / true_vol_3d * 100:.6f}%")
        
        # 厚み分布の統計
        thick_nonzero = thickness_calibrated[thickness_calibrated > 1e-6]
        if len(thick_nonzero) > 0:
            print(f"\nCalibrated thickness statistics:")
            print(f"  Range: {thick_nonzero.min():.3f} - {thick_nonzero.max():.3f} mm")
            print(f"  Mean: {thick_nonzero.mean():.3f} mm")
            print(f"  Std: {thick_nonzero.std():.3f} mm")
            print(f"  Unique values: {len(np.unique(thick_nonzero))}")
    else:
        print("Warning: No thickness detected!")
        thickness_calibrated = thickness_np
        vol_2d_calibrated = 0
        k = 0
    
    # 7) Z方向の真の厚み（比較用）
    print("\n=== Step 7: True Z-thickness Analysis ===")
    thickness_z_true = np.sum(mask_data > 0, axis=2) * voxel_dims[2]
    thickness_z_nonzero = thickness_z_true[thickness_z_true > 0]
    
    if len(thickness_z_nonzero) > 0:
        print(f"True Z-direction thickness:")
        print(f"  Range: {thickness_z_nonzero.min():.2f} - {thickness_z_nonzero.max():.2f} mm")
        print(f"  Mean: {thickness_z_nonzero.mean():.2f} mm")
        print(f"  Std: {thickness_z_nonzero.std():.2f} mm")
        print(f"  Unique values: {len(np.unique(thickness_z_nonzero))}")
    
    # 8) 可視化
    print("\n=== Step 8: Visualization ===")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # マスクの中央スライス
    mid_slice = mask_data.shape[2] // 2
    axes[0, 0].imshow(mask_data[:, :, mid_slice].T, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'3D Mask (slice {mid_slice})')
    axes[0, 0].axis('off')
    
    # Z方向の真の厚み
    im1 = axes[0, 1].imshow(thickness_z_true.T, cmap='hot', origin='lower')
    axes[0, 1].set_title(f'True Z-thickness\nMax: {thickness_z_true.max():.1f} mm')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], label='mm')
    
    # 生の厚みマップ
    im2 = axes[0, 2].imshow(thickness_np, cmap='hot')
    axes[0, 2].set_title(f'Raw Thickness (DRR)\nSum: {vol_2d_raw:.0f} mm³')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], label='Value')
    
    # キャリブレート後
    if vol_2d_calibrated > 0:
        im3 = axes[0, 3].imshow(thickness_calibrated, cmap='hot')
        axes[0, 3].set_title(f'Calibrated Thickness\nSum: {vol_2d_calibrated:.0f} mm³')
        axes[0, 3].axis('off')
        plt.colorbar(im3, ax=axes[0, 3], label='mm')
    else:
        axes[0, 3].text(0.5, 0.5, 'No calibration\n(no thickness detected)', 
                       ha='center', va='center', transform=axes[0, 3].transAxes)
        axes[0, 3].axis('off')
    
    # 真の厚みのヒストグラム
    if len(thickness_z_nonzero) > 0:
        axes[1, 0].hist(thickness_z_nonzero, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_title('True Z-thickness Distribution')
        axes[1, 0].set_xlabel('Thickness (mm)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
    
    # DRR厚みのヒストグラム（生）
    if np.any(thickness_np > 1e-6):
        axes[1, 1].hist(thickness_np[thickness_np > 1e-6], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].set_title('Raw DRR Thickness Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
    
    # キャリブレート後のヒストグラム
    if vol_2d_calibrated > 0 and np.any(thickness_calibrated > 1e-6):
        axes[1, 2].hist(thickness_calibrated[thickness_calibrated > 1e-6], bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 2].set_title('Calibrated Thickness Distribution')
        axes[1, 2].set_xlabel('Thickness (mm)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].axis('off')
    
    # 体積比較
    volumes = [true_vol_3d, vol_2d_raw, vol_2d_calibrated]
    labels = ['3D True', '2D Raw', '2D Calibrated']
    colors = ['green', 'blue', 'red']
    bars = axes[1, 3].bar(labels, volumes, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 3].set_title('Volume Comparison')
    axes[1, 3].set_ylabel('Volume (mm³)')
    axes[1, 3].grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for bar, vol in zip(bars, volumes):
        if vol > 0:
            axes[1, 3].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{vol:.0f}', ha='center', va='bottom')
    
    # エラー率を表示
    if vol_2d_calibrated > 0:
        axes[1, 3].text(0.95, 0.95, f'Error: {abs(vol_2d_calibrated - true_vol_3d) / true_vol_3d * 100:.4f}%', 
                       transform=axes[1, 3].transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / "correct_thickness_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    
    # NumPy配列として保存
    np.save(output_dir / "thickness_raw.npy", thickness_np)
    np.save(output_dir / "thickness_calibrated.npy", thickness_calibrated)
    np.save(output_dir / "calibration_factor.npy", np.array([k]))
    
    # 結果のサマリーを保存
    with open(output_dir / "results_summary.txt", "w") as f:
        f.write("=== Mask Thickness Projection Results ===\n\n")
        f.write(f"True 3D Volume: {true_vol_3d:.2f} mm³\n")
        f.write(f"2D Volume (raw): {vol_2d_raw:.2f} mm³\n")
        f.write(f"2D Volume (calibrated): {vol_2d_calibrated:.2f} mm³\n")
        f.write(f"Calibration factor: {k:.6f}\n")
        f.write(f"Volume error: {abs(vol_2d_calibrated - true_vol_3d) / true_vol_3d * 100:.6f}%\n")
        
        if len(thick_nonzero) > 0:
            f.write(f"\nCalibrated thickness statistics:\n")
            f.write(f"  Range: {thick_nonzero.min():.3f} - {thick_nonzero.max():.3f} mm\n")
            f.write(f"  Mean: {thick_nonzero.mean():.3f} mm\n")
            f.write(f"  Std: {thick_nonzero.std():.3f} mm\n")
    
    plt.show()
    
    return thickness_calibrated, true_vol_3d

if __name__ == "__main__":
    thickness, volume = test_correct_mask_projection()
    print("\n=== Complete ===")