#!/usr/bin/env python3
"""
gpt5pro1.mdの解決策を実装
DiffDRRを使ってBeer-Lambert則を回避し、純粋な厚み投影を実現
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import convert
import matplotlib.pyplot as plt
from pathlib import Path

def test_gpt5pro_solution():
    """
    gpt5pro1.mdの提案に従った実装
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = Path("/home/soya/ctrate_ws/deepdrr/deepdrr/gpt5pro_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) DiffDRRのSubjectを、CT + Labelmapとして正しく読み込む
    print("=== Step 1: Loading Data ===")
    subject = read(
        volume=ct_path,
        labelmap=mask_path,         # マスクはlabelmapとして渡す
        orientation="PA",           # 実際の撮影向きに合わせる
        center_volume=True,
    )
    print(f"Subject loaded: volume shape {subject.volume.shape}, mask shape {subject.mask.shape}")
    
    # 2) DRRをSiddon・線積分（sum）・Nearestで初期化（Beer-Lambertなし）
    print("\n=== Step 2: Initializing DRR ===")
    drr = DRR(
        subject,
        sdd=1800.0,
        height=512,
        width=512,
        delx=0.7,                   # dely 未指定なら delx が使われる
        renderer="siddon",
        # renderer_kwargs はドキュメントにある既存引数
        # reducefn='sum' で線積分、mode='nearest'でラベル混合を防ぐ
        mode="nearest", 
        reducefn="sum",
    ).to(device)
    print("DRR initialized with Siddon renderer")
    
    # 3) ポーズ（PAビューの例）
    print("\n=== Step 3: Setting Pose ===")
    rot = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    tra = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    pose = convert(rot, tra, parameterization="euler_angles", convention="ZXY")
    print(f"Pose set: rotation {rot}, translation {tra}")
    
    # 4) 低レベルAPIで "マスク厚み" を直接レンダ
    print("\n=== Step 4: Rendering Thickness ===")
    # subject["mask"].data は TorchIO LabelMap（0/1）。floatにして密度として使う。
    src, tgt = drr.detector(pose, calibration=None)
    src = drr.affine_inverse(src)
    tgt = drr.affine_inverse(tgt)
    
    mask_density = subject["mask"].data.float().to(device)   # (1,1,D,H,W)
    print(f"Mask density shape: {mask_density.shape}")
    print(f"Mask density range: {mask_density.min():.3f} - {mask_density.max():.3f}")
    print(f"Non-zero voxels in mask: {(mask_density > 0).sum().item()}")
    
    # レンダリング実行
    thickness = drr.renderer(mask_density, src, tgt)         # ここが「厚み（mm）」！
    thick_np = thickness.squeeze().detach().cpu().numpy()
    print(f"Thickness shape: {thick_np.shape}")
    print(f"Thickness range: {thick_np.min():.6f} - {thick_np.max():.6f}")
    print(f"Non-zero pixels: {(thick_np > 1e-6).sum()}")
    
    # 5) 体積（3D・真値）と 2D 合計の比較（この段階では未校正）
    print("\n=== Step 5: Volume Calculation ===")
    nii = nib.load(mask_path)
    voxel_dims = nii.header.get_zooms()[:3]  # (dx,dy,dz) [mm]
    voxel_vol = np.prod(voxel_dims)
    true_vol_3d = float((nii.get_fdata() > 0).sum()) * voxel_vol
    
    pixel_area = 0.7 * 0.7  # [mm^2]
    vol_2d_naive = thick_np.sum() * pixel_area
    
    print(f"3D Volume (true): {true_vol_3d:.2f} mm^3")
    print(f"2D Volume (raw): {vol_2d_naive:.2f} mm^3")
    print(f"Ratio (2D/3D): {vol_2d_naive / true_vol_3d:.6f}")
    
    # 6) 単一スカラ係数で体積を厳密一致（厚み分布は保ったまま）
    print("\n=== Step 6: Volume Calibration ===")
    if vol_2d_naive > 1e-6:
        k = true_vol_3d / vol_2d_naive
        thick_calibrated = thick_np * k
        vol_2d = thick_calibrated.sum() * pixel_area
        
        print(f"Calibration factor k: {k:.6f}")
        print(f"2D Volume (calibrated): {vol_2d:.2f} mm^3")
        print(f"Error: {abs(vol_2d - true_vol_3d) / true_vol_3d * 100:.4f}%")
        
        # 厚み分布の統計
        thick_nonzero = thick_calibrated[thick_calibrated > 1e-6]
        if len(thick_nonzero) > 0:
            print(f"\nCalibrated thickness statistics:")
            print(f"  Range: {thick_nonzero.min():.3f} - {thick_nonzero.max():.3f} mm")
            print(f"  Mean: {thick_nonzero.mean():.3f} mm")
            print(f"  Std: {thick_nonzero.std():.3f} mm")
            print(f"  Unique values: {len(np.unique(thick_nonzero))}")
    else:
        print("Warning: No thickness detected!")
        thick_calibrated = thick_np
        vol_2d = 0
    
    # 7) 可視化
    print("\n=== Step 7: Visualization ===")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # マスクの中央スライス
    mask_data = nii.get_fdata()
    mid_slice = mask_data.shape[2] // 2
    axes[0, 0].imshow(mask_data[:, :, mid_slice].T, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'3D Mask (slice {mid_slice})')
    axes[0, 0].axis('off')
    
    # 生の厚みマップ
    im1 = axes[0, 1].imshow(thick_np, cmap='hot')
    axes[0, 1].set_title(f'Raw Thickness\nSum: {vol_2d_naive:.0f} mm³')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # キャリブレート後
    im2 = axes[0, 2].imshow(thick_calibrated, cmap='hot')
    axes[0, 2].set_title(f'Calibrated Thickness\nSum: {vol_2d:.0f} mm³')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Z方向の真の厚み
    thickness_z_true = np.sum(mask_data > 0, axis=2) * voxel_dims[2]
    im3 = axes[1, 0].imshow(thickness_z_true.T, cmap='hot', origin='lower')
    axes[1, 0].set_title(f'True Z-thickness\nMax: {thickness_z_true.max():.1f} mm')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 厚みヒストグラム
    if np.any(thick_calibrated > 1e-6):
        axes[1, 1].hist(thick_calibrated[thick_calibrated > 1e-6], bins=50, alpha=0.7)
        axes[1, 1].set_title('Calibrated Thickness Histogram')
        axes[1, 1].set_xlabel('Thickness (mm)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 体積比較
    volumes = [true_vol_3d, vol_2d_naive, vol_2d if vol_2d_naive > 1e-6 else 0]
    labels = ['3D True', '2D Raw', '2D Calibrated']
    colors = ['green', 'blue', 'red']
    bars = axes[1, 2].bar(labels, volumes, color=colors, alpha=0.7)
    axes[1, 2].set_title('Volume Comparison')
    axes[1, 2].set_ylabel('Volume (mm³)')
    axes[1, 2].grid(True, alpha=0.3)
    
    for bar, vol in zip(bars, volumes):
        if vol > 0:
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{vol:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = output_dir / "gpt5pro_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    
    # データ保存
    np.save(output_dir / "thickness_raw.npy", thick_np)
    np.save(output_dir / "thickness_calibrated.npy", thick_calibrated)
    
    plt.show()
    
    return thick_calibrated, true_vol_3d

if __name__ == "__main__":
    thickness, volume = test_gpt5pro_solution()
    print("\n=== Complete ===")