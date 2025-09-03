#!/usr/bin/env python3
"""
DRR生成のデバッグスクリプト
"""

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from diffdrr.data import load_example_ct
from diffdrr.drr import DRR
from diffdrr.pose import convert

def test_example_ct():
    """DiffDRRの例データでテスト"""
    print("=== Testing with DiffDRR example CT ===")
    
    # 例のCTを読み込み
    subject = load_example_ct()
    print(f"Example CT shape: {subject.volume.shape}")
    print(f"Example CT spacing: {subject.spacing}")
    
    # DRR生成
    device = "cuda" if torch.cuda.is_available() else "cpu"
    drr = DRR(subject, sdd=1800.0, height=256, delx=1.0).to(device)
    
    # 正面から投影
    rotations = torch.tensor([[0.0, 0.0, 0.0]]).to(device)
    translations = torch.tensor([[0.0, 0.0, 0.0]]).to(device)
    transform = convert(rotations, translations, 
                       parameterization="euler_angles",
                       convention="XYZ").to(device)
    
    drr_image = drr(transform)
    
    # 表示
    plt.figure(figsize=(6, 6))
    plt.imshow(drr_image.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Example CT - DRR")
    plt.axis('off')
    plt.savefig('debug_example_drr.png')
    print("Saved: debug_example_drr.png")
    
    return drr_image


def test_our_ct():
    """実際のCTデータでテスト"""
    print("\n=== Testing with our CT ===")
    
    from diffdrr.data import read
    import tempfile
    
    # CTデータを読み込み
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    
    # DiffDRRのread関数で直接読み込み
    print("Loading CT with DiffDRR read...")
    subject = read(ct_path, bone_attenuation_multiplier=1.0)
    
    print(f"Our CT volume shape: {subject.volume.shape}")
    print(f"Our CT density shape: {subject.density.shape}")
    print(f"Our CT spacing: {subject.spacing}")
    
    # volume と density の値を確認
    vol_data = subject.volume.data.cpu().numpy()
    den_data = subject.density.data.cpu().numpy()
    print(f"Volume range: [{vol_data.min():.1f}, {vol_data.max():.1f}]")
    print(f"Density range: [{den_data.min():.4f}, {den_data.max():.4f}]")
    
    # DRR生成
    device = "cuda" if torch.cuda.is_available() else "cpu"
    drr = DRR(subject, sdd=1800.0, height=512, delx=0.7).to(device)
    
    # いくつかの角度から投影
    angles = [
        (0, 0, 0, "Front"),
        (0, 90, 0, "Side"),
        (0, 0, 90, "Rotated"),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (rx, ry, rz, title) in enumerate(angles):
        rotations = torch.tensor([[rx, ry, rz]], dtype=torch.float32).to(device)
        translations = torch.tensor([[0.0, 0.0, 0.0]]).to(device)
        
        # 度からラジアンに変換
        rotations_rad = torch.deg2rad(rotations)
        
        transform = convert(rotations_rad, translations,
                          parameterization="euler_angles",
                          convention="XYZ").to(device)
        
        drr_image = drr(transform)
        
        # 表示
        img_np = drr_image.squeeze().cpu().numpy()
        axes[idx].imshow(img_np, cmap='gray')
        axes[idx].set_title(f"{title} (rx={rx}, ry={ry}, rz={rz})")
        axes[idx].axis('off')
        
        print(f"{title} - DRR range: [{img_np.min():.2f}, {img_np.max():.2f}]")
    
    plt.tight_layout()
    plt.savefig('debug_our_ct_drr.png')
    print("\nSaved: debug_our_ct_drr.png")


def check_ct_orientation():
    """CTデータの向きを確認"""
    print("\n=== Checking CT orientation ===")
    
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    
    # nibabelで読み込み
    img = nib.load(ct_path)
    data = img.get_fdata()
    
    # 各軸のスライスを取得
    mid_x = data.shape[0] // 2
    mid_y = data.shape[1] // 2
    mid_z = data.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sagittal (X軸方向)
    axes[0].imshow(data[mid_x, :, :].T, cmap='gray', origin='lower')
    axes[0].set_title(f'Sagittal (X={mid_x})')
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('Z')
    
    # Coronal (Y軸方向)
    axes[1].imshow(data[:, mid_y, :].T, cmap='gray', origin='lower')
    axes[1].set_title(f'Coronal (Y={mid_y})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    
    # Axial (Z軸方向)
    axes[2].imshow(data[:, :, mid_z].T, cmap='gray', origin='lower')
    axes[2].set_title(f'Axial (Z={mid_z})')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('debug_ct_slices.png')
    print("Saved: debug_ct_slices.png")


if __name__ == "__main__":
    # DiffDRRの例でテスト
    test_example_ct()
    
    # 実際のCTでテスト
    test_our_ct()
    
    # CT方向を確認
    check_ct_orientation()
    
    print("\n✓ Debug complete. Check the generated PNG files.")