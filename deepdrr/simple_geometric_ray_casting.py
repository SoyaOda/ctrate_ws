#!/usr/bin/env python3
"""
DiffDRRのカメラパラメータを使用した純粋な幾何学的レイキャスティング実装
Beer-Lambert則を使用せずに、マスクの厚みを直接計算
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
import os
import matplotlib.pyplot as plt
from typing import Tuple, Dict


def create_ray_directions(
    height: int,
    width: int,
    sdd: float,
    delx: float,
    dely: float,
    device: torch.device
) -> torch.Tensor:
    """
    各ピクセルに対応するレイの方向ベクトルを生成
    
    Args:
        height: 画像の高さ
        width: 画像の幅
        sdd: Source-to-detector distance (mm)
        delx: ピクセルサイズX (mm)
        dely: ピクセルサイズY (mm)
        device: デバイス
    
    Returns:
        ray_directions: (H, W, 3) のレイ方向ベクトル（正規化済み）
    """
    # ピクセル座標を生成 (0, 0)が画像の左上
    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    
    # メッシュグリッドを作成
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # 画像中心を原点とした座標系に変換
    # DiffDRRと同じ座標系を使用
    xx_centered = (xx - width / 2.0) * delx
    yy_centered = (yy - height / 2.0) * dely
    
    # レイの方向ベクトルを計算
    # Z軸が投影方向、X, Yが画像平面
    ray_directions = torch.stack([
        xx_centered,
        yy_centered,
        torch.full_like(xx_centered, sdd)
    ], dim=-1)
    
    # 正規化
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    
    return ray_directions


def ray_box_intersection(
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    レイとAABB（軸平行境界ボックス）の交差を計算
    
    Args:
        ray_origin: レイの開始点 (3,)
        ray_direction: レイの方向 (H, W, 3)
        box_min: ボックスの最小座標 (3,)
        box_max: ボックスの最大座標 (3,)
    
    Returns:
        t_min: レイがボックスに入る距離
        t_max: レイがボックスから出る距離
    """
    # 各軸での交差パラメータを計算
    inv_direction = 1.0 / (ray_direction + 1e-8)
    
    t1 = (box_min - ray_origin) * inv_direction
    t2 = (box_max - ray_origin) * inv_direction
    
    # 各軸での最小・最大を計算
    t_mins = torch.minimum(t1, t2)
    t_maxs = torch.maximum(t1, t2)
    
    # 全軸での最大の最小値と最小の最大値を取得
    t_min = torch.max(t_mins, dim=-1)[0]
    t_max = torch.min(t_maxs, dim=-1)[0]
    
    # レイがボックスと交差しない場合の処理
    valid_mask = t_max > t_min
    t_min = torch.where(valid_mask, t_min, torch.zeros_like(t_min))
    t_max = torch.where(valid_mask, t_max, torch.zeros_like(t_max))
    
    return t_min, t_max


def sample_along_rays(
    mask_volume: torch.Tensor,
    voxel_spacing: torch.Tensor,
    ray_origin: torch.Tensor,
    ray_directions: torch.Tensor,
    t_min: torch.Tensor,
    t_max: torch.Tensor,
    num_samples: int = 512
) -> torch.Tensor:
    """
    レイに沿ってボリュームをサンプリングし、厚みを計算
    
    Args:
        mask_volume: マスクボリューム (D, H, W)
        voxel_spacing: ボクセルサイズ (3,)
        ray_origin: レイの開始点 (3,)
        ray_directions: レイの方向 (H, W, 3)
        t_min: サンプリング開始距離 (H, W)
        t_max: サンプリング終了距離 (H, W)
        num_samples: レイあたりのサンプル数
    
    Returns:
        thickness_map: 厚みマップ (H, W)
    """
    device = mask_volume.device
    H, W = ray_directions.shape[:2]
    
    # サンプリング点を生成
    t_vals = torch.linspace(0, 1, num_samples, device=device)
    t_vals = t_vals.view(1, 1, num_samples)
    
    # 各レイに対してサンプリング距離を計算
    t_samples = t_min.unsqueeze(-1) + (t_max - t_min).unsqueeze(-1) * t_vals
    
    # サンプリング点の3D座標を計算
    # points: (H, W, num_samples, 3)
    points = ray_origin.view(1, 1, 1, 3) + ray_directions.unsqueeze(2) * t_samples.unsqueeze(-1)
    
    # ボリューム座標系に変換（ボクセル単位）
    volume_shape = torch.tensor(mask_volume.shape, device=device, dtype=torch.float32)
    volume_coords = points / voxel_spacing.view(1, 1, 1, 3)
    
    # ボリュームの中心を原点とした座標系から、
    # [0, volume_shape]の座標系に変換
    volume_coords = volume_coords + volume_shape.view(1, 1, 1, 3) / 2.0
    
    # grid_sample用に正規化 [-1, 1]
    normalized_coords = 2.0 * volume_coords / volume_shape.view(1, 1, 1, 3) - 1.0
    
    # 座標の順序を調整 (Z, Y, X) -> (X, Y, Z)
    normalized_coords = normalized_coords[..., [2, 1, 0]]
    
    # grid_sampleの形式に合わせる
    # (1, 1, H*W*num_samples, 1, 3)
    sample_coords = normalized_coords.reshape(1, 1, -1, 1, 3)
    
    # マスクボリュームに次元を追加 (1, 1, D, H, W)
    mask_volume_5d = mask_volume.unsqueeze(0).unsqueeze(0)
    
    # サンプリング
    sampled_values = torch.nn.functional.grid_sample(
        mask_volume_5d,
        sample_coords,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    
    # 形状を戻す (H, W, num_samples)
    sampled_values = sampled_values.squeeze().reshape(H, W, num_samples)
    
    # 各サンプル間の距離を計算
    dt = (t_max - t_min) / num_samples
    
    # 厚みを積分（台形則）
    thickness_map = torch.sum(sampled_values, dim=-1) * dt.squeeze()
    
    return thickness_map


def geometric_ray_casting_simple(
    mask_volume: np.ndarray,
    voxel_spacing: tuple,
    camera_params: dict,
) -> np.ndarray:
    """
    シンプルな幾何学的レイキャスティングによるマスク厚み投影
    
    Args:
        mask_volume: 3Dマスクボリューム
        voxel_spacing: ボクセルサイズ (x, y, z)
        camera_params: カメラパラメータ
    
    Returns:
        thickness_map: 2D厚みマップ
    """
    device = camera_params["device"]
    height = camera_params["height"]
    width = camera_params["width"]
    sdd = camera_params["sdd"]
    delx = camera_params["delx"]
    dely = camera_params["dely"]
    translations = camera_params["translations"]
    
    # マスクボリュームをテンソルに変換
    mask_tensor = torch.from_numpy(mask_volume).to(device).float()
    voxel_spacing_tensor = torch.tensor(voxel_spacing, device=device, dtype=torch.float32)
    
    # ボリュームのバウンディングボックスを計算
    volume_shape = torch.tensor(mask_tensor.shape, device=device, dtype=torch.float32)
    volume_size = volume_shape * voxel_spacing_tensor
    box_min = -volume_size / 2.0
    box_max = volume_size / 2.0
    
    # レイの開始点（X線源の位置）
    # DiffDRRの座標系：translations[0, 1]がsource-to-isocenter距離
    ray_origin = torch.tensor([0.0, -translations[0, 1].item(), 0.0], device=device)
    
    # 各ピクセルのレイ方向を計算
    ray_directions = create_ray_directions(height, width, sdd, delx, dely, device)
    
    # レイとボリュームの交差を計算
    t_min, t_max = ray_box_intersection(ray_origin, ray_directions, box_min, box_max)
    
    # レイに沿ってサンプリングし、厚みを計算
    thickness_map = sample_along_rays(
        mask_tensor, voxel_spacing_tensor,
        ray_origin, ray_directions,
        t_min, t_max,
        num_samples=512
    )
    
    return thickness_map.cpu().numpy()


def main():
    """メイン処理"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パスの設定
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = "/home/soya/ctrate_ws/deepdrr/simple_ray_casting_output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # カメラパラメータ
    sdd = 1800.0
    height, width = 512, 512
    delx, dely = 0.7, 0.7
    
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    # 1. CT DRRの生成（比較用）
    print("1. Generating CT DRR with DiffDRR...")
    ct_subject = read(ct_path, bone_attenuation_multiplier=3.0)
    ct_drr_module = DRR(
        ct_subject, sdd=sdd, height=height, width=width, delx=delx, dely=dely
    ).to(device)
    ct_drr_raw = ct_drr_module(rotations, translations, parameterization="euler_angles", convention="ZXY")
    ct_drr_img = -torch.log(torch.clamp(ct_drr_raw, 1e-9)).squeeze().cpu().numpy()
    print(f"   CT DRR shape: {ct_drr_img.shape}")
    print(f"   CT DRR range: [{ct_drr_img.min():.3f}, {ct_drr_img.max():.3f}]")
    
    # 2. マスク厚みマップの生成
    print("\n2. Generating Mask Thickness Map with Simple Ray Casting...")
    mask_nifti = nib.load(mask_path)
    mask_data = (mask_nifti.get_fdata() > 0).astype(np.float32)
    voxel_spacing = mask_nifti.header.get_zooms()[:3]
    
    camera_params = {
        "sdd": sdd,
        "height": height,
        "width": width,
        "delx": delx,
        "dely": dely,
        "translations": translations,
        "device": device
    }
    
    thickness_map = geometric_ray_casting_simple(mask_data, voxel_spacing, camera_params)
    print(f"   Thickness Map shape: {thickness_map.shape}")
    print(f"   Thickness Map range: [{thickness_map.min():.3f}, {thickness_map.max():.3f}] mm")
    
    # 3. 体積の検証
    print("\n3. Verifying volume preservation...")
    voxel_volume = np.prod(voxel_spacing)
    volume_3d = np.sum(mask_data) * voxel_volume
    
    pixel_area = delx * dely
    volume_2d = np.sum(thickness_map) * pixel_area
    
    print(f"   Original 3D Volume: {volume_3d:,.2f} mm³")
    print(f"   Projected 2D Volume: {volume_2d:,.2f} mm³")
    
    if volume_3d > 0:
        error = abs(volume_3d - volume_2d) / volume_3d * 100
        print(f"   Volume Preservation Error: {error:.4f}%")
    
    print(f"   Max Thickness: {np.max(thickness_map):.2f} mm")
    print(f"   Non-zero pixels: {np.sum(thickness_map > 0)}")
    
    # 4. 結果の可視化
    print("\n4. Saving visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # CT DRR
    axes[0].imshow(ct_drr_img, cmap='gray')
    axes[0].set_title('CT DRR')
    axes[0].axis('off')
    
    # 厚みマップ
    im = axes[1].imshow(thickness_map, cmap='hot')
    axes[1].set_title(f'Thickness Map\n(max: {np.max(thickness_map):.1f} mm)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # オーバーレイ
    axes[2].imshow(ct_drr_img, cmap='gray', alpha=0.7)
    mask_overlay = thickness_map > 0
    axes[2].imshow(mask_overlay, cmap='Reds', alpha=0.3)
    axes[2].set_title('CT DRR + Mask Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results.png'), dpi=150)
    print(f"   Saved to {os.path.join(output_dir, 'results.png')}")
    
    # データの保存
    thickness_nifti = nib.Nifti1Image(thickness_map.astype(np.float32), np.eye(4))
    nib.save(thickness_nifti, os.path.join(output_dir, 'thickness_map.nii.gz'))
    
    ct_drr_nifti = nib.Nifti1Image(ct_drr_img.astype(np.float32), np.eye(4))
    nib.save(ct_drr_nifti, os.path.join(output_dir, 'ct_drr.nii.gz'))
    
    print(f"   Data saved to {output_dir}")


if __name__ == "__main__":
    main()