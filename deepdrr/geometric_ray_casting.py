#!/usr/bin/env python3
"""
DiffDRRとPyTorch3Dを用いた、CT DRRと体積保存型マスク厚みマップの正確な生成
"""
import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
import os
import matplotlib.pyplot as plt

try:
    from pytorch3d.renderer import (
        PerspectiveCameras,
        NDCMultinomialRaysampler,
        MonteCarloRaysampler,
    )
    from pytorch3d.structures import Volumes
    from pytorch3d.transforms import euler_angles_to_matrix
except ImportError:
    print("PyTorch3Dが見つかりません。インストールしてください。")
    print("例: pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py312_cu121_torch2.4/download.html")
    exit()


def convert_diffdrr_to_pytorch3d_camera(
    sdd: float,
    height: int,
    width: int,
    delx: float,
    dely: float,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    device: torch.device,
) -> PerspectiveCameras:
    """
    DiffDRRのカメラパラメータをPyTorch3DのPerspectiveCamerasオブジェクトに変換する。
    これにより、両者の投影ジオメトリが完全に一致する。
    """
    # DiffDRRの回転(ZXYオイラー角)と並進から、PyTorch3Dが要求する
    # 回転行列(R)と並進ベクトル(T)を構築する
    # PyTorch3Dのカメラは「カメラから見たワールド」を定義するため、
    # ワールドからカメラへの変換の逆行列を用いる
    R_diffdrr = euler_angles_to_matrix(rotations, "ZXY")
    T_diffdrr = translations.unsqueeze(-1)  # (batch, 3, 1)に変換

    # ワールド→カメラ変換
    R_world_to_cam = R_diffdrr.transpose(1, 2)
    T_world_to_cam = -torch.bmm(R_world_to_cam, T_diffdrr).squeeze(-1)

    # 焦点距離と主点を計算
    focal_length_x = sdd / delx
    focal_length_y = sdd / dely
    principal_point_x = width / 2
    principal_point_y = height / 2

    focal_length = torch.tensor([[focal_length_x, focal_length_y]], device=device)
    principal_point = torch.tensor([[principal_point_x, principal_point_y]], device=device)

    return PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=R_world_to_cam,
        T=T_world_to_cam,
        image_size=((height, width),),
        in_ndc=False,
        device=device,
    )


def geometric_ray_casting(
    mask_volume: np.ndarray,
    voxel_spacing: tuple,
    camera_params: dict,
) -> np.ndarray:
    """
    PyTorch3Dを用いて、3Dバイナリマスクの純粋な幾何学的投影（厚みマップ）を計算する。
    Beer-Lambert則を一切使用せず、レイの経路長を直接積分する。
    """
    device = camera_params["device"]
    height, width = camera_params["height"], camera_params["width"]

    # 1. ボリュームの準備
    mask_tensor = torch.from_numpy(mask_volume).to(device).float().unsqueeze(0).unsqueeze(0)
    # PyTorch3DのVolumesは (N, C, D, H, W) 形式を期待
    mask_tensor = mask_tensor.permute(0, 1, 4, 3, 2)
    
    # ボリュームの物理的なサイズを計算
    vol_shape = torch.tensor(mask_tensor.shape[2:], device=device, dtype=torch.float32)
    vol_spacing = torch.tensor(voxel_spacing[::-1], device=device, dtype=torch.float32) # D, H, Wに合わせる
    vol_size = vol_shape * vol_spacing
    
    # Volumesオブジェクトを作成
    volumes = Volumes(
        densities=mask_tensor,
        voxel_size=vol_spacing
    )

    # 2. DiffDRRと完全に一致するPyTorch3Dカメラを作成
    cameras = convert_diffdrr_to_pytorch3d_camera(**camera_params)

    # 3. レイサンプラーの初期化
    # NDCMultinomialRaysamplerはボリュームの境界付近でレイをサンプリングする
    raysampler = NDCMultinomialRaysampler(
        image_width=width,
        image_height=height,
        n_pts_per_ray=512,  # 解像度に応じて調整
        min_depth=0.1,
        max_depth=torch.max(vol_size).item() * 2.0,
    )

    # 4. レイの生成とサンプリング
    ray_bundle = raysampler(cameras)
    
    # 5. ボリューム内でのサンプリング
    # ray_bundleから各レイ上の点を生成
    # ray_bundle.origins: (1, H, W, 3) -> (1, H*W, 3) にreshape
    # ray_bundle.directions: (1, H, W, 3) -> (1, H*W, 3) にreshape
    # ray_bundle.lengths: (1, H, W, n_pts) -> (1, H*W, n_pts) にreshape
    
    batch_size = ray_bundle.origins.shape[0]
    origins = ray_bundle.origins.reshape(batch_size, height * width, 3).squeeze(0)  # (H*W, 3)
    directions = ray_bundle.directions.reshape(batch_size, height * width, 3).squeeze(0)  # (H*W, 3)
    lengths = ray_bundle.lengths.reshape(batch_size, height * width, -1).squeeze(0)  # (H*W, n_pts)
    
    # レイ上のサンプル点を計算
    # points = origins + directions * lengths
    n_rays = height * width  # 明示的にH*Wを計算
    n_pts = lengths.shape[1]
    
    # デバッグ用
    print(f"   Ray bundle shape - origins: {origins.shape}, directions: {directions.shape}, lengths: {lengths.shape}")
    print(f"   n_rays: {n_rays}, n_pts: {n_pts}")
    
    # サンプル点の3D座標を計算
    points = origins.unsqueeze(1) + directions.unsqueeze(1) * lengths.unsqueeze(-1)
    # points: (H*W, n_pts, 3)
    
    # ボリュームの中心とスケールに合わせて座標を正規化
    # PyTorch3Dのgrid_sampleは[-1, 1]の範囲を期待
    # ボリュームは原点中心、[-vol_size/2, vol_size/2]の範囲にあると仮定
    normalized_points = points / (vol_size / 2.0)
    
    # grid_sample用に形状を調整
    # grid_sampleは (N, D, H, W, 3) の形式を期待（3D grid sampling）
    # しかし、pointsをそのままサンプリングするためには、
    # (N=1, n_points, 1, 1, 3) の形式にして、各ポイントを個別にサンプリング
    sample_points = normalized_points.reshape(1, -1, 1, 1, 3)
    
    # ボリュームからサンプリング
    # mask_tensorの形状: (1, 1, D, H, W)
    sampled_values = torch.nn.functional.grid_sample(
        mask_tensor,
        sample_points,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    # sampled_values: (1, 1, n_total_points, 1, 1)
    
    sampled_densities = sampled_values.squeeze().reshape(n_rays, n_pts)  # (H*W, n_pts)

    # 6. 線積分（厚みの計算）
    # 各レイのステップ長を計算
    step_size = lengths[:, 1:] - lengths[:, :-1]  # (H*W, n_pts-1)
    
    # 各ステップでのマスク値とステップ長を掛け合わせ、レイごとに合計する
    # これが物理的な厚み(mm)の計算に相当する
    # sampled_densitiesの最後の点を除外するか、step_sizeに合わせる
    thickness_map = torch.sum(sampled_densities[:, :-1] * step_size, dim=-1)
    
    # 2D画像形式にリシェイプ
    thickness_map_2d = thickness_map.view(height, width).cpu().numpy()

    return thickness_map_2d


def main():
    # --- 共通パラメータ設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = "/home/soya/ctrate_ws/deepdrr/pytorch3d_geometric_output"
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    sdd = 1800.0
    height, width = 512, 512
    delx, dely = 0.7, 0.7
    
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)

    # --- 1. CT DRRの生成 (DiffDRRを使用) ---
    print("1. Generating CT DRR with DiffDRR...")
    from diffdrr.data import read
    ct_subject = read(ct_path, bone_attenuation_multiplier=3.0)
    ct_drr_module = DRR(
        ct_subject, sdd=sdd, height=height, width=width, delx=delx, dely=dely
    ).to(device)
    ct_drr_raw = ct_drr_module(rotations, translations, parameterization="euler_angles", convention="ZXY")
    ct_drr_img = -torch.log(torch.clamp(ct_drr_raw, 1e-9)).squeeze().cpu().numpy()
    print(f"   CT DRR shape: {ct_drr_img.shape}")
    print(f"   CT DRR range: [{ct_drr_img.min():.3f}, {ct_drr_img.max():.3f}]")

    # --- 2. マスク厚みマップの生成 (PyTorch3Dを使用) ---
    print("\n2. Generating Mask Thickness Map with PyTorch3D...")
    mask_nifti = nib.load(mask_path)
    mask_data = (mask_nifti.get_fdata() > 0).astype(np.float32)
    voxel_spacing = mask_nifti.header.get_zooms()[:3]

    camera_params = {
        "sdd": sdd, "height": height, "width": width, "delx": delx, "dely": dely,
        "rotations": rotations, "translations": translations, "device": device
    }
    
    thickness_map = geometric_ray_casting(mask_data, voxel_spacing, camera_params)
    print(f"   Thickness Map shape: {thickness_map.shape}")
    print(f"   Thickness Map range: [{thickness_map.min():.3f}, {thickness_map.max():.3f}] mm")

    # --- 3. 検証 ---
    print("\n3. Verifying the results...")
    # 3D体積
    voxel_volume = np.prod(voxel_spacing)
    volume_3d = np.sum(mask_data) * voxel_volume
    
    # 2Dから計算した体積
    pixel_area = delx * dely
    volume_2d_from_thickness = np.sum(thickness_map) * pixel_area
    
    print(f"   Original 3D Volume: {volume_3d:,.2f} mm³")
    print(f"   Projected 2D Volume: {volume_2d_from_thickness:,.2f} mm³")
    error = abs(volume_3d - volume_2d_from_thickness) / volume_3d * 100
    print(f"   Volume Preservation Error: {error:.4f}%")
    print(f"   Max Thickness: {np.max(thickness_map):.2f} mm")

    # --- 4. 結果の可視化と保存 ---
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
    plt.savefig(os.path.join(output_dir, 'results_comparison.png'), dpi=150)
    print(f"   Visualization saved to {os.path.join(output_dir, 'results_comparison.png')}")
    
    # Niftiファイルとして保存
    thickness_nifti = nib.Nifti1Image(thickness_map.astype(np.float32), np.eye(4))
    nib.save(thickness_nifti, os.path.join(output_dir, 'thickness_map.nii.gz'))
    
    ct_drr_nifti = nib.Nifti1Image(ct_drr_img.astype(np.float32), np.eye(4))
    nib.save(ct_drr_nifti, os.path.join(output_dir, 'ct_drr.nii.gz'))
    
    print(f"   Data files saved to {output_dir}")


if __name__ == "__main__":
    main()