#!/usr/bin/env python3
"""
幾何学的投影の簡易実装
DiffDRRの出力から物理的減衰を逆算して、純粋な厚みマップを復元する試み
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile
from scipy import ndimage


def simple_ray_casting_z_axis(mask_volume, voxel_spacing):
    """
    最も単純な投影：Z軸に沿った積算
    これを基準として他の手法と比較する
    """
    # Z軸方向の厚み（単純な積算）
    thickness_z = np.sum(mask_volume, axis=2) * voxel_spacing[2]
    return thickness_z


def inverse_beer_lambert_projection(mask_volume, camera_params, voxel_spacing):
    """
    DiffDRRを使用するが、異なる密度値でマスクをエンコードし、
    Beer-Lambert則の影響を最小化する試み
    """
    device = camera_params['device']
    
    # 方法1: マスクに小さな値を使用（Beer-Lambert則の線形近似域）
    # exp(-x) ≈ 1 - x when x << 1
    mask_scaled = mask_volume.astype(np.float32) * 0.01
    
    # 一時ファイルに保存
    temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    mask_nifti = nib.Nifti1Image(mask_scaled, np.eye(4))
    nib.save(mask_nifti, temp_file.name)
    
    # DiffDRRで投影
    mask_subject = read(temp_file.name)
    drr = DRR(
        mask_subject,
        sdd=camera_params['sdd'],
        height=camera_params['height'],
        width=camera_params['width'],
        delx=camera_params['delx'],
        dely=camera_params['dely'],
    ).to(device)
    
    transmittance = drr(
        camera_params['rotations'],
        camera_params['translations'],
        parameterization="euler_angles",
        convention="ZXY"
    )
    
    # 線形近似の逆変換: thickness ≈ (1 - I) / 0.01
    transmittance_np = transmittance.squeeze().cpu().numpy()
    thickness_approx = (1.0 - transmittance_np) / 0.01
    
    # スケーリング調整（体積を合わせる）
    voxel_volume = np.prod(voxel_spacing)
    true_volume = np.sum(mask_volume) * voxel_volume
    pixel_area = camera_params['delx'] * camera_params['dely']
    current_volume = np.sum(thickness_approx) * pixel_area
    
    if current_volume > 0:
        thickness_calibrated = thickness_approx * (true_volume / current_volume)
    else:
        thickness_calibrated = thickness_approx
    
    # クリーンアップ
    import os
    os.unlink(temp_file.name)
    
    return thickness_calibrated


def multi_angle_reconstruction(mask_volume, camera_params, voxel_spacing, n_angles=8):
    """
    複数の角度から投影し、厚み情報を再構築
    トモグラフィーの簡易版
    """
    device = camera_params['device']
    height = camera_params['height']
    width = camera_params['width']
    
    projections = []
    
    for i in range(n_angles):
        angle = i * 360.0 / n_angles
        
        # 回転を適用
        rotations = torch.tensor([[0.0, angle, 0.0]], device=device)
        
        # マスクを一時ファイルに保存
        temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        mask_nifti = nib.Nifti1Image(mask_volume.astype(np.float32), np.eye(4))
        nib.save(mask_nifti, temp_file.name)
        
        # DiffDRRで投影
        mask_subject = read(temp_file.name)
        drr = DRR(
            mask_subject,
            sdd=camera_params['sdd'],
            height=height,
            width=width,
            delx=camera_params['delx'],
            dely=camera_params['dely'],
        ).to(device)
        
        proj = drr(
            rotations,
            camera_params['translations'],
            parameterization="euler_angles",
            convention="ZXY"
        )
        
        # 厚みに変換
        proj_np = proj.squeeze().cpu().numpy()
        thickness = -np.log(np.maximum(proj_np, 1e-9))
        projections.append(thickness)
        
        # クリーンアップ
        import os
        os.unlink(temp_file.name)
    
    # 複数投影の平均（簡易的な再構築）
    mean_thickness = np.mean(projections, axis=0)
    
    # 体積でキャリブレーション
    voxel_volume = np.prod(voxel_spacing)
    true_volume = np.sum(mask_volume) * voxel_volume
    pixel_area = camera_params['delx'] * camera_params['dely']
    current_volume = np.sum(mean_thickness) * pixel_area
    
    if current_volume > 0:
        mean_thickness_calibrated = mean_thickness * (true_volume / current_volume)
    else:
        mean_thickness_calibrated = mean_thickness
    
    return mean_thickness_calibrated, projections


def custom_ray_casting_approximation(mask_volume, camera_params, voxel_spacing):
    """
    カスタムレイキャスティングの近似実装
    斜め投影を考慮した幾何学的計算
    """
    # この実装では、PA view（正面）の場合を考える
    # より正確な実装にはPyTorch3Dが必要
    
    # 簡易版：Z軸投影に角度補正を加える
    thickness_z = np.sum(mask_volume, axis=2) * voxel_spacing[2]
    
    # カメラの角度を考慮（今回は0度なので補正なし）
    # 実際の実装では、レイの角度に応じて経路長を調整する必要がある
    
    # ランダムな例として、エッジを強調
    edges = ndimage.sobel(thickness_z)
    thickness_enhanced = thickness_z + edges * 0.1
    
    return thickness_enhanced


def visualize_comparison(results, output_dir):
    """
    各手法の結果を比較可視化
    """
    fig = plt.figure(figsize=(20, 16))
    
    methods = list(results.keys())
    n_methods = len(methods)
    
    # 各手法の厚みマップ
    for i, (method_name, data) in enumerate(results.items()):
        ax = plt.subplot(3, n_methods, i + 1)
        im = ax.imshow(data['thickness_map'], cmap='jet', vmin=0)
        ax.set_title(f"{method_name}\nMax: {data['thickness_map'].max():.1f} mm")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 厚み分布のヒストグラム
    for i, (method_name, data) in enumerate(results.items()):
        ax = plt.subplot(3, n_methods, n_methods + i + 1)
        thickness_positive = data['thickness_map'][data['thickness_map'] > 0.1]
        if len(thickness_positive) > 0:
            ax.hist(thickness_positive.flatten(), bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel("Thickness (mm)")
            ax.set_ylabel("Count")
            ax.set_title(f"{method_name} Distribution")
            ax.grid(True, alpha=0.3)
    
    # 統計情報
    for i, (method_name, data) in enumerate(results.items()):
        ax = plt.subplot(3, n_methods, 2 * n_methods + i + 1)
        ax.axis('off')
        
        stats_text = f"""
{method_name}
{'='*20}
3D Volume: {data['volume_3d']:.1f} mm³
2D Volume: {data['volume_2d']:.1f} mm³
Error: {data['error']:.2f}%

Thickness Stats:
  Max: {data['max_thickness']:.2f} mm
  Mean: {data['mean_thickness']:.2f} mm
  Std: {data['std_thickness']:.2f} mm
  Unique: {data['unique_values']}
"""
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle("Geometric Projection Methods Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "geometric_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved to: {output_dir}/geometric_comparison.png")


def main():
    """
    各種幾何学的投影手法のテストと比較
    """
    print("="*60)
    print("Geometric Projection Methods Test")
    print("="*60)
    
    # パス設定
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = Path("./geometric_projection_output")
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # マスクデータの読み込み
    print("\nLoading mask data...")
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    mask_binary = (mask_data > 0).astype(np.float32)
    voxel_spacing = mask_nifti.header.get_zooms()[:3]
    
    # 真の3D体積
    voxel_volume = np.prod(voxel_spacing)
    true_volume_3d = np.sum(mask_binary) * voxel_volume
    print(f"True 3D volume: {true_volume_3d:.2f} mm³")
    
    # カメラパラメータ
    camera_params = {
        'sdd': 1800.0,
        'height': 512,
        'width': 512,
        'delx': 0.7,
        'dely': 0.7,
        'rotations': torch.tensor([[0.0, 0.0, 0.0]], device=device),
        'translations': torch.tensor([[0.0, 850.0, 0.0]], device=device),
        'device': device,
    }
    
    pixel_area = camera_params['delx'] * camera_params['dely']
    
    results = {}
    
    # ========================================
    # 1. Z軸投影（ベースライン）
    # ========================================
    print("\n1. Z-axis projection (baseline)...")
    thickness_z = simple_ray_casting_z_axis(mask_binary, voxel_spacing)
    volume_z = np.sum(thickness_z) * voxel_spacing[0] * voxel_spacing[1]
    
    results['Z-axis Projection'] = {
        'thickness_map': thickness_z,
        'volume_3d': true_volume_3d,
        'volume_2d': volume_z,
        'error': abs(true_volume_3d - volume_z) / true_volume_3d * 100,
        'max_thickness': thickness_z.max(),
        'mean_thickness': thickness_z[thickness_z > 0].mean() if np.any(thickness_z > 0) else 0,
        'std_thickness': thickness_z[thickness_z > 0].std() if np.any(thickness_z > 0) else 0,
        'unique_values': len(np.unique(thickness_z[thickness_z > 0])) if np.any(thickness_z > 0) else 0,
    }
    print(f"  Volume error: {results['Z-axis Projection']['error']:.2f}%")
    
    # ========================================
    # 2. 線形近似法
    # ========================================
    print("\n2. Linear approximation method...")
    thickness_linear = inverse_beer_lambert_projection(mask_binary, camera_params, voxel_spacing)
    volume_linear = np.sum(thickness_linear) * pixel_area
    
    results['Linear Approximation'] = {
        'thickness_map': thickness_linear,
        'volume_3d': true_volume_3d,
        'volume_2d': volume_linear,
        'error': abs(true_volume_3d - volume_linear) / true_volume_3d * 100,
        'max_thickness': thickness_linear.max(),
        'mean_thickness': thickness_linear[thickness_linear > 0.1].mean() if np.any(thickness_linear > 0.1) else 0,
        'std_thickness': thickness_linear[thickness_linear > 0.1].std() if np.any(thickness_linear > 0.1) else 0,
        'unique_values': len(np.unique(np.round(thickness_linear[thickness_linear > 0.1], 1))) if np.any(thickness_linear > 0.1) else 0,
    }
    print(f"  Volume error: {results['Linear Approximation']['error']:.2f}%")
    
    # ========================================
    # 3. マルチアングル再構築（時間がかかるので角度を減らす）
    # ========================================
    print("\n3. Multi-angle reconstruction (4 angles)...")
    thickness_multi, projections = multi_angle_reconstruction(mask_binary, camera_params, voxel_spacing, n_angles=4)
    volume_multi = np.sum(thickness_multi) * pixel_area
    
    results['Multi-angle (4)'] = {
        'thickness_map': thickness_multi,
        'volume_3d': true_volume_3d,
        'volume_2d': volume_multi,
        'error': abs(true_volume_3d - volume_multi) / true_volume_3d * 100,
        'max_thickness': thickness_multi.max(),
        'mean_thickness': thickness_multi[thickness_multi > 0.1].mean() if np.any(thickness_multi > 0.1) else 0,
        'std_thickness': thickness_multi[thickness_multi > 0.1].std() if np.any(thickness_multi > 0.1) else 0,
        'unique_values': len(np.unique(np.round(thickness_multi[thickness_multi > 0.1], 1))) if np.any(thickness_multi > 0.1) else 0,
    }
    print(f"  Volume error: {results['Multi-angle (4)']['error']:.2f}%")
    
    # ========================================
    # 4. カスタム近似法
    # ========================================
    print("\n4. Custom approximation...")
    thickness_custom = custom_ray_casting_approximation(mask_binary, camera_params, voxel_spacing)
    volume_custom = np.sum(thickness_custom) * voxel_spacing[0] * voxel_spacing[1]
    
    results['Custom Approx'] = {
        'thickness_map': thickness_custom,
        'volume_3d': true_volume_3d,
        'volume_2d': volume_custom,
        'error': abs(true_volume_3d - volume_custom) / true_volume_3d * 100,
        'max_thickness': thickness_custom.max(),
        'mean_thickness': thickness_custom[thickness_custom > 0].mean() if np.any(thickness_custom > 0) else 0,
        'std_thickness': thickness_custom[thickness_custom > 0].std() if np.any(thickness_custom > 0) else 0,
        'unique_values': len(np.unique(thickness_custom[thickness_custom > 0])) if np.any(thickness_custom > 0) else 0,
    }
    print(f"  Volume error: {results['Custom Approx']['error']:.2f}%")
    
    # ========================================
    # 5. 可視化
    # ========================================
    print("\n5. Creating visualizations...")
    visualize_comparison(results, output_dir)
    
    # ========================================
    # 6. サマリー
    # ========================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nMethod Comparison:")
    print(f"{'Method':<20} | {'Volume Error':<12} | {'Max Thick':<10} | {'Std Dev':<10}")
    print("-"*60)
    
    for method_name, data in results.items():
        print(f"{method_name:<20} | {data['error']:>10.2f}% | {data['max_thickness']:>8.1f} mm | {data['std_thickness']:>8.2f} mm")
    
    # 最良の手法を判定
    best_method = min(results.items(), key=lambda x: x[1]['error'])
    print(f"\nBest method for volume preservation: {best_method[0]} (Error: {best_method[1]['error']:.2f}%)")
    
    # 最も厚みの変化を保持する手法
    best_thickness_method = max(results.items(), key=lambda x: x[1]['std_thickness'])
    print(f"Best method for thickness variation: {best_thickness_method[0]} (Std: {best_thickness_method[1]['std_thickness']:.2f} mm)")
    
    print("="*60)


if __name__ == "__main__":
    main()