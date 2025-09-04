#!/usr/bin/env python3
"""
減衰係数を0.0に設定してBeer-Lambert則を完全回避できるかテスト
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
import os
import matplotlib.pyplot as plt
import tempfile


def test_zero_attenuation():
    """減衰係数0.0でのテスト"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = "/home/soya/ctrate_ws/deepdrr/zero_attenuation_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # カメラパラメータ
    sdd = 1800.0
    height, width = 512, 512
    delx, dely = 0.7, 0.7
    
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    # マスクデータを読み込み
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    voxel_spacing = mask_nifti.header.get_zooms()[:3]
    mask_binary = (mask_data > 0).astype(np.float32)
    
    # 3D体積とZ方向厚み
    voxel_volume = np.prod(voxel_spacing)
    volume_3d = np.sum(mask_binary) * voxel_volume
    z_thickness_actual = np.sum(mask_binary, axis=2) * voxel_spacing[2]
    
    print("\n" + "="*60)
    print("Testing Different Attenuation Values")
    print("="*60)
    print(f"3D volume: {volume_3d:.2f} mm³")
    print(f"Actual Z-thickness range: [{z_thickness_actual[z_thickness_actual > 0].min():.1f}, "
          f"{z_thickness_actual[z_thickness_actual > 0].max():.1f}] mm")
    
    # 異なる減衰係数でテスト
    attenuation_values = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    results = {}
    
    for atten in attenuation_values:
        print(f"\n{'='*40}")
        print(f"Testing attenuation = {atten:.0e}")
        print('='*40)
        
        try:
            # マスクをスケール
            if atten == 0.0:
                # 0.0の場合、マスクの値をそのまま使用（0 or 1）
                mask_scaled = mask_binary * 0.0  # すべて0にする
                print("  Note: Setting all values to 0")
            else:
                mask_scaled = mask_binary * atten
            
            # 一時ファイルとして保存
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
                mask_nifti_scaled = nib.Nifti1Image(mask_scaled, mask_nifti.affine)
                nib.save(mask_nifti_scaled, temp_file.name)
                temp_path = temp_file.name
            
            try:
                # DiffDRRで投影
                mask_subject = read(temp_path, bone_attenuation_multiplier=1.0)
                
                drr = DRR(
                    mask_subject,
                    sdd=sdd,
                    height=height,
                    width=width,
                    delx=delx,
                    dely=dely,
                ).to(device)
                
                # 投影を実行
                transmittance = drr(rotations, translations,
                                  parameterization="euler_angles",
                                  convention="ZXY")
                
                transmittance_np = transmittance.squeeze().cpu().numpy()
                
                print(f"  Transmittance range: [{transmittance_np.min():.6f}, {transmittance_np.max():.6f}]")
                
                # 厚みを計算
                if atten == 0.0:
                    # exp(0) = 1なので、transmittance = 1になるはず
                    # この場合、厚み情報は得られない
                    thickness = 1.0 - transmittance_np  # これは0になるはず
                    print(f"  Thickness calculation: 1 - transmittance")
                else:
                    # 通常の線形近似
                    # I ≈ exp(-μ*t) ≈ 1 - μ*t （小さいμの場合）
                    thickness = (1.0 - transmittance_np) / atten
                
                thickness = np.maximum(thickness, 0)
                
                # 統計
                non_zero = thickness[thickness > 0]
                if len(non_zero) > 0:
                    stats = {
                        'min': non_zero.min(),
                        'max': non_zero.max(),
                        'mean': non_zero.mean(),
                        'std': non_zero.std(),
                        'cv': non_zero.std() / non_zero.mean() * 100,
                        'range': non_zero.max() - non_zero.min(),
                        'count': len(non_zero)
                    }
                else:
                    stats = {
                        'min': 0, 'max': 0, 'mean': 0, 'std': 0,
                        'cv': 0, 'range': 0, 'count': 0
                    }
                
                results[atten] = {
                    'transmittance': transmittance_np,
                    'thickness': thickness,
                    'stats': stats
                }
                
                print(f"  Thickness range: [{stats['min']:.3f}, {stats['max']:.3f}] mm")
                print(f"  Dynamic range: {stats['range']:.3f} mm")
                print(f"  CV: {stats['cv']:.1f}%")
                print(f"  Non-zero pixels: {stats['count']}")
                
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"  Error: {e}")
            results[atten] = None
    
    # 可視化
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    valid_results = [(a, r) for a, r in results.items() if r is not None]
    
    for i, (atten, result) in enumerate(valid_results[:9]):
        if result is not None:
            thickness = result['thickness']
            im = axes[i].imshow(thickness, cmap='hot')
            axes[i].set_title(f'μ = {atten:.0e}\nRange: {result["stats"]["range"]:.2f} mm\n'
                            f'CV: {result["stats"]["cv"]:.1f}%')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    # 余ったaxesを非表示
    for i in range(len(valid_results), 9):
        axes[i].axis('off')
    
    plt.suptitle('Effect of Attenuation Coefficient on Thickness Map', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attenuation_comparison.png'), dpi=150)
    print(f"\nVisualization saved to: {os.path.join(output_dir, 'attenuation_comparison.png')}")
    
    # Beer-Lambert則の影響を分析
    print("\n" + "="*60)
    print("ANALYSIS OF BEER-LAMBERT LAW EFFECT")
    print("="*60)
    
    if 0.0 in results and results[0.0] is not None:
        print("\n🔬 Attenuation = 0.0:")
        print("  When μ = 0, exp(-μ*t) = exp(0) = 1 for all thicknesses")
        print("  This means NO thickness information is transmitted!")
        print("  The projection becomes uniform (all pixels = 1)")
    
    # 最小の非ゼロ減衰係数での結果
    non_zero_attens = [a for a in attenuation_values if a > 0 and a in results and results[a] is not None]
    if non_zero_attens:
        best_atten = min(non_zero_attens)
        best_result = results[best_atten]
        
        print(f"\n✅ Best result with μ = {best_atten:.0e}:")
        print(f"  Dynamic range: {best_result['stats']['range']:.2f} mm")
        print(f"  CV: {best_result['stats']['cv']:.1f}%")
        
        compression_ratio = 67.5 / best_result['stats']['range'] if best_result['stats']['range'] > 0 else float('inf')
        print(f"  Compression ratio: {compression_ratio:.1f}x")
        
        if compression_ratio > 10:
            print("  ⚠️ Still significant compression due to Beer-Lambert law!")
    
    return results


if __name__ == "__main__":
    test_zero_attenuation()