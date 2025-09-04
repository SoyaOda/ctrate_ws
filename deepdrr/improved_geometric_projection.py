#!/usr/bin/env python3
"""
改良版：厚みの動的範囲を保持する幾何学的投影
DiffDRRを複数回使用して線形性を改善
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
import os
import matplotlib.pyplot as plt
import tempfile
from typing import Dict, Tuple


class ImprovedGeometricProjector:
    """
    改良版幾何学的投影
    厚みの動的範囲を保持
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # カメラパラメータ
        self.sdd = 1800.0
        self.height = 512
        self.width = 512
        self.delx = 0.7
        self.dely = 0.7
        
        # デフォルトの姿勢
        self.rotations = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        self.translations = torch.tensor([[0.0, 850.0, 0.0]], device=self.device)
    
    def project_mask_improved(self, mask_path: str) -> Dict:
        """
        改良版：複数の減衰係数で投影し、線形領域を抽出
        """
        print("\n" + "="*60)
        print("Improved Geometric Projection")
        print("="*60)
        
        # マスクデータを読み込み
        mask_nifti = nib.load(mask_path)
        mask_data = mask_nifti.get_fdata()
        voxel_spacing = mask_nifti.header.get_zooms()[:3]
        mask_binary = (mask_data > 0).astype(np.float32)
        
        # 3D体積とZ方向厚み
        voxel_volume = np.prod(voxel_spacing)
        volume_3d = np.sum(mask_binary) * voxel_volume
        z_thickness_actual = np.sum(mask_binary, axis=2) * voxel_spacing[2]
        
        print(f"3D volume: {volume_3d:.2f} mm³")
        print(f"Actual Z-thickness range: [{z_thickness_actual[z_thickness_actual > 0].min():.1f}, "
              f"{z_thickness_actual[z_thickness_actual > 0].max():.1f}] mm")
        
        # 方法1: 極小減衰係数での投影（線形近似が最も良い）
        print("\nMethod 1: Ultra-small attenuation coefficient")
        print("-"*40)
        
        attenuation = 0.0001  # さらに小さくする
        thickness_ultra_small = self._project_with_attenuation(mask_binary, mask_nifti.affine, attenuation)
        
        # 方法2: 差分法（2つの異なる減衰係数の差から線形成分を抽出）
        print("\nMethod 2: Differential method")
        print("-"*40)
        
        atten1 = 0.001
        atten2 = 0.002
        
        proj1 = self._project_with_attenuation(mask_binary, mask_nifti.affine, atten1)
        proj2 = self._project_with_attenuation(mask_binary, mask_nifti.affine, atten2)
        
        # 差分から線形成分を推定
        # I1 = exp(-μ1*t) ≈ 1 - μ1*t
        # I2 = exp(-μ2*t) ≈ 1 - μ2*t
        # I1 - I2 ≈ (μ2 - μ1)*t
        thickness_differential = (proj1 - proj2) / (atten2 - atten1)
        thickness_differential = np.maximum(thickness_differential, 0)
        
        # 方法3: 対数線形化（マスクを対数空間で処理）
        print("\nMethod 3: Logarithmic linearization")
        print("-"*40)
        
        # マスクの値を厚みに応じて事前にエンコード
        # Z方向の累積厚みを各ボクセルに割り当て
        thickness_encoded = np.zeros_like(mask_binary)
        for z in range(mask_binary.shape[2]):
            z_sum = np.sum(mask_binary[:, :, z:], axis=2)
            # 厚みを非線形にエンコード（Beer-Lambert則の逆関数）
            # 目標: DiffDRRの出力が線形になるようにする
            # exp(-μ*encoded) = linear_thickness となるようにする
            # encoded = -log(linear_thickness) / μ
            max_thickness = np.max(z_sum) * voxel_spacing[2]
            if max_thickness > 0:
                normalized = z_sum * voxel_spacing[2] / max_thickness
                # 0.01から1の範囲にマッピング（log(0)を避ける）
                normalized = np.clip(normalized, 0, 1) * 0.99 + 0.01
                thickness_encoded[:, :, z] = -np.log(normalized)
        
        thickness_logarithmic = self._project_with_values(thickness_encoded, mask_nifti.affine, scale=1.0)
        
        # 逆変換で線形厚みに戻す
        thickness_logarithmic = np.exp(-thickness_logarithmic) * 70  # 最大厚み70mmでスケール
        
        # 体積でキャリブレーション
        pixel_area = self.delx * self.dely
        
        results = {}
        
        for method_name, thickness in [
            ("ultra_small", thickness_ultra_small),
            ("differential", thickness_differential),
            ("logarithmic", thickness_logarithmic)
        ]:
            volume_projected = np.sum(thickness) * pixel_area
            if volume_projected > 0:
                calibration_factor = volume_3d / volume_projected
                thickness_calibrated = thickness * calibration_factor
            else:
                calibration_factor = 1.0
                thickness_calibrated = thickness
            
            volume_final = np.sum(thickness_calibrated) * pixel_area
            
            non_zero = thickness_calibrated[thickness_calibrated > 0]
            if len(non_zero) > 0:
                cv = non_zero.std() / non_zero.mean() * 100
                range_val = non_zero.max() - non_zero.min()
            else:
                cv = 0
                range_val = 0
            
            results[method_name] = {
                'thickness': thickness_calibrated,
                'thickness_raw': thickness,
                'calibration_factor': calibration_factor,
                'volume': volume_final,
                'cv': cv,
                'range': range_val,
                'stats': {
                    'min': non_zero.min() if len(non_zero) > 0 else 0,
                    'max': non_zero.max() if len(non_zero) > 0 else 0,
                    'mean': non_zero.mean() if len(non_zero) > 0 else 0,
                    'std': non_zero.std() if len(non_zero) > 0 else 0,
                }
            }
            
            print(f"\n{method_name.upper()} Results:")
            print(f"  Volume: {volume_final:.2f} mm³ (error: {abs(volume_3d - volume_final)/volume_3d*100:.2f}%)")
            print(f"  Range: {range_val:.2f} mm")
            print(f"  CV: {cv:.1f}%")
            print(f"  Thickness: [{results[method_name]['stats']['min']:.2f}, {results[method_name]['stats']['max']:.2f}] mm")
        
        return results, volume_3d, z_thickness_actual
    
    def _project_with_attenuation(self, mask_binary: np.ndarray, affine: np.ndarray, 
                                  attenuation: float) -> np.ndarray:
        """指定した減衰係数でマスクを投影"""
        mask_scaled = mask_binary * attenuation
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
            mask_nifti_scaled = nib.Nifti1Image(mask_scaled, affine)
            nib.save(mask_nifti_scaled, temp_file.name)
            temp_path = temp_file.name
        
        try:
            mask_subject = read(temp_path, bone_attenuation_multiplier=1.0)
            drr = DRR(
                mask_subject, sdd=self.sdd, height=self.height,
                width=self.width, delx=self.delx, dely=self.dely
            ).to(self.device)
            
            transmittance = drr(self.rotations, self.translations,
                              parameterization="euler_angles", convention="ZXY")
            transmittance_np = transmittance.squeeze().cpu().numpy()
            
            # 線形近似
            thickness = (1.0 - transmittance_np) / attenuation
            thickness = np.maximum(thickness, 0)
            
        finally:
            os.unlink(temp_path)
        
        return thickness
    
    def _project_with_values(self, volume: np.ndarray, affine: np.ndarray, 
                           scale: float) -> np.ndarray:
        """任意の値を持つボリュームを投影"""
        volume_scaled = volume * scale
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
            volume_nifti = nib.Nifti1Image(volume_scaled.astype(np.float32), affine)
            nib.save(volume_nifti, temp_file.name)
            temp_path = temp_file.name
        
        try:
            volume_subject = read(temp_path, bone_attenuation_multiplier=1.0)
            drr = DRR(
                volume_subject, sdd=self.sdd, height=self.height,
                width=self.width, delx=self.delx, dely=self.dely
            ).to(self.device)
            
            transmittance = drr(self.rotations, self.translations,
                              parameterization="euler_angles", convention="ZXY")
            result = -torch.log(torch.clamp(transmittance, 1e-9)).squeeze().cpu().numpy()
            
        finally:
            os.unlink(temp_path)
        
        return result


def visualize_comparison(results: Dict, volume_3d: float, z_thickness_actual: np.ndarray,
                         output_path: str):
    """結果を比較可視化"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # 実際のZ厚み（グラウンドトゥルース）
    im = axes[0, 0].imshow(z_thickness_actual, cmap='hot')
    axes[0, 0].set_title(f'Ground Truth Z-thickness\nRange: {z_thickness_actual[z_thickness_actual>0].min():.1f}-{z_thickness_actual[z_thickness_actual>0].max():.1f} mm')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
    
    # 各手法の結果
    methods = list(results.keys())
    for i, method in enumerate(methods):
        row = i
        thickness = results[method]['thickness']
        
        # 厚みマップ
        im = axes[row, 1].imshow(thickness, cmap='hot', vmin=0, vmax=70)
        axes[row, 1].set_title(f'{method.upper()}\nRange: {results[method]["range"]:.1f} mm')
        axes[row, 1].axis('off')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)
        
        # ヒストグラム
        non_zero = thickness[thickness > 0]
        if len(non_zero) > 0:
            axes[row, 2].hist(non_zero, bins=50, edgecolor='black')
            axes[row, 2].set_title(f'CV: {results[method]["cv"]:.1f}%')
            axes[row, 2].set_xlabel('Thickness (mm)')
            axes[row, 2].set_ylabel('Count')
            axes[row, 2].grid(True, alpha=0.3)
        
        # 詳細（中央100x100）
        h, w = thickness.shape
        center = thickness[h//2-50:h//2+50, w//2-50:w//2+50]
        im = axes[row, 3].imshow(center, cmap='hot', vmin=0, vmax=70)
        axes[row, 3].set_title(f'Center 100x100')
        axes[row, 3].axis('off')
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046)
    
    plt.suptitle(f'Comparison of Geometric Projection Methods\nTarget Volume: {volume_3d:.0f} mm³', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nComparison saved to: {output_path}")


def main():
    print("Improved Geometric Projection with Dynamic Range Preservation")
    
    # パス設定
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = "/home/soya/ctrate_ws/deepdrr/improved_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 投影を実行
    projector = ImprovedGeometricProjector()
    results, volume_3d, z_thickness_actual = projector.project_mask_improved(mask_path)
    
    # 最良の手法を選択（CVが最も高い = 動的範囲が保持されている）
    best_method = max(results.keys(), key=lambda k: results[k]['cv'])
    
    print("\n" + "="*60)
    print(f"BEST METHOD: {best_method.upper()}")
    print("="*60)
    print(f"Coefficient of Variation: {results[best_method]['cv']:.1f}%")
    print(f"Dynamic Range: {results[best_method]['range']:.1f} mm")
    print(f"Volume Error: {abs(volume_3d - results[best_method]['volume'])/volume_3d*100:.2f}%")
    
    # 可視化
    visualize_comparison(results, volume_3d, z_thickness_actual,
                        os.path.join(output_dir, 'improved_comparison.png'))
    
    # 最良の結果を保存
    np.save(os.path.join(output_dir, f'best_thickness_{best_method}.npy'),
            results[best_method]['thickness'])
    
    return results


if __name__ == "__main__":
    main()