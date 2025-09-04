#!/usr/bin/env python3
"""
最終実装：DiffDRRを使用した幾何学的投影
線形近似を用いてBeer-Lambert則の影響を最小化
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


class GeometricProjector:
    """
    幾何学的投影を実現するクラス
    DiffDRRの座標系を利用しつつ、線形近似で厚みを計算
    """
    
    def __init__(self, device: torch.device = None):
        """
        初期化
        
        Args:
            device: 計算デバイス
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # カメラパラメータ（DiffDRRと同じ）
        self.sdd = 1800.0
        self.height = 512
        self.width = 512
        self.delx = 0.7
        self.dely = 0.7
        
        # デフォルトの姿勢（PA view）
        self.rotations = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        self.translations = torch.tensor([[0.0, 850.0, 0.0]], device=self.device)
    
    def project_mask(
        self,
        mask_path: str,
        use_calibration: bool = True,
        attenuation_scale: float = 0.001
    ) -> Dict:
        """
        マスクの幾何学的投影を実行
        
        Args:
            mask_path: マスクのNIfTIファイルパス
            use_calibration: 体積保存のためのキャリブレーションを使用するか
            attenuation_scale: 減衰係数のスケール（小さいほど線形近似が良い）
        
        Returns:
            結果の辞書
        """
        print(f"\nProjecting mask: {os.path.basename(mask_path)}")
        print(f"Parameters: attenuation_scale={attenuation_scale}, calibration={use_calibration}")
        
        # マスクデータを読み込み
        mask_nifti = nib.load(mask_path)
        mask_data = mask_nifti.get_fdata()
        voxel_spacing = mask_nifti.header.get_zooms()[:3]
        
        # バイナリマスクに変換
        mask_binary = (mask_data > 0).astype(np.float32)
        
        # 3D体積を計算
        voxel_volume = np.prod(voxel_spacing)
        volume_3d = np.sum(mask_binary) * voxel_volume
        
        print(f"3D volume: {volume_3d:.2f} mm³")
        
        # マスクをスケール（減衰係数を小さくして線形領域で動作させる）
        mask_scaled = mask_binary * attenuation_scale
        
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
                sdd=self.sdd,
                height=self.height,
                width=self.width,
                delx=self.delx,
                dely=self.dely,
            ).to(self.device)
            
            # 投影を実行
            transmittance = drr(
                self.rotations,
                self.translations,
                parameterization="euler_angles",
                convention="ZXY"
            )
            
            # 透過率を厚みに変換
            # 小さいμの場合: I ≈ exp(-μ*t) ≈ 1 - μ*t （テイラー展開の1次項）
            # したがって: t ≈ (1 - I) / μ
            transmittance_np = transmittance.squeeze().cpu().numpy()
            
            # 線形近似を使用
            thickness_raw = (1.0 - transmittance_np) / attenuation_scale
            
            # 負の値を0にクリップ
            thickness_raw = np.maximum(thickness_raw, 0)
            
            # キャリブレーション
            if use_calibration:
                pixel_area = self.delx * self.dely
                volume_projected = np.sum(thickness_raw) * pixel_area
                
                if volume_projected > 0:
                    calibration_factor = volume_3d / volume_projected
                    thickness_calibrated = thickness_raw * calibration_factor
                    print(f"Calibration factor: {calibration_factor:.4f}")
                else:
                    thickness_calibrated = thickness_raw
                    calibration_factor = 1.0
                
                volume_final = np.sum(thickness_calibrated) * pixel_area
                print(f"Projected volume: {volume_final:.2f} mm³")
                print(f"Volume preservation error: {abs(volume_3d - volume_final) / volume_3d * 100:.4f}%")
            else:
                thickness_calibrated = thickness_raw
                calibration_factor = 1.0
            
        finally:
            # 一時ファイルを削除
            os.unlink(temp_path)
        
        return {
            'thickness': thickness_calibrated,
            'thickness_raw': thickness_raw,
            'volume_3d': volume_3d,
            'calibration_factor': calibration_factor,
            'voxel_spacing': voxel_spacing,
            'transmittance': transmittance_np
        }
    
    def project_ct(self, ct_path: str) -> np.ndarray:
        """
        CTのDRRを生成（比較用）
        
        Args:
            ct_path: CTのNIfTIファイルパス
        
        Returns:
            CT DRR画像
        """
        print(f"\nGenerating CT DRR: {os.path.basename(ct_path)}")
        
        ct_subject = read(ct_path, bone_attenuation_multiplier=3.0)
        
        drr = DRR(
            ct_subject,
            sdd=self.sdd,
            height=self.height,
            width=self.width,
            delx=self.delx,
            dely=self.dely,
        ).to(self.device)
        
        ct_drr_raw = drr(
            self.rotations,
            self.translations,
            parameterization="euler_angles",
            convention="ZXY"
        )
        
        # 対数変換で可視化用に調整
        ct_drr_img = -torch.log(torch.clamp(ct_drr_raw, 1e-9)).squeeze().cpu().numpy()
        
        return ct_drr_img


def visualize_results(
    ct_drr: np.ndarray,
    mask_thickness: np.ndarray,
    mask_thickness_raw: np.ndarray,
    output_path: str
):
    """
    結果を可視化
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # CT DRR
    axes[0, 0].imshow(ct_drr, cmap='gray')
    axes[0, 0].set_title('CT DRR')
    axes[0, 0].axis('off')
    
    # 生の厚みマップ
    im1 = axes[0, 1].imshow(mask_thickness_raw, cmap='hot')
    axes[0, 1].set_title(f'Raw Thickness\nMax: {mask_thickness_raw.max():.1f} mm')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # キャリブレート後の厚みマップ
    im2 = axes[0, 2].imshow(mask_thickness, cmap='hot')
    axes[0, 2].set_title(f'Calibrated Thickness\nMax: {mask_thickness.max():.1f} mm')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # オーバーレイ（生）
    axes[1, 0].imshow(ct_drr, cmap='gray', alpha=0.7)
    axes[1, 0].imshow(mask_thickness_raw > 0, cmap='Reds', alpha=0.3)
    axes[1, 0].set_title('Raw Overlay')
    axes[1, 0].axis('off')
    
    # オーバーレイ（キャリブレート後）
    axes[1, 1].imshow(ct_drr, cmap='gray', alpha=0.7)
    axes[1, 1].imshow(mask_thickness > 0, cmap='Reds', alpha=0.3)
    axes[1, 1].set_title('Calibrated Overlay')
    axes[1, 1].axis('off')
    
    # ヒストグラム
    axes[1, 2].hist(mask_thickness[mask_thickness > 0], bins=50, alpha=0.7, label='Calibrated')
    axes[1, 2].hist(mask_thickness_raw[mask_thickness_raw > 0], bins=50, alpha=0.7, label='Raw')
    axes[1, 2].set_xlabel('Thickness (mm)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Thickness Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to: {output_path}")


def main():
    """
    メイン処理
    """
    print("="*60)
    print("Geometric Projection of Binary Masks using DiffDRR")
    print("="*60)
    
    # パス設定
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = "/home/soya/ctrate_ws/deepdrr/final_output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # プロジェクタを初期化
    projector = GeometricProjector()
    
    # CT DRRを生成
    ct_drr = projector.project_ct(ct_path)
    
    # マスクを投影（異なる設定でテスト）
    results = []
    attenuation_scales = [0.001, 0.01, 0.1]
    
    for scale in attenuation_scales:
        print(f"\n{'='*40}")
        print(f"Testing attenuation scale: {scale}")
        print('='*40)
        
        result = projector.project_mask(mask_path, use_calibration=True, attenuation_scale=scale)
        result['attenuation_scale'] = scale
        results.append(result)
    
    # 最良の結果を選択（体積保存が最も良いもの）
    best_result = min(results, key=lambda r: abs(r['volume_3d'] - np.sum(r['thickness']) * projector.delx * projector.dely))
    best_scale = best_result['attenuation_scale']
    
    print(f"\n{'='*40}")
    print(f"Best attenuation scale: {best_scale}")
    print('='*40)
    
    # 結果を可視化
    visualize_results(
        ct_drr,
        best_result['thickness'],
        best_result['thickness_raw'],
        os.path.join(output_dir, 'final_results.png')
    )
    
    # 結果を保存
    np.save(os.path.join(output_dir, 'thickness_calibrated.npy'), best_result['thickness'])
    np.save(os.path.join(output_dir, 'thickness_raw.npy'), best_result['thickness_raw'])
    np.save(os.path.join(output_dir, 'ct_drr.npy'), ct_drr)
    
    # 統計情報を出力
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Attenuation scale: {best_scale}")
    print(f"Calibration factor: {best_result['calibration_factor']:.4f}")
    print(f"3D Volume: {best_result['volume_3d']:.2f} mm³")
    
    pixel_area = projector.delx * projector.dely
    volume_projected = np.sum(best_result['thickness']) * pixel_area
    print(f"Projected Volume: {volume_projected:.2f} mm³")
    
    error = abs(best_result['volume_3d'] - volume_projected) / best_result['volume_3d'] * 100
    print(f"Volume preservation error: {error:.4f}%")
    
    thickness = best_result['thickness']
    print(f"\nThickness statistics:")
    print(f"  Min: {thickness.min():.3f} mm")
    print(f"  Max: {thickness.max():.3f} mm")
    print(f"  Mean (non-zero): {thickness[thickness > 0].mean():.3f} mm")
    print(f"  Std (non-zero): {thickness[thickness > 0].std():.3f} mm")
    print(f"  Non-zero pixels: {np.sum(thickness > 0)}")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()