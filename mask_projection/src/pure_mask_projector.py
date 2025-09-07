#!/usr/bin/env python3
"""
純粋マスク投影機能
既存のDeepDRR結果と位置一致するDiffDRRマスク投影を実装
DeepDRRは呼び出さず、既存結果を読み込んでオーバーレイ表示
"""

import numpy as np
import torch
import nibabel as nib
from pathlib import Path
import imageio.v2 as imageio
import logging
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI不要

# DiffDRR（マスク専用）
from diffdrr.drr import DRR
from diffdrr.data import read

# LPS正規化
from nibabel import orientations as nio
from nibabel.affines import voxel_sizes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PureMaskProjector:
    """
    純粋マスク投影クラス
    既存のDeepDRR結果と位置一致するDiffDRRマスク投影
    """
    
    def __init__(self, mask_path):
        """
        初期化
        
        Args:
            mask_path: マスクファイルのパス
        """
        self.mask_path = Path(mask_path)
        
        # DiffDRR components  
        self.mask_subject = None
        self.mask_drr = None
        
        logger.info("PureMaskProjector初期化")
        
    def to_lps_array_and_spacing(self, nifti_path):
        """
        NIfTIをLPS正規化（test_deepdrr_final_pa_fixed.pyと同じ処理）
        
        Args:
            nifti_path: NIfTIファイルのパス
            
        Returns:
            tuple: (lps_array, spacing, affine_lps)
        """
        logger.info(f"📁 LPS正規化: {nifti_path}")
        
        img = nib.load(str(nifti_path))
        in_ornt = nio.io_orientation(img.affine)
        out_ornt = nio.axcodes2ornt(('L','P','S'))
        to_lps = nio.ornt_transform(in_ornt, out_ornt)

        logger.info(f"🧭 元座標系: {nio.aff2axcodes(img.affine)}")
        logger.info(f"🔄 変換行列: {to_lps}")

        data_lps = nio.apply_orientation(img.get_fdata().astype(np.float32), to_lps)
        aff_lps = img.affine @ nio.inv_ornt_aff(to_lps, img.shape)
        spacing = voxel_sizes(aff_lps)[:3]

        logger.info(f"✅ 変換後座標系: {nio.aff2axcodes(aff_lps)}")
        logger.info(f"📊 データ形状: {data_lps.shape}")
        logger.info(f"📏 spacing: {spacing}")

        return data_lps, spacing, aff_lps
        
    def save_temp_mask_lps(self, mask_lps, aff_lps, temp_path):
        """
        LPS正規化されたマスクを一時ファイルとして保存
        
        Args:
            mask_lps: LPS正規化マスクデータ
            aff_lps: LPS affine行列
            temp_path: 保存先パス
        """
        mask_img = nib.Nifti1Image(mask_lps.astype(np.float32), aff_lps)
        nib.save(mask_img, str(temp_path))
        logger.info(f"一時マスクファイル保存: {temp_path}")
        
    def setup_mask_diffdrr(self):
        """
        マスク専用DiffDRRセットアップ
        test_deepdrr_final_pa_fixed.pyと同じ幾何パラメータ
        """
        logger.info("=== マスクDiffDRRセットアップ ===")
            
        # マスクを同じLPS正規化で処理
        mask_lps_data, spacing, mask_aff_lps = self.to_lps_array_and_spacing(self.mask_path)
        
        # 一時マスクファイルを保存
        temp_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_mask_path = temp_dir / "temp_pure_mask_lps.nii.gz"
        self.save_temp_mask_lps(mask_lps_data, mask_aff_lps, temp_mask_path)
        
        # DiffDRRでマスク読み込み（center_volume=True）
        self.mask_subject = read(
            volume=str(temp_mask_path),
            orientation="PA",
            center_volume=True
        )
        
        # マスク専用DRR（test_deepdrr_final_pa_fixed.pyと同じ幾何パラメータ）
        # SID=1800, ピクセル=0.14mm, 検出器=350x430mm
        plate_w, plate_h = 350.0, 430.0
        pixel_size = 0.14
        width = int(round(plate_w / pixel_size)) & ~1   # 2500
        height = int(round(plate_h / pixel_size)) & ~1  # 3070
        
        self.mask_drr = DRR(
            self.mask_subject,
            sdd=1800.0,               # Source-to-Detector Distance
            height=height,            # 3070
            width=width,              # 2500
            delx=pixel_size,          # 0.14mm
            dely=pixel_size,          # 0.14mm
            reverse_x_axis=True,      # radiographic view
            renderer="siddon",        # 純粋な線積分
        )
        
        # 一時ファイル削除
        if temp_mask_path.exists():
            temp_mask_path.unlink()
            
        logger.info(f"マスクDiffDRRセットアップ完了: {width}x{height}")
        
    def deepdrr_to_diffdrr_angles(self, alpha_deg, beta_deg, gamma_deg):
        """
        DeepDRRからDiffDRRへの角度変換（gpt5pro3.md準拠）
        """
        a = math.radians(alpha_deg)
        b = math.radians(beta_deg)
        g = math.radians(gamma_deg)
        
        angles_diffdrr = torch.tensor([[
            a,                    # α′ = α
            (math.pi/2) - b,      # β′ = 90° - β  
            -g                    # γ′ = -γ
        ]], dtype=torch.float32)
        
        logger.info(f"角度変換: DeepDRR({alpha_deg}°,{beta_deg}°,{gamma_deg}°) -> DiffDRR({math.degrees(angles_diffdrr[0,0]):.1f}°,{math.degrees(angles_diffdrr[0,1]):.1f}°,{math.degrees(angles_diffdrr[0,2]):.1f}°)")
        
        return angles_diffdrr
        
    def project_mask(self, alpha=0., beta=0., gamma=90.):
        """
        DiffDRRによるマスク投影
        
        Args:
            alpha, beta, gamma: DeepDRRと同じ投影角度（度）
            
        Returns:
            np.ndarray: マスク厚みマップ [mm]
        """
        logger.info(f"📸 DiffDRR マスク投影: α={alpha}°, β={beta}°, γ={gamma}°")
        
        # 角度変換
        angles = self.deepdrr_to_diffdrr_angles(alpha, beta, gamma)
        translation = torch.zeros(1, 3, dtype=torch.float32)
        
        # マスク投影
        with torch.no_grad():
            mask_proj = self.mask_drr(
                angles, translation,
                parameterization="euler_angles",
                convention="YZX"
            )
            
        thickness_mm = mask_proj[0, 0].cpu().numpy()
        
        # PA投影の場合は左右反転（DeepDRRと一致させる）
        if alpha == 0. and beta == 0.:  # PA判定
            logger.info("PA投影: マスクも左右反転適用")
            thickness_mm = thickness_mm[:, ::-1].copy()
            
        logger.info(f"マスク投影完了: 厚み範囲{thickness_mm.min():.3f}-{thickness_mm.max():.3f}mm, 形状{thickness_mm.shape}")
        return thickness_mm
        
    def load_existing_deepdrr(self, deepdrr_image_path):
        """
        既存のDeepDRR画像を読み込み
        
        Args:
            deepdrr_image_path: DeepDRR画像ファイルのパス
            
        Returns:
            np.ndarray: DeepDRR画像
        """
        logger.info(f"📖 既存DeepDRR画像読み込み: {deepdrr_image_path}")
        
        drr_img = imageio.imread(deepdrr_image_path)
        
        # uint16 -> float32正規化
        if drr_img.dtype == np.uint16:
            drr_img = drr_img.astype(np.float32) / 65535.0
        elif drr_img.dtype == np.uint8:
            drr_img = drr_img.astype(np.float32) / 255.0
            
        logger.info(f"DeepDRR画像読み込み完了: 範囲{drr_img.min():.3f}-{drr_img.max():.3f}, 形状{drr_img.shape}")
        return drr_img
        
    def create_overlay_visualization(self, drr_img, mask_thickness, save_path, title="DRR & Mask Overlay"):
        """
        DRR・マスク・オーバーレイの可視化作成
        
        Args:
            drr_img: DeepDRR画像
            mask_thickness: マスク厚みマップ
            save_path: 保存先パス
            title: タイトル
        """
        logger.info("オーバーレイ可視化作成中...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # 1. 既存のDeepDRR画像
        ax1 = axes[0, 0]
        im1 = ax1.imshow(drr_img, cmap='bone', aspect='equal')
        ax1.set_title('Existing DeepDRR PA Image', fontsize=14, weight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. マスク厚みマップ
        ax2 = axes[0, 1]
        mask_display = np.where(mask_thickness > 0, mask_thickness, np.nan)
        im2 = ax2.imshow(mask_display, cmap='hot', aspect='equal')
        ax2.set_title('DiffDRR Mask Thickness [mm]', fontsize=14, weight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. オーバーレイ（DRR + マスク）
        ax3 = axes[1, 0]
        # DRRを背景として表示
        ax3.imshow(drr_img, cmap='gray', aspect='equal', alpha=0.7)
        
        # マスクをカラーでオーバーレイ
        mask_overlay = np.where(mask_thickness > 0, mask_thickness, np.nan)
        im3 = ax3.imshow(mask_overlay, cmap='hot', aspect='equal', alpha=0.8)
        ax3.set_title('Overlay: DeepDRR + DiffDRR Mask', fontsize=14, weight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Thickness [mm]')
        
        # 4. 統計情報と位置一致確認
        ax4 = axes[1, 1]
        ax4.text(0.05, 0.95, f'{title}', fontsize=16, weight='bold', transform=ax4.transAxes)
        
        # DRR統計
        ax4.text(0.05, 0.85, 'DeepDRR Statistics:', fontsize=12, weight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.8, f'Shape: {drr_img.shape}', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.75, f'Range: {drr_img.min():.3f} - {drr_img.max():.3f}', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f'Mean: {drr_img.mean():.3f}', fontsize=10, transform=ax4.transAxes)
        
        # マスク統計
        nonzero_mask = mask_thickness > 0
        ax4.text(0.05, 0.6, 'Mask Statistics:', fontsize=12, weight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.55, f'Shape: {mask_thickness.shape}', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.5, f'Max thickness: {mask_thickness.max():.3f} mm', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.45, f'Non-zero pixels: {np.sum(nonzero_mask):,}', fontsize=10, transform=ax4.transAxes)
        if np.sum(nonzero_mask) > 0:
            ax4.text(0.1, 0.4, f'Mean thickness: {mask_thickness[nonzero_mask].mean():.3f} mm', fontsize=10, transform=ax4.transAxes)
        
        # 位置一致確認情報
        ax4.text(0.05, 0.3, 'Alignment Info:', fontsize=12, weight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.25, '• Same LPS normalization', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.2, '• Same geometry parameters', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.15, '• Same PA left-right correction', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.1, '• center_volume=True alignment', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.05, f'• Shape match: {drr_img.shape == mask_thickness.shape}', fontsize=10, transform=ax4.transAxes)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"オーバーレイ可視化保存完了: {save_path}")
        
    def test_pure_mask_projection(self, deepdrr_image_path, output_dir=None):
        """
        純粋マスク投影のテスト実行
        
        Args:
            deepdrr_image_path: 既存のDeepDRR画像パス
            output_dir: 出力ディレクトリ
            
        Returns:
            dict: 投影結果
        """
        logger.info("=== 純粋マスク投影テスト開始 ===")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # セットアップ
        self.setup_mask_diffdrr()
        
        # 既存DeepDRR画像読み込み
        drr_img = self.load_existing_deepdrr(deepdrr_image_path)
        
        # PA投影（test_deepdrr_final_pa_fixed.pyと同じパラメータ）
        GAMMA_ROLL = 90.0
        
        logger.info("PA正面マスク投影実行中...")
        mask_pa = self.project_mask(alpha=0, beta=0, gamma=GAMMA_ROLL)
        
        # 個別画像保存
        if output_dir:
            # マスク画像保存
            if mask_pa.max() > 0:
                mask_normalized = np.clip(mask_pa / mask_pa.max(), 0, 1)
                mask_u16 = (mask_normalized * 65535).astype(np.uint16)
                mask_file = output_dir / "pure_mask_pa.png"
                imageio.imwrite(mask_file, mask_u16)
                logger.info(f"マスク画像保存: {mask_file}")
                
            # オーバーレイ可視化作成
            viz_file = output_dir / "pure_mask_overlay_visualization.png"
            self.create_overlay_visualization(
                drr_img, mask_pa, viz_file, 
                title="PA Projection: DeepDRR & DiffDRR Mask Overlay"
            )
        
        results = {
            'drr_image': drr_img,
            'mask_thickness': mask_pa,
            'drr_shape': drr_img.shape,
            'mask_shape': mask_pa.shape,
            'max_thickness': mask_pa.max(),
            'nonzero_pixels': np.sum(mask_pa > 0),
            'shape_match': drr_img.shape == mask_pa.shape
        }
        
        logger.info("✅ 純粋マスク投影テスト完了")
        logger.info(f"DRR形状: {results['drr_shape']}")
        logger.info(f"マスク形状: {results['mask_shape']}")
        logger.info(f"形状一致: {results['shape_match']}")
        logger.info(f"最大厚み: {results['max_thickness']:.3f} mm")
        logger.info(f"非ゼロピクセル数: {results['nonzero_pixels']:,}")
        
        return results

def test_pure_mask_projection():
    """
    純粋マスク投影のテスト実行
    """
    logger.info("=== 純粋マスク投影テスト開始 ===")
    
    # ファイルパス
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    deepdrr_image_path = "/home/soya/ctrate_ws/deepdrr/deepdrr/outputs/deepdrr_PA_final_pa_fixed.png"
    output_dir = "/home/soya/ctrate_ws/mask_projection/outputs"
    
    try:
        # プロジェクター初期化
        projector = PureMaskProjector(mask_path)
        
        # テスト実行
        results = projector.test_pure_mask_projection(
            deepdrr_image_path=deepdrr_image_path,
            output_dir=output_dir
        )
        
        logger.info("🎉 純粋マスク投影テスト成功")
        return True
        
    except Exception as e:
        logger.error(f"テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pure_mask_projection()
    if success:
        logger.info("✅ 純粋マスク投影実装完了")
    else:
        logger.error("❌ 実装失敗")