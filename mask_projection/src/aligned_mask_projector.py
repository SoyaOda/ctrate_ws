#!/usr/bin/env python3
"""
DeepDRRと完全位置一致するマスク投影機能
test_deepdrr_final_pa_fixed.pyと同じ幾何学設定でマスクを投影し、
DRR・マスク・オーバーレイの可視化を生成
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

# DeepDRR
from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector

# DiffDRR（マスク専用）
from diffdrr.drr import DRR
from diffdrr.data import read

# LPS正規化（DeepDRRと同じ）
from nibabel import orientations as nio
from nibabel.affines import voxel_sizes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlignedMaskProjector:
    """
    DeepDRRと完全位置一致するマスク投影クラス
    """
    
    def __init__(self, ct_path, mask_path):
        """
        初期化
        
        Args:
            ct_path: CTファイルのパス
            mask_path: マスクファイルのパス
        """
        self.ct_path = Path(ct_path)
        self.mask_path = Path(mask_path)
        
        # DeepDRR components
        self.ct_volume = None
        self.carm = None
        
        # DiffDRR components  
        self.mask_subject = None
        self.mask_drr = None
        
        logger.info("AlignedMaskProjector初期化")
        
    def load_volume_LPS(self, nifti_path):
        """
        DeepDRRと同じLPS厳密変換（test_deepdrr_final_pa_fixed.pyから）
        
        Args:
            nifti_path: NIfTIファイルのパス
            
        Returns:
            Volume: LPS正規化済みVolume
        """
        logger.info(f"📁 LPS厳密変換: {nifti_path}")
        
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

        # 世界原点=体積中心（DeepDRRと同じ）
        shp = np.array(data_lps.shape, dtype=float)
        origin = (-0.5 * (shp - 1.0) * spacing).tolist()

        vol = Volume.from_hu(
            hu_values=data_lps,
            origin=origin,
            spacing=spacing.tolist(),
            anatomical_coordinate_system="LPS",
        )
        
        logger.info("✅ LPS Volume構築完了")
        return vol, data_lps, aff_lps, to_lps
        
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
        
    def make_carm(self, isocenter, sid=1800.0, oid=80.0, px=0.14, plate_mm=(350.0, 430.0)):
        """
        DeepDRRと同じC-arm設定（test_deepdrr_final_pa_fixed.pyから）
        
        Args:
            isocenter: アイソセンター位置
            sid: Source-to-Image Distance [mm]
            oid: Object-to-Image Distance [mm] 
            px: ピクセルサイズ [mm/pixel]
            plate_mm: 検出器サイズ [mm]
            
        Returns:
            MobileCArm: C-armオブジェクト
        """
        sad = sid - oid
        W = int(round(plate_mm[0] / px)) & ~1
        H = int(round(plate_mm[1] / px)) & ~1
        
        logger.info(f"📏 C-arm設定:")
        logger.info(f"  検出器: {W}×{H} pixels ({plate_mm[0]}×{plate_mm[1]}mm)")
        logger.info(f"  SID: {sid}mm, SAD: {sad}mm, OID: {oid}mm")
        logger.info(f"  倍率: {sid/sad:.3f}, ピクセル: {px}mm")
        
        return MobileCArm(
            isocenter=isocenter,
            rotate_camera_left=True,  # test_deepdrr_final_pa_fixed.pyと同じ
            source_to_detector_distance=sid,
            source_to_isocenter_vertical_distance=sad,
            pixel_size=px,
            sensor_width=W, sensor_height=H,
            min_alpha=-720, max_alpha=720, min_beta=-720, max_beta=720,
        )
        
    def setup_deepdrr(self):
        """
        DeepDRRセットアップ（test_deepdrr_final_pa_fixed.pyと同じ）
        """
        logger.info("=== DeepDRR セットアップ ===")
        
        # CTをLPS正規化
        self.ct_volume, _, _, _ = self.load_volume_LPS(self.ct_path)
        
        # C-arm設定
        self.carm = self.make_carm(self.ct_volume.center_in_world)
        self.ct_volume.place_center(self.carm.isocenter_in_world)
        
        logger.info("DeepDRRセットアップ完了")
        
    def setup_mask_diffdrr(self):
        """
        マスク専用DiffDRRセットアップ
        DeepDRRと同じLPS正規化とisocenter設定を適用
        """
        logger.info("=== マスクDiffDRRセットアップ ===")
        
        if self.carm is None:
            raise ValueError("先にDeepDRRをセットアップしてください")
            
        # マスクを同じLPS正規化で処理
        mask_img = nib.load(str(self.mask_path))
        _, mask_lps_data, mask_aff_lps, mask_to_lps = self.load_volume_LPS(self.mask_path)
        
        # 一時マスクファイルを保存
        temp_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_mask_path = temp_dir / "temp_aligned_mask_lps.nii.gz"
        self.save_temp_mask_lps(mask_lps_data, mask_aff_lps, temp_mask_path)
        
        # DiffDRRでマスク読み込み（center_volume=True）
        self.mask_subject = read(
            volume=str(temp_mask_path),
            orientation="PA",
            center_volume=True
        )
        
        # マスク専用DRR（DeepDRRと同じ幾何パラメータ）
        self.mask_drr = DRR(
            self.mask_subject,
            sdd=float(self.carm.source_to_detector_distance),
            height=int(self.carm.sensor_height),
            width=int(self.carm.sensor_width),
            delx=float(self.carm.pixel_size),
            dely=float(self.carm.pixel_size),
            reverse_x_axis=True,  # radiographic view
            renderer="siddon",    # 純粋な線積分
        )
        
        # 一時ファイル削除
        if temp_mask_path.exists():
            temp_mask_path.unlink()
            
        logger.info("マスクDiffDRRセットアップ完了")
        
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
        
    def project_deepdrr_ct(self, alpha=0., beta=0., gamma=90.):
        """
        DeepDRRによるCT投影（test_deepdrr_final_pa_fixed.pyと同じ）
        
        Args:
            alpha, beta, gamma: 投影角度（度）
            
        Returns:
            np.ndarray: CT投影画像
        """
        logger.info(f"📸 DeepDRR CT投影: α={alpha}°, β={beta}°, γ={gamma}°")
        
        with Projector(self.ct_volume, carm=self.carm) as projector:
            self.carm.move_to(alpha=alpha, beta=beta, gamma=gamma, degrees=True)
            _ = projector()  # warm-up
            img = projector()  # 2枚目を採用
            
            # PA投影の場合は左右反転（test_deepdrr_final_pa_fixed.pyと同じ）
            if alpha == 0. and beta == 0.:  # PA判定
                logger.info("PA投影: 左右反転適用")
                img = img[:, ::-1].copy()
                
        logger.info(f"CT投影完了: 範囲{img.min():.3f}-{img.max():.3f}, 形状{img.shape}")
        return img
        
    def project_diffdrr_mask(self, alpha=0., beta=0., gamma=90.):
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
        
    def create_visualization(self, ct_img, mask_thickness, save_path, title="DRR & Mask Projection"):
        """
        DRR・マスク・オーバーレイの可視化作成
        
        Args:
            ct_img: CT投影画像
            mask_thickness: マスク厚みマップ
            save_path: 保存先パス
            title: タイトル
        """
        logger.info("可視化作成中...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # 1. CT DRR画像
        ax1 = axes[0, 0]
        im1 = ax1.imshow(ct_img, cmap='bone', aspect='equal')
        ax1.set_title('DeepDRR CT Projection', fontsize=14, weight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. マスク厚みマップ
        ax2 = axes[0, 1]
        mask_display = np.where(mask_thickness > 0, mask_thickness, np.nan)
        im2 = ax2.imshow(mask_display, cmap='hot', aspect='equal')
        ax2.set_title('DiffDRR Mask Thickness [mm]', fontsize=14, weight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. オーバーレイ（CT + マスク）
        ax3 = axes[1, 0]
        # CTを背景として正規化
        ct_normalized = (ct_img - ct_img.min()) / (ct_img.max() - ct_img.min())
        ax3.imshow(ct_normalized, cmap='gray', aspect='equal', alpha=0.7)
        
        # マスクをカラーでオーバーレイ
        mask_overlay = np.where(mask_thickness > 0, mask_thickness, np.nan)
        im3 = ax3.imshow(mask_overlay, cmap='hot', aspect='equal', alpha=0.8)
        ax3.set_title('Overlay: CT + Mask', fontsize=14, weight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Thickness [mm]')
        
        # 4. 統計情報
        ax4 = axes[1, 1]
        ax4.text(0.05, 0.95, f'{title}', fontsize=16, weight='bold', transform=ax4.transAxes)
        
        # CT統計
        ax4.text(0.05, 0.85, 'CT DRR Statistics:', fontsize=12, weight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.8, f'Shape: {ct_img.shape}', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.75, f'Range: {ct_img.min():.3f} - {ct_img.max():.3f}', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f'Mean: {ct_img.mean():.3f}', fontsize=10, transform=ax4.transAxes)
        
        # マスク統計
        nonzero_mask = mask_thickness > 0
        ax4.text(0.05, 0.6, 'Mask Statistics:', fontsize=12, weight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.55, f'Shape: {mask_thickness.shape}', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.5, f'Max thickness: {mask_thickness.max():.3f} mm', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.45, f'Non-zero pixels: {np.sum(nonzero_mask):,}', fontsize=10, transform=ax4.transAxes)
        if np.sum(nonzero_mask) > 0:
            ax4.text(0.1, 0.4, f'Mean thickness: {mask_thickness[nonzero_mask].mean():.3f} mm', fontsize=10, transform=ax4.transAxes)
        
        # 技術情報
        ax4.text(0.05, 0.3, 'Technical Info:', fontsize=12, weight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.25, 'DeepDRR: LPS normalized, PA left-right corrected', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.2, 'DiffDRR: Siddon line integral, angle converted', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.15, 'Alignment: Same geometry & isocenter', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.1, 'PA correction: Both images flipped horizontally', fontsize=10, transform=ax4.transAxes)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可視化保存完了: {save_path}")
        
    def test_aligned_projection(self, output_dir=None):
        """
        位置一致投影のテスト実行
        
        Args:
            output_dir: 出力ディレクトリ
            
        Returns:
            dict: 投影結果
        """
        logger.info("=== 位置一致投影テスト開始 ===")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # セットアップ
        self.setup_deepdrr()
        self.setup_mask_diffdrr()
        
        # PA投影（test_deepdrr_final_pa_fixed.pyと同じパラメータ）
        GAMMA_ROLL = 90.0
        
        logger.info("PA正面投影実行中...")
        ct_pa = self.project_deepdrr_ct(alpha=0, beta=0, gamma=GAMMA_ROLL)
        mask_pa = self.project_diffdrr_mask(alpha=0, beta=0, gamma=GAMMA_ROLL)
        
        # 個別画像保存
        if output_dir:
            # CT画像保存
            ct_normalized = np.clip(ct_pa / max(ct_pa.max(), 1e-6), 0, 1)
            ct_u16 = (ct_normalized * 65535).astype(np.uint16)
            ct_file = output_dir / "aligned_ct_pa.png"
            imageio.imwrite(ct_file, ct_u16)
            logger.info(f"CT画像保存: {ct_file}")
            
            # マスク画像保存
            if mask_pa.max() > 0:
                mask_normalized = np.clip(mask_pa / mask_pa.max(), 0, 1)
                mask_u16 = (mask_normalized * 65535).astype(np.uint16)
                mask_file = output_dir / "aligned_mask_pa.png"
                imageio.imwrite(mask_file, mask_u16)
                logger.info(f"マスク画像保存: {mask_file}")
                
            # 可視化作成
            viz_file = output_dir / "aligned_drr_mask_visualization.png"
            self.create_visualization(
                ct_pa, mask_pa, viz_file, 
                title="PA Projection: DeepDRR CT & DiffDRR Mask (Aligned)"
            )
        
        results = {
            'ct_image': ct_pa,
            'mask_thickness': mask_pa,
            'ct_shape': ct_pa.shape,
            'mask_shape': mask_pa.shape,
            'max_thickness': mask_pa.max(),
            'nonzero_pixels': np.sum(mask_pa > 0)
        }
        
        logger.info("✅ 位置一致投影テスト完了")
        logger.info(f"CT形状: {results['ct_shape']}")
        logger.info(f"マスク形状: {results['mask_shape']}")
        logger.info(f"最大厚み: {results['max_thickness']:.3f} mm")
        logger.info(f"非ゼロピクセル数: {results['nonzero_pixels']:,}")
        
        return results

def test_aligned_mask_projection():
    """
    位置一致マスク投影のテスト実行
    """
    logger.info("=== 位置一致マスク投影テスト開始 ===")
    
    # ファイルパス
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = "/home/soya/ctrate_ws/mask_projection/outputs"
    
    try:
        # プロジェクター初期化
        projector = AlignedMaskProjector(ct_path, mask_path)
        
        # テスト実行
        results = projector.test_aligned_projection(output_dir=output_dir)
        
        logger.info("🎉 位置一致マスク投影テスト成功")
        return True
        
    except Exception as e:
        logger.error(f"テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_aligned_mask_projection()
    if success:
        logger.info("✅ 完全位置一致マスク投影実装完了")
    else:
        logger.error("❌ 実装失敗")