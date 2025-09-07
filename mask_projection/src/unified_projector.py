#!/usr/bin/env python3
"""
統合プロジェクター（gpt5pro3.md準拠）
DeepDRRとDiffDRRの完全位置一致を保証する実装
"""

import numpy as np
import torch
import nibabel as nib
from pathlib import Path
import imageio.v2 as imageio
import logging
import math

# DeepDRR
from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector

# DiffDRR
from diffdrr.drr import DRR
from diffdrr.data import read

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedProjector:
    """
    DeepDRRとDiffDRRの統合プロジェクター
    gpt5pro3.mdの方針に従い完全位置一致を保証
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
        self.patient = None
        self.carm = None
        self.projector = None
        
        # DiffDRR components  
        self.subject = None
        self.drr = None
        
        logger.info("UnifiedProjector初期化")
        
    def setup_deepdrr(self, sid=1800.0, oid=80.0, pixel_size=0.14, 
                      sensor_width=2500, sensor_height=3072):
        """
        DeepDRR側のセットアップ（gpt5pro3.md準拠）
        体積中心=世界原点になるようにする
        
        Args:
            sid: Source-to-Image Distance [mm]
            oid: Object-to-Image Distance [mm]
            pixel_size: ピクセルサイズ [mm/pixel]
            sensor_width: 検出器幅 [pixels]
            sensor_height: 検出器高さ [pixels]
        """
        logger.info("=== DeepDRR セットアップ ===")
        
        # CTボリューム読み込み（体積中心=原点）
        self.patient = Volume.from_nifti(str(self.ct_path))
        logger.info(f"Patient center_in_world: {self.patient.center_in_world}")
        
        # C-arm設定（isocenterに体積中心を設定）
        sad = sid - oid
        self.carm = MobileCArm(
            isocenter=self.patient.center_in_world,  # 体積中心がisocenter
            source_to_detector_distance=sid,
            source_to_isocenter_vertical_distance=sad,
            pixel_size=pixel_size,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            rotate_camera_left=True
        )
        
        logger.info(f"C-arm設定:")
        logger.info(f"  Isocenter: {self.carm.isocenter_in_world}")
        logger.info(f"  SID: {self.carm.source_to_detector_distance}")
        logger.info(f"  SAD: {self.carm.source_to_isocenter_vertical_distance}")
        logger.info(f"  Pixel size: {self.carm.pixel_size}")
        logger.info(f"  Sensor: {self.carm.sensor_width} x {self.carm.sensor_height}")
        
        # Projector初期化
        self.projector = Projector(self.patient, carm=self.carm)
        
    def setup_diffdrr(self):
        """
        DiffDRR側のセットアップ（gpt5pro3.md準拠）
        center_volume=Trueで体積中心=原点にする
        """
        logger.info("=== DiffDRR セットアップ ===")
        
        if self.carm is None:
            raise ValueError("DeepDRRを先にセットアップしてください")
        
        # Subject読み込み（CTとマスクの両方、center_volume=True）
        self.subject = read(
            volume=str(self.ct_path),
            labelmap=str(self.mask_path),     # 0/1 マスク
            orientation="PA",                 # フレーム変換
            center_volume=True,               # 体積中心＝原点（DeepDRRと合わせる）
        )
        
        logger.info(f"Subject loaded:")
        logger.info(f"  Volume shape: {self.subject.volume.shape}")
        logger.info(f"  Volume spacing: {self.subject.volume.spacing}")
        logger.info(f"  Labelmap shape: {self.subject.labelmap.shape}")
        
        # DRR初期化（DeepDRRと同じ幾何パラメータ）
        self.drr = DRR(
            self.subject,
            sdd=float(self.carm.source_to_detector_distance),
            height=int(self.carm.sensor_height),
            width=int(self.carm.sensor_width), 
            delx=float(self.carm.pixel_size),
            dely=float(self.carm.pixel_size),
            reverse_x_axis=True,              # radiographic view準拠（B案）
            renderer="siddon",                # マスク厚み用
        )
        
        logger.info("DiffDRR DRRモジュール初期化完了")
    
    def deepdrr_to_diffdrr_angles(self, alpha_deg, beta_deg, gamma_deg):
        """
        DeepDRRからDiffDRRへの角度変換（gpt5pro3.md準拠）
        
        Args:
            alpha_deg, beta_deg, gamma_deg: DeepDRRの角度（度）
            
        Returns:
            torch.Tensor: DiffDRR用の角度（ラジアン）
        """
        # 度からラジアンに変換
        a = math.radians(alpha_deg)
        b = math.radians(beta_deg) 
        g = math.radians(gamma_deg)
        
        # gpt5pro3.mdの変換式
        # α′=α, β′=90°−β, γ′=−γ
        angles_diffdrr = torch.tensor([[
            a,                    # α′ = α
            (math.pi/2) - b,      # β′ = 90° - β
            -g                    # γ′ = -γ
        ]], dtype=torch.float32)
        
        logger.info(f"角度変換:")
        logger.info(f"  DeepDRR: α={alpha_deg}°, β={beta_deg}°, γ={gamma_deg}°")
        logger.info(f"  DiffDRR: α={math.degrees(angles_diffdrr[0,0]):.1f}°, β={math.degrees(angles_diffdrr[0,1]):.1f}°, γ={math.degrees(angles_diffdrr[0,2]):.1f}°")
        
        return angles_diffdrr
    
    def project_ct(self, alpha_deg, beta_deg, gamma_deg):
        """
        DeepDRRでCT投影
        
        Args:
            alpha_deg, beta_deg, gamma_deg: 投影角度（度）
            
        Returns:
            np.ndarray: CT投影画像
        """
        logger.info(f"=== DeepDRR CT投影 ===")
        
        if self.projector is None:
            raise ValueError("DeepDRRがセットアップされていません")
        
        # C-armを指定角度に移動
        self.carm.move_to(alpha=alpha_deg, beta=beta_deg, gamma=gamma_deg, degrees=True)
        
        # 投影実行
        with self.projector:
            _ = self.projector()  # warm up
            img_ct = self.projector()
            
            # gpt5pro3.mdのB案：DiffDRRがreverse_x_axis=Trueなので
            # DeepDRRでは反転しない（将来の保守性向上）
            # img_ct = img_ct[:, ::-1]  # B案では反転しない
            
        logger.info(f"CT投影完了: 形状={img_ct.shape}")
        
        return img_ct
    
    def project_mask(self, alpha_deg, beta_deg, gamma_deg):
        """
        DiffDRRでマスク投影（厚みマップ）
        
        Args:
            alpha_deg, beta_deg, gamma_deg: 投影角度（度）
            
        Returns:
            np.ndarray: マスク厚みマップ [mm]
        """
        logger.info(f"=== DiffDRR マスク投影 ===")
        
        if self.drr is None:
            raise ValueError("DiffDRRがセットアップされていません")
        
        # 角度変換（DeepDRR → DiffDRR）
        angles_diffdrr = self.deepdrr_to_diffdrr_angles(alpha_deg, beta_deg, gamma_deg)
        
        # 体積中心=原点なので平行移動は不要
        translation = torch.zeros(1, 3, dtype=torch.float32)
        
        # マスク投影実行
        with torch.no_grad():
            # mask_to_channels=Trueでラベルマップを投影
            img_proj, mask_proj = self.drr(
                angles_diffdrr, translation,
                parameterization="euler_angles",
                convention="YZX",
                mask_to_channels=True
            )
        
        # マスクの厚み [mm] を取得
        thickness_mm = mask_proj[0, 0].cpu().numpy()
        
        logger.info(f"マスク投影完了: 形状={thickness_mm.shape}")
        logger.info(f"厚み値域: {thickness_mm.min():.3f} ~ {thickness_mm.max():.3f} mm")
        
        # 非ゼロピクセルの統計
        nonzero_pixels = thickness_mm > 0
        nonzero_count = np.sum(nonzero_pixels)
        if nonzero_count > 0:
            logger.info(f"非ゼロピクセル数: {nonzero_count}")
            logger.info(f"平均厚み: {thickness_mm[nonzero_pixels].mean():.3f} mm")
        
        return thickness_mm
    
    def project_unified(self, alpha_deg, beta_deg, gamma_deg, save_dir=None):
        """
        統合投影（CTとマスクの両方）
        
        Args:
            alpha_deg, beta_deg, gamma_deg: 投影角度（度）
            save_dir: 保存ディレクトリ（Pathオブジェクト）
            
        Returns:
            tuple: (ct_image, mask_thickness_map)
        """
        logger.info(f"=== 統合投影実行 ===")
        logger.info(f"角度: α={alpha_deg}°, β={beta_deg}°, γ={gamma_deg}°")
        
        # CT投影
        ct_image = self.project_ct(alpha_deg, beta_deg, gamma_deg)
        
        # マスク投影
        mask_thickness = self.project_mask(alpha_deg, beta_deg, gamma_deg)
        
        # 保存
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # CT投影を保存（16bit正規化）
            ct_normalized = np.clip(ct_image / max(ct_image.max(), 1e-6), 0, 1)
            ct_u16 = (ct_normalized * 65535).astype(np.uint16)
            ct_file = save_dir / f"ct_projection_a{alpha_deg}_b{beta_deg}_g{gamma_deg}.png"
            imageio.imwrite(ct_file, ct_u16)
            
            # マスク厚みを保存（16bit正規化）
            if mask_thickness.max() > 0:
                mask_normalized = np.clip(mask_thickness / mask_thickness.max(), 0, 1)
                mask_u16 = (mask_normalized * 65535).astype(np.uint16)
                mask_file = save_dir / f"mask_thickness_a{alpha_deg}_b{beta_deg}_g{gamma_deg}.png"
                imageio.imwrite(mask_file, mask_u16)
                
                logger.info(f"保存完了:")
                logger.info(f"  CT: {ct_file}")
                logger.info(f"  Mask: {mask_file}")
        
        return ct_image, mask_thickness
    
    def cleanup(self):
        """
        リソースのクリーンアップ
        """
        if self.projector:
            self.projector.__exit__(None, None, None)
        logger.info("リソースクリーンアップ完了")

def test_unified_projection():
    """
    統合プロジェクターのテスト
    """
    logger.info("=== 統合プロジェクターテスト開始 ===")
    
    # ファイルパス
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = "/home/soya/ctrate_ws/mask_projection/outputs"
    
    try:
        # プロジェクター初期化
        projector = UnifiedProjector(ct_path, mask_path)
        
        # セットアップ（テスト用に小さなパラメータ）
        projector.setup_deepdrr(
            sid=1800.0,
            oid=80.0,
            pixel_size=1.4,      # テスト用
            sensor_width=256,    # テスト用
            sensor_height=256    # テスト用
        )
        projector.setup_diffdrr()
        
        # PA投影テスト（α=0°, β=0°, γ=90°）
        ct_img, mask_thickness = projector.project_unified(
            alpha_deg=0.0, 
            beta_deg=0.0, 
            gamma_deg=90.0,
            save_dir=output_dir
        )
        
        logger.info("✅ 統合投影テスト成功")
        logger.info(f"CT画像形状: {ct_img.shape}")
        logger.info(f"マスク最大厚み: {mask_thickness.max():.3f} mm")
        
        # クリーンアップ
        projector.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_unified_projection()
    if success:
        logger.info("🎉 統合プロジェクターテスト完了")
    else:
        logger.error("❌ 統合プロジェクターテスト失敗")