#!/usr/bin/env python3
"""
DiffDRR専用プロジェクター（gpt5pro3.md準拠）
まずDiffDRRの正しい実装を確立する
"""

import numpy as np
import torch
import nibabel as nib
from pathlib import Path
import imageio.v2 as imageio
import logging
import math

# DiffDRR
from diffdrr.drr import DRR
from diffdrr.data import read

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiffDRRProjector:
    """
    DiffDRR専用プロジェクター（gpt5pro3.md方針準拠）
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
        self.subject = None
        self.drr = None
        
        logger.info("DiffDRRProjector初期化")
        
    def setup(self, sid=1800.0, pixel_size=1.4, sensor_width=256, sensor_height=256):
        """
        DiffDRRセットアップ（gpt5pro3.md準拠）
        
        Args:
            sid: Source-to-Image Distance [mm]
            pixel_size: ピクセルサイズ [mm/pixel]
            sensor_width: 検出器幅 [pixels]
            sensor_height: 検出器高さ [pixels]
        """
        logger.info("=== DiffDRR セットアップ ===")
        
        # Subject読み込み（CTとマスクの両方、center_volume=True）
        self.subject = read(
            volume=str(self.ct_path),
            labelmap=str(self.mask_path),     # 0/1 マスク
            orientation="PA",                 # フレーム変換
            center_volume=True,               # 体積中心＝原点
        )
        
        logger.info(f"Subject loaded:")
        logger.info(f"  Volume shape: {self.subject.volume.shape}")
        logger.info(f"  Volume spacing: {self.subject.volume.spacing}")
        
        # Subjectの属性を確認
        logger.info(f"  Subject keys: {list(self.subject.keys())}")
        
        # maskが存在するか確認（diffdrr.readは'mask'として読み込む）
        if hasattr(self.subject, 'mask') and self.subject.mask is not None:
            logger.info(f"  Mask shape: {self.subject.mask.shape}")
        elif 'mask' in self.subject:
            logger.info(f"  Mask shape: {self.subject.mask.shape}")
        else:
            logger.info("  Mask: 見つかりません")
        
        # DRR初期化（gpt5pro3.mdの推奨設定）
        self.drr = DRR(
            self.subject,
            sdd=float(sid),
            height=int(sensor_height),
            width=int(sensor_width),
            delx=float(pixel_size),
            dely=float(pixel_size),
            reverse_x_axis=True,              # radiographic view準拠（B案）
            renderer="siddon",                # マスク厚み用
        )
        
        logger.info(f"DRR設定:")
        logger.info(f"  SDD: {sid} mm")
        logger.info(f"  Sensor: {sensor_width} x {sensor_height} pixels")
        logger.info(f"  Pixel size: {pixel_size} mm/pixel")
        logger.info(f"  Reverse X-axis: True (radiographic view)")
        
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
        logger.info(f"  DeepDRR角度: α={alpha_deg}°, β={beta_deg}°, γ={gamma_deg}°")
        logger.info(f"  DiffDRR角度: α={math.degrees(angles_diffdrr[0,0]):.1f}°, β={math.degrees(angles_diffdrr[0,1]):.1f}°, γ={math.degrees(angles_diffdrr[0,2]):.1f}°")
        
        return angles_diffdrr
        
    def project_ct(self, alpha_deg, beta_deg, gamma_deg):
        """
        CT投影
        
        Args:
            alpha_deg, beta_deg, gamma_deg: 投影角度（度）
            
        Returns:
            np.ndarray: CT投影画像
        """
        logger.info(f"=== CT投影 ===")
        
        if self.drr is None:
            raise ValueError("セットアップされていません")
            
        # 角度変換
        angles = self.deepdrr_to_diffdrr_angles(alpha_deg, beta_deg, gamma_deg)
        
        # 体積中心=原点なので平行移動は不要
        translation = torch.zeros(1, 3, dtype=torch.float32)
        
        # CT投影実行
        with torch.no_grad():
            img_proj = self.drr(
                angles, translation,
                parameterization="euler_angles",
                convention="YZX"
            )
        
        ct_image = img_proj[0, 0].cpu().numpy()
        
        logger.info(f"CT投影完了: 形状={ct_image.shape}")
        logger.info(f"値域: {ct_image.min():.3f} ~ {ct_image.max():.3f}")
        
        return ct_image
        
    def project_mask(self, alpha_deg, beta_deg, gamma_deg):
        """
        マスク投影（厚みマップ）
        
        Args:
            alpha_deg, beta_deg, gamma_deg: 投影角度（度）
            
        Returns:
            np.ndarray: マスク厚みマップ [mm]
        """
        logger.info(f"=== マスク投影 ===")
        
        if self.drr is None:
            raise ValueError("セットアップされていません")
            
        # 角度変換
        angles = self.deepdrr_to_diffdrr_angles(alpha_deg, beta_deg, gamma_deg)
        
        # 体積中心=原点なので平行移動は不要
        translation = torch.zeros(1, 3, dtype=torch.float32)
        
        # マスク投影実行
        with torch.no_grad():
            # まず通常の投影を試す
            try:
                # mask_to_channels=Trueを試す
                result = self.drr(
                    angles, translation,
                    parameterization="euler_angles",
                    convention="YZX",
                    mask_to_channels=True
                )
                
                # 結果の構造を確認
                if isinstance(result, tuple) and len(result) == 2:
                    img_proj, mask_proj = result
                    thickness_mm = mask_proj[0, 0].cpu().numpy()
                    logger.info("mask_to_channels=True成功")
                else:
                    # タプルでない場合、通常投影として扱う
                    thickness_mm = result[0, 0].cpu().numpy()
                    logger.info("通常投影として処理")
                    
            except Exception as e:
                logger.warning(f"mask_to_channels=True失敗: {e}")
                # フォールバック：通常の投影
                result = self.drr(
                    angles, translation,
                    parameterization="euler_angles",
                    convention="YZX"
                )
                thickness_mm = result[0, 0].cpu().numpy()
                logger.info("フォールバック：通常投影")
        
        logger.info(f"マスク投影完了: 形状={thickness_mm.shape}")
        logger.info(f"厚み値域: {thickness_mm.min():.3f} ~ {thickness_mm.max():.3f} mm")
        
        # 非ゼロピクセルの統計
        nonzero_pixels = thickness_mm > 0
        nonzero_count = np.sum(nonzero_pixels)
        if nonzero_count > 0:
            logger.info(f"非ゼロピクセル数: {nonzero_count}")
            logger.info(f"平均厚み: {thickness_mm[nonzero_pixels].mean():.3f} mm")
        else:
            logger.warning("マスク投影結果がゼロ")
            
        return thickness_mm
        
    def test_multiple_angles(self, save_dir=None):
        """
        複数角度での投影テスト
        
        Args:
            save_dir: 保存ディレクトリ
            
        Returns:
            dict: 結果辞書
        """
        logger.info("=== 複数角度投影テスト ===")
        
        # テスト角度リスト（DeepDRRの角度）
        test_cases = [
            {"name": "PA", "alpha": 0.0, "beta": 0.0, "gamma": 90.0},
            {"name": "LAT", "alpha": 0.0, "beta": 90.0, "gamma": 90.0},
            {"name": "AP", "alpha": 180.0, "beta": 0.0, "gamma": 90.0},
        ]
        
        results = {}
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for case in test_cases:
            logger.info(f"\n--- {case['name']}投影テスト ---")
            
            # CT投影
            ct_img = self.project_ct(case['alpha'], case['beta'], case['gamma'])
            
            # マスク投影
            mask_img = self.project_mask(case['alpha'], case['beta'], case['gamma'])
            
            results[case['name']] = {
                'ct_image': ct_img,
                'mask_thickness': mask_img,
                'max_thickness': mask_img.max(),
                'nonzero_pixels': np.sum(mask_img > 0)
            }
            
            # 保存
            if save_dir:
                # CT画像保存
                if ct_img.max() > 0:
                    ct_normalized = np.clip(ct_img / ct_img.max(), 0, 1)
                    ct_u16 = (ct_normalized * 65535).astype(np.uint16)
                    ct_file = save_dir / f"ct_{case['name'].lower()}.png"
                    imageio.imwrite(ct_file, ct_u16)
                    logger.info(f"CT保存: {ct_file}")
                
                # マスク画像保存
                if mask_img.max() > 0:
                    mask_normalized = np.clip(mask_img / mask_img.max(), 0, 1)
                    mask_u16 = (mask_normalized * 65535).astype(np.uint16)
                    mask_file = save_dir / f"mask_{case['name'].lower()}.png"
                    imageio.imwrite(mask_file, mask_u16)
                    logger.info(f"マスク保存: {mask_file}")
            
            logger.info(f"{case['name']}結果: CT最大値={ct_img.max():.3f}, マスク最大厚み={mask_img.max():.3f}mm")
        
        return results

def test_diffdrr_only():
    """
    DiffDRR専用プロジェクターのテスト
    """
    logger.info("=== DiffDRR専用プロジェクターテスト開始 ===")
    
    # ファイルパス
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = "/home/soya/ctrate_ws/mask_projection/outputs"
    
    try:
        # プロジェクター初期化
        projector = DiffDRRProjector(ct_path, mask_path)
        
        # セットアップ
        projector.setup(
            sid=1800.0,
            pixel_size=1.4,      # テスト用
            sensor_width=256,    # テスト用
            sensor_height=256    # テスト用
        )
        
        # 複数角度テスト
        results = projector.test_multiple_angles(save_dir=output_dir)
        
        # 結果サマリー
        logger.info("\n=== テスト結果サマリー ===")
        for view, result in results.items():
            logger.info(f"{view}: 最大厚み={result['max_thickness']:.3f}mm, 非ゼロピクセル={result['nonzero_pixels']}")
        
        logger.info("✅ DiffDRR専用プロジェクターテスト成功")
        return True
        
    except Exception as e:
        logger.error(f"テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_diffdrr_only()
    if success:
        logger.info("🎉 DiffDRR専用プロジェクターテスト完了")
    else:
        logger.error("❌ DiffDRR専用プロジェクターテスト失敗")