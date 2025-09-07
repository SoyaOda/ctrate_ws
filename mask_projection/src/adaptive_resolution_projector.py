#!/usr/bin/env python3
"""
段階的解像度アップによるマスク投影
重い処理を回避するため、段階的に解像度を上げてテスト
"""
import os
import gc
import time
import logging
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from diffdrr.drr import DRR
from diffdrr.data import read

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveResolutionProjector:
    """段階的解像度でマスク投影を実行"""
    
    def __init__(self, mask_path):
        self.mask_path = mask_path
        self.mask_subject = None
        
    def setup_mask_diffdrr(self, resolution_scale=0.25):
        """
        DiffDRRマスクセットアップ（解像度スケール調整可能）
        
        Args:
            resolution_scale: 解像度倍率 (0.1=最小, 1.0=最大)
        """
        logger.info(f"=== マスクDiffDRRセットアップ (解像度倍率: {resolution_scale:.2f}) ===")
        
        # マスク読み込み・LPS正規化（既存コードと同じ）
        mask_lps_data, spacing, mask_aff_lps = self.to_lps_array_and_spacing(self.mask_path)
        
        # 一時マスクファイル保存
        temp_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_mask_path = temp_dir / f"temp_adaptive_mask_{resolution_scale:.2f}.nii.gz"
        self.save_temp_mask_lps(mask_lps_data, mask_aff_lps, temp_mask_path)
        
        # DiffDRRでマスク読み込み
        self.mask_subject = read(
            volume=str(temp_mask_path),
            orientation="PA",
            center_volume=True
        )
        
        # 解像度を段階的に調整
        base_plate_w, base_plate_h = 350.0, 430.0
        base_pixel_size = 0.14
        
        # スケールに応じてピクセルサイズを調整（解像度を下げる）
        pixel_size = base_pixel_size / resolution_scale
        plate_w = base_plate_w * resolution_scale
        plate_h = base_plate_h * resolution_scale
        
        width = int(round(plate_w / base_pixel_size)) & ~1
        height = int(round(plate_h / base_pixel_size)) & ~1
        
        logger.info(f"解像度設定: {width}x{height}, ピクセルサイズ: {pixel_size:.3f}mm")
        
        self.mask_drr = DRR(
            self.mask_subject,
            sdd=1800.0,
            height=height,
            width=width,
            delx=pixel_size,
            dely=pixel_size,
            reverse_x_axis=True,
            renderer="siddon",
        )
        
        # 一時ファイル削除
        if temp_mask_path.exists():
            temp_mask_path.unlink()
            
        logger.info(f"マスクDiffDRRセットアップ完了: {width}x{height}")
        
        return width, height
        
    def to_lps_array_and_spacing(self, nifti_path):
        """LPS正規化（pure_mask_projector.pyと同じ）"""
        from nibabel import orientations as nio
        from nibabel.affines import voxel_sizes
        
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
        
    def save_temp_mask_lps(self, data_lps, aff_lps, save_path):
        """一時マスクファイル保存"""
        logger.info(f"一時マスクファイル保存: {save_path}")
        temp_nib = nib.Nifti1Image(data_lps, aff_lps)
        nib.save(temp_nib, save_path)
        
    def project_mask(self, alpha=0., beta=0., gamma=90.):
        """マスク投影"""
        logger.info(f"📸 マスク投影: α={alpha}°, β={beta}°, γ={gamma}°")
        
        # 角度変換（pure_mask_projector.pyと同じ）
        import math
        a = math.radians(alpha)
        b = math.radians(beta)
        g = math.radians(gamma)
        
        angles_diffdrr = torch.tensor([[
            a,
            b + math.pi/2,
            g - math.pi/2
        ]], dtype=torch.float32)
        
        translation = torch.zeros(1, 3, dtype=torch.float32)
        
        # マスク投影
        start_time = time.time()
        with torch.no_grad():
            mask_proj = self.mask_drr(
                angles_diffdrr, translation,
                parameterization="euler_angles",
                convention="YZX"
            )
            
        thickness_mm = mask_proj[0, 0].cpu().numpy()
        projection_time = time.time() - start_time
        
        logger.info(f"投影完了: {thickness_mm.shape}, 範囲{thickness_mm.min():.3f}-{thickness_mm.max():.3f}mm, 時間{projection_time:.2f}秒")
        
        return thickness_mm, projection_time
        
    def test_progressive_resolution(self, deepdrr_image_path, output_dir=None):
        """段階的解像度テスト"""
        logger.info("=== 段階的解像度マスク投影テスト ===")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # DeepDRR画像読み込み
        drr_img = imageio.imread(deepdrr_image_path)
        if drr_img.dtype == np.uint16:
            drr_img = drr_img.astype(np.float32) / 65535.0
        logger.info(f"DeepDRR画像: {drr_img.shape}")
        
        # 段階的解像度テスト
        resolutions = [0.1, 0.25, 0.5, 0.75]  # 1.0は重すぎるので除外
        results = {}
        
        for scale in resolutions:
            logger.info(f"\\n--- 解像度倍率 {scale:.2f} ---")
            
            try:
                # セットアップ
                width, height = self.setup_mask_diffdrr(resolution_scale=scale)
                
                # 投影実行
                mask_proj, proj_time = self.project_mask(alpha=0, beta=0, gamma=90.0)
                
                results[scale] = {
                    'projection': mask_proj,
                    'time': proj_time,
                    'resolution': (width, height)
                }
                
                logger.info(f"✅ 解像度倍率 {scale:.2f} 成功: {width}x{height}, {proj_time:.2f}秒")
                
                # メモリ解放
                del self.mask_drr, self.mask_subject
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ 解像度倍率 {scale:.2f} 失敗: {e}")
                results[scale] = {'error': str(e)}
                
        # 結果可視化
        if output_dir:
            self.visualize_progressive_results(drr_img, results, output_dir)
            
        return results
        
    def visualize_progressive_results(self, drr_img, results, output_dir):
        """段階的結果の可視化"""
        logger.info("段階的結果可視化作成中...")
        
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        n_results = len(successful_results)
        
        if n_results == 0:
            logger.warning("表示可能な結果がありません")
            return
            
        # 2x3レイアウト（DeepDRR + 最大5つの解像度結果）
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # DeepDRR画像表示
        ax = axes[0]
        im = ax.imshow(drr_img, cmap='bone', aspect='equal')
        ax.set_title('Original DeepDRR PA', fontsize=12, weight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 各解像度結果表示
        for i, (scale, result) in enumerate(successful_results.items()):
            if i >= 5:  # 最大5つまで表示
                break
                
            ax = axes[i + 1]
            mask_proj = result['projection']
            width, height = result['resolution']
            proj_time = result['time']
            
            # マスク表示
            mask_display = np.where(mask_proj > 0, mask_proj, np.nan)
            im = ax.imshow(mask_display, cmap='hot', aspect='equal')
            ax.set_title(f'Scale {scale:.2f}\\n{width}x{height}\\n{proj_time:.2f}s', 
                        fontsize=10, weight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        # 残りの軸を非表示
        for i in range(n_results + 1, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        save_path = output_dir / "progressive_resolution_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"段階的結果可視化保存: {save_path}")

def main():
    """段階的解像度テスト実行"""
    
    # ファイルパス
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    deepdrr_image_path = "/home/soya/ctrate_ws/deepdrr/deepdrr/outputs/deepdrr_PA_final_pa_fixed.png"
    output_dir = "/home/soya/ctrate_ws/mask_projection/outputs"
    
    try:
        projector = AdaptiveResolutionProjector(mask_path)
        results = projector.test_progressive_resolution(deepdrr_image_path, output_dir)
        
        # 結果サマリー
        logger.info("\\n=== 結果サマリー ===")
        for scale, result in results.items():
            if 'error' in result:
                logger.info(f"解像度倍率 {scale:.2f}: ❌ {result['error']}")
            else:
                width, height = result['resolution']
                time_taken = result['time']
                logger.info(f"解像度倍率 {scale:.2f}: ✅ {width}x{height}, {time_taken:.2f}秒")
                
        logger.info("段階的解像度テスト完了")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")

if __name__ == "__main__":
    main()