#!/usr/bin/env python3
"""
投影問題の診断用ツール
厚み0の問題を解決するためのデバッグ機能
"""

import numpy as np
import torch
import nibabel as nib
from pathlib import Path
import imageio.v2 as imageio
import logging
from diffdrr.drr import DRR
from diffdrr.data import read

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_mask_content(mask_path):
    """
    マスクの内容を詳しく診断
    
    Args:
        mask_path: マスクファイルのパス
    """
    logger.info("=== マスク内容診断 ===")
    
    # Nibabelで直接読み込み
    mask_img = nib.load(str(mask_path))
    mask_data = mask_img.get_fdata()
    
    logger.info(f"形状: {mask_data.shape}")
    logger.info(f"データ型: {mask_data.dtype}")
    logger.info(f"値域: {mask_data.min()} ~ {mask_data.max()}")
    logger.info(f"ユニーク値: {np.unique(mask_data)}")
    
    # 非ゼロ部分の統計
    nonzero_mask = mask_data > 0
    nonzero_count = np.sum(nonzero_mask)
    total_voxels = mask_data.size
    
    logger.info(f"総ボクセル数: {total_voxels:,}")
    logger.info(f"非ゼロボクセル数: {nonzero_count:,}")
    logger.info(f"非ゼロ割合: {nonzero_count / total_voxels * 100:.3f}%")
    
    if nonzero_count > 0:
        # 非ゼロ部分の重心を計算
        indices = np.where(nonzero_mask)
        center_of_mass = np.array([
            np.mean(indices[0]),
            np.mean(indices[1]), 
            np.mean(indices[2])
        ])
        logger.info(f"マスクの重心（ボクセル座標）: {center_of_mass}")
        
        # 物理座標での重心
        physical_center = nib.affines.apply_affine(mask_img.affine, center_of_mass)
        logger.info(f"マスクの重心（物理座標 mm）: {physical_center}")
        
    return mask_data, nonzero_count > 0

def debug_subject_content(subject):
    """
    DiffDRR Subjectの内容を診断
    
    Args:
        subject: DiffDRR Subject
    """
    logger.info("=== Subject内容診断 ===")
    
    logger.info(f"Subject type: {type(subject)}")
    logger.info(f"Subject keys: {list(subject.keys()) if hasattr(subject, 'keys') else 'No keys'}")
    
    if hasattr(subject, 'volume'):
        vol = subject.volume
        logger.info(f"Volume shape: {vol.shape}")
        logger.info(f"Volume spacing: {vol.spacing}")
        logger.info(f"Volume affine:\n{vol.affine}")
        
        # テンソル内容の確認
        tensor = vol.data
        logger.info(f"Tensor shape: {tensor.shape}")
        logger.info(f"Tensor dtype: {tensor.dtype}")
        logger.info(f"Tensor device: {tensor.device}")
        logger.info(f"Tensor min: {tensor.min().item():.6f}")
        logger.info(f"Tensor max: {tensor.max().item():.6f}")
        
        # 非ゼロ要素の数
        nonzero_count = torch.sum(tensor > 0).item()
        total_elements = tensor.numel()
        logger.info(f"非ゼロ要素数: {nonzero_count:,} / {total_elements:,}")
        logger.info(f"非ゼロ割合: {nonzero_count / total_elements * 100:.3f}%")

def test_simple_projection():
    """
    より単純な設定で投影をテスト
    """
    logger.info("=== 単純投影テスト ===")
    
    # ファイルパス
    mask_path = Path("/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz")
    output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
    
    if not mask_path.exists():
        logger.error(f"マスクファイルが見つかりません: {mask_path}")
        return False
    
    # マスク内容診断
    mask_data, has_content = debug_mask_content(mask_path)
    if not has_content:
        logger.error("マスクに非ゼロ要素がありません")
        return False
    
    try:
        # DiffDRRでSubject読み込み
        logger.info("DiffDRRでSubject読み込み中...")
        subject = read(
            volume=str(mask_path),
            orientation="PA",
            center_volume=True
        )
        
        # Subject内容診断
        debug_subject_content(subject)
        
        # より大きなピクセルサイズで投影（視野を広く）
        logger.info("大きな視野で投影テスト...")
        
        drr = DRR(
            subject=subject,
            sdd=1000.0,        # より短いSDD
            height=128,        # より小さな画像
            width=128,
            delx=5.0,          # より大きなピクセルサイズ（視野を広く）
            dely=5.0,
            reverse_x_axis=True,
            renderer='siddon',
            reducefn='sum'
        )
        
        # マスクの実際の重心を使用
        mask_img = nib.load(str(mask_path))
        mask_data = mask_img.get_fdata()
        nonzero_mask = mask_data > 0
        indices = np.where(nonzero_mask)
        center_of_mass = np.array([
            np.mean(indices[0]),
            np.mean(indices[1]), 
            np.mean(indices[2])
        ])
        
        # 物理座標での重心
        physical_center = nib.affines.apply_affine(mask_img.affine, center_of_mass)
        
        logger.info(f"マスクの重心（ボクセル座標）: {center_of_mass}")
        logger.info(f"マスクの重心（物理座標）: {physical_center}")
        
        # 複数のケースを試す
        test_cases = [
            {
                "name": "マスク重心",
                "pose": [0., 0., 0.],
                "trans": physical_center.astype(np.float32)
            },
            {
                "name": "原点中心",
                "pose": [0., 0., 0.],
                "trans": np.array([0., 0., 0.], dtype=np.float32)
            },
            {
                "name": "Z方向離す",
                "pose": [0., 0., 0.],
                "trans": np.array([physical_center[0], physical_center[1], physical_center[2] + 500], dtype=np.float32)
            },
            {
                "name": "Z方向遠く",
                "pose": [0., 0., 0.],
                "trans": np.array([physical_center[0], physical_center[1], physical_center[2] - 500], dtype=np.float32)
            }
        ]
        
        best_thickness = 0
        best_case = None
        
        for case in test_cases:
            logger.info(f"\n--- テストケース: {case['name']} ---")
            logger.info(f"ポーズ: {case['pose']}")
            logger.info(f"平行移動: {case['trans']}")
            
            pose_angles = torch.tensor([case['pose']], dtype=torch.float32)
            translation = torch.tensor([case['trans']], dtype=torch.float32)
            
            try:
                with torch.no_grad():
                    thickness_proj = drr(pose_angles, translation,
                                       parameterization="euler_angles",
                                       convention="ZXY")
                
                thickness_mm = thickness_proj[0, 0].cpu().numpy()
                max_thickness = thickness_mm.max()
                nonzero_pixels = np.sum(thickness_mm > 0)
                
                logger.info(f"最大厚み: {max_thickness:.6f} mm")
                logger.info(f"非ゼロピクセル数: {nonzero_pixels}")
                
                if max_thickness > best_thickness:
                    best_thickness = max_thickness
                    best_case = case
                    
            except Exception as e:
                logger.error(f"テストケース失敗: {e}")
        
        # 最も良い結果を使用
        if best_case:
            logger.info(f"\n=== 最良のケース: {best_case['name']} ===")
            pose_angles = torch.tensor([best_case['pose']], dtype=torch.float32)
            translation = torch.tensor([best_case['trans']], dtype=torch.float32)
            
            with torch.no_grad():
                thickness_proj = drr(pose_angles, translation,
                                   parameterization="euler_angles",
                                   convention="ZXY")
        
        thickness_mm = thickness_proj[0, 0].cpu().numpy()
        
        logger.info(f"投影結果形状: {thickness_mm.shape}")
        logger.info(f"厚み値域: {thickness_mm.min():.6f} ~ {thickness_mm.max():.6f} mm")
        
        # より詳しい統計
        nonzero_pixels = thickness_mm > 0
        nonzero_count = np.sum(nonzero_pixels)
        logger.info(f"非ゼロピクセル数: {nonzero_count} / {thickness_mm.size}")
        
        if nonzero_count > 0:
            logger.info(f"非ゼロピクセルの平均厚み: {thickness_mm[nonzero_pixels].mean():.6f} mm")
            logger.info(f"非ゼロピクセルの最大厚み: {thickness_mm[nonzero_pixels].max():.6f} mm")
        
        # 結果保存
        thickness_normalized = np.clip(thickness_mm / max(thickness_mm.max(), 1e-6), 0, 1)
        thickness_u16 = (thickness_normalized * 65535).astype(np.uint16)
        
        output_file = output_dir / "debug_simple_projection.png"
        imageio.imwrite(output_file, thickness_u16)
        
        logger.info(f"結果保存: {output_file}")
        
        return thickness_mm.max() > 0
        
    except Exception as e:
        logger.error(f"テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_projection()
    if success:
        logger.info("✅ 投影成功")
    else:
        logger.error("❌ 投影失敗")