#!/usr/bin/env python3
"""
最小限のデバッグ用マスク投影テスト
メモリ使用量や処理時間を監視しながら問題を特定
"""
import os
import gc
import time
import psutil
import logging
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_memory_usage(prefix=""):
    """メモリ使用量をログ出力"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"{prefix} Memory: RSS={memory_info.rss/1024/1024:.1f}MB, VMS={memory_info.vms/1024/1024:.1f}MB")

def minimal_mask_projection_test():
    """最小限のマスク投影テスト"""
    logger.info("=== 最小限マスク投影デバッグテスト開始 ===")
    
    log_memory_usage("開始時")
    
    # マスクファイルパス
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    if not Path(mask_path).exists():
        logger.error(f"マスクファイルが見つかりません: {mask_path}")
        return
    
    try:
        # 1. マスク読み込み
        logger.info("ステップ1: マスク読み込み")
        start_time = time.time()
        
        mask_nib = nib.load(mask_path)
        mask_data = mask_nib.get_fdata()
        logger.info(f"マスク読み込み完了: {mask_data.shape}, unique values: {np.unique(mask_data)}")
        logger.info(f"読み込み時間: {time.time() - start_time:.2f}秒")
        log_memory_usage("マスク読み込み後")
        
        # 2. 小さいサブセットでテスト（メモリ節約）
        logger.info("ステップ2: 小さいサブセットでテスト")
        
        # 中央の小さい領域だけ取得
        h, w, d = mask_data.shape
        subset_data = mask_data[h//4:3*h//4, w//4:3*w//4, d//4:3*d//4]
        logger.info(f"サブセット形状: {subset_data.shape}")
        
        # サブセットのNiftiファイルを一時保存
        temp_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / "temp_debug_mask.nii.gz"
        
        subset_nib = nib.Nifti1Image(subset_data, mask_nib.affine)
        nib.save(subset_nib, temp_path)
        
        log_memory_usage("サブセット保存後")
        
        # 3. DiffDRRでの読み込み
        logger.info("ステップ3: DiffDRR読み込み")
        start_time = time.time()
        
        subject = read(
            volume=str(temp_path),
            orientation="PA", 
            center_volume=True
        )
        
        logger.info(f"DiffDRR読み込み完了: {time.time() - start_time:.2f}秒")
        log_memory_usage("DiffDRR読み込み後")
        
        # 4. 小さい解像度でDRR作成
        logger.info("ステップ4: 小さい解像度DRR作成")
        start_time = time.time()
        
        # 非常に小さい解像度でテスト
        small_width = 256
        small_height = 256
        
        drr = DRR(
            subject,
            sdd=1800.0,
            height=small_height,
            width=small_width, 
            delx=1.0,  # 大きいピクセルサイズ
            dely=1.0,
            reverse_x_axis=True,
            renderer="siddon"
        )
        
        logger.info(f"DRR作成完了: {small_width}x{small_height}, {time.time() - start_time:.2f}秒")
        log_memory_usage("DRR作成後")
        
        # 5. 投影実行
        logger.info("ステップ5: 投影実行")
        start_time = time.time()
        
        # シンプルなPA投影
        angles = torch.tensor([[0.0, np.pi/2, -np.pi/2]], dtype=torch.float32)
        translation = torch.zeros(1, 3, dtype=torch.float32)
        
        with torch.no_grad():
            projection = drr(
                angles, translation,
                parameterization="euler_angles", 
                convention="YZX"
            )
            
        result = projection[0, 0].cpu().numpy()
        logger.info(f"投影完了: 形状{result.shape}, 範囲{result.min():.3f}-{result.max():.3f}")
        logger.info(f"投影時間: {time.time() - start_time:.2f}秒")
        log_memory_usage("投影後")
        
        # 6. クリーンアップ
        logger.info("ステップ6: クリーンアップ")
        if temp_path.exists():
            temp_path.unlink()
        
        del projection, drr, subject, subset_data, mask_data
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        log_memory_usage("クリーンアップ後")
        
        logger.info("=== 最小限テスト完了 ===")
        return True
        
    except Exception as e:
        logger.error(f"エラー発生: {e}")
        log_memory_usage("エラー時")
        return False

if __name__ == "__main__":
    success = minimal_mask_projection_test()
    if success:
        logger.info("✅ 最小限テスト成功")
    else:
        logger.error("❌ 最小限テスト失敗")