#!/usr/bin/env python3
"""
SimpleITKを使用したNIfTIデータの読み込み（DiffDRR対応版）
"""

import SimpleITK as sitk
import torch
import numpy as np
from typing import Tuple


def load_nifti_data_sitk(ct_path: str, mask_path: str = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NIfTIファイルからCTデータとメタ情報を読み込む

    Args:
        ct_path (str): CT NIfTIファイルのパス
        mask_path (str): マスクファイルのパス（オプション）

    Returns:
        tuple: (ct_tensor, mask_tensor, spacing_tensor)
            - ct_tensor (torch.Tensor): CTボリューム (HU単位)
            - mask_tensor (torch.Tensor): マスクボリューム（Noneの場合あり）
            - spacing_tensor (torch.Tensor): ボクセル間隔 [x, y, z] (mm)
    """
    # SimpleITKでNIfTIファイルを読み込む
    sitk_img = sitk.ReadImage(ct_path)

    # NumPy配列としてボリュームデータを取得
    # SimpleITKの軸順序は (x, y, z)
    volume_np = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    # PyTorchのテンソルは (D, H, W) の順序になるため、(z, y, x) に対応
    # 軸順序を (z, y, x) -> (x, y, z) の順に並び替える
    volume_np = np.transpose(volume_np, (2, 1, 0))

    # ボクセル間隔を取得 (x, y, z)
    spacing_np = np.array(sitk_img.GetSpacing(), dtype=np.float32)

    # PyTorchテンソルに変換
    ct_tensor = torch.from_numpy(volume_np)
    spacing_tensor = torch.from_numpy(spacing_np)
    
    # マスクを読み込む（提供されている場合）
    mask_tensor = None
    if mask_path:
        sitk_mask = sitk.ReadImage(mask_path)
        mask_np = sitk.GetArrayFromImage(sitk_mask).astype(np.float32)
        mask_np = np.transpose(mask_np, (2, 1, 0))
        mask_tensor = torch.from_numpy(mask_np)

    return ct_tensor, mask_tensor, spacing_tensor


def test_load():
    """テスト関数"""
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    print("Loading data with SimpleITK...")
    ct, mask, spacing = load_nifti_data_sitk(ct_path, mask_path)
    
    print(f"CT shape: {ct.shape}")
    print(f"CT range: [{ct.min():.1f}, {ct.max():.1f}]")
    print(f"Mask shape: {mask.shape if mask is not None else 'None'}")
    print(f"Spacing: {spacing.tolist()}")
    

if __name__ == "__main__":
    test_load()