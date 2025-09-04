#!/usr/bin/env python3
"""
Step 1: NIfTI データの読み込み機能
TorchIOを使用してCTとマスクのNIfTIファイルを読み込み、管理する
"""

import torchio as tio
import torch
from pathlib import Path
from typing import Optional, Dict, Any


def load_nifti_data(ct_path: str, mask_path: str) -> tio.Subject:
    """
    NIfTI形式のCTとマスクファイルを読み込み、TorchIOのSubjectオブジェクトを作成する。
    
    Args:
        ct_path (str): CT画像のファイルパス
        mask_path (str): マスク画像のファイルパス
    
    Returns:
        tio.Subject: CTとマスクを格納したSubjectオブジェクト
    
    Raises:
        FileNotFoundError: 指定されたファイルが存在しない場合
    """
    # ファイルの存在確認
    ct_path = Path(ct_path)
    mask_path = Path(mask_path)
    
    if not ct_path.exists():
        raise FileNotFoundError(f"CT file not found: {ct_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # TorchIOのSubjectオブジェクトを作成
    subject = tio.Subject(
        ct=tio.ScalarImage(str(ct_path)),
        mask=tio.LabelMap(str(mask_path)),
    )
    
    return subject


def get_subject_info(subject: tio.Subject) -> Dict[str, Any]:
    """
    Subjectオブジェクトから情報を抽出して辞書形式で返す
    
    Args:
        subject (tio.Subject): TorchIOのSubjectオブジェクト
    
    Returns:
        Dict[str, Any]: 画像情報を含む辞書
    """
    info = {
        "ct": {
            "shape": subject.ct.shape,
            "spacing": subject.ct.spacing,
            "origin": subject.ct.origin,
            "orientation": subject.ct.orientation,
            "dtype": str(subject.ct.data.dtype),
            "min": float(subject.ct.data.min()),
            "max": float(subject.ct.data.max()),
        },
        "mask": {
            "shape": subject.mask.shape,
            "spacing": subject.mask.spacing,
            "origin": subject.mask.origin,
            "orientation": subject.mask.orientation,
            "dtype": str(subject.mask.data.dtype),
            "unique_values": subject.mask.data.unique().tolist(),
            "min": float(subject.mask.data.min()),
            "max": float(subject.mask.data.max()),
        }
    }
    
    return info


def print_subject_info(subject: tio.Subject) -> None:
    """
    Subjectオブジェクトの情報をコンソールに表示する
    
    Args:
        subject (tio.Subject): TorchIOのSubjectオブジェクト
    """
    print("=== Subject Information ===")
    print(subject)
    
    print("\n=== CT Image ===")
    print(f"Shape: {subject.ct.shape}")
    print(f"Spacing (mm): {subject.ct.spacing}")
    print(f"Origin: {subject.ct.origin}")
    print(f"Orientation: {subject.ct.orientation}")
    print(f"Data type: {subject.ct.data.dtype}")
    print(f"Value range: [{subject.ct.data.min():.2f}, {subject.ct.data.max():.2f}]")
    
    print("\n=== Mask Image ===")
    print(f"Shape: {subject.mask.shape}")
    print(f"Spacing (mm): {subject.mask.spacing}")
    print(f"Origin: {subject.mask.origin}")
    print(f"Orientation: {subject.mask.orientation}")
    print(f"Data type: {subject.mask.data.dtype}")
    print(f"Unique values: {subject.mask.data.unique().tolist()}")
    print(f"Value range: [{subject.mask.data.min():.2f}, {subject.mask.data.max():.2f}]")
    
    # メモリ使用量の推定
    ct_memory = subject.ct.data.element_size() * subject.ct.data.nelement() / (1024 ** 2)
    mask_memory = subject.mask.data.element_size() * subject.mask.data.nelement() / (1024 ** 2)
    print(f"\n=== Memory Usage ===")
    print(f"CT: {ct_memory:.2f} MB")
    print(f"Mask: {mask_memory:.2f} MB")
    print(f"Total: {ct_memory + mask_memory:.2f} MB")


# テスト用の関数
def test_load_data():
    """
    実際のデータでload_nifti_data関数をテストする
    """
    # 実際のデータパス
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    print("Loading NIfTI data...")
    try:
        subject = load_nifti_data(ct_path, mask_path)
        print("✓ Data loaded successfully!\n")
        
        # 情報を表示
        print_subject_info(subject)
        
        # アフィン行列の取得
        print("\n=== Affine Matrix ===")
        print("CT Affine:")
        print(subject.ct.affine)
        print("\nMask Affine:")
        print(subject.mask.affine)
        
        return subject
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


if __name__ == "__main__":
    # テスト実行
    subject = test_load_data()