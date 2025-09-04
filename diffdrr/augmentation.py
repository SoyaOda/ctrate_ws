#!/usr/bin/env python3
"""
Step 2: 3Dデータ拡張機能
TorchIOを使用して3D医用画像に対する様々な拡張を適用する
"""

import torchio as tio
from typing import Optional, Dict, Any, List
import numpy as np


def get_augmentation_transform(
    enable_random_affine: bool = True,
    enable_random_elastic: bool = True,
    enable_random_noise: bool = True,
    enable_random_blur: bool = True,
    enable_random_bias_field: bool = True,
    enable_random_gamma: bool = True,
    probability: float = 0.5
) -> tio.Compose:
    """
    3Dデータ拡張のための変換パイプラインを定義する。
    
    Args:
        enable_random_affine: ランダムアフィン変換を有効にするか
        enable_random_elastic: ランダム弾性変形を有効にするか
        enable_random_noise: ランダムノイズを有効にするか  
        enable_random_blur: ランダムブラーを有効にするか
        enable_random_bias_field: ランダムバイアスフィールドを有効にするか
        enable_random_gamma: ランダムガンマ補正を有効にするか
        probability: 各拡張を適用する確率
    
    Returns:
        tio.Compose: 拡張変換のパイプライン
    """
    transforms = []
    
    # 空間変換（CTとマスクの両方に適用）
    if enable_random_affine:
        transforms.append(
            tio.RandomAffine(
                scales=(0.95, 1.05),  # スケーリング範囲
                degrees=10,  # 回転角度範囲（度）
                translation=20,  # 平行移動範囲（mm）
                isotropic=False,  # 非等方的スケーリングを許可
                center='image',  # 画像中心を基準に変換
                default_pad_value='minimum',  # パディング値
                p=probability
            )
        )
    
    if enable_random_elastic:
        transforms.append(
            tio.RandomElasticDeformation(
                num_control_points=7,  # 制御点の数
                max_displacement=20,  # 最大変位（mm）
                locked_borders=2,  # 境界をロック
                p=probability
            )
        )
    
    # 強度変換（CTのみに適用）
    if enable_random_noise:
        transforms.append(
            tio.RandomNoise(
                mean=0,
                std=(0, 25),  # ノイズの標準偏差範囲
                include=['ct'],  # CTのみに適用
                p=probability
            )
        )
    
    if enable_random_blur:
        transforms.append(
            tio.RandomBlur(
                std=(0, 1),  # ブラーの標準偏差範囲
                include=['ct'],  # CTのみに適用
                p=probability
            )
        )
    
    if enable_random_bias_field:
        transforms.append(
            tio.RandomBiasField(
                coefficients=0.3,  # バイアスフィールドの係数
                order=3,  # 多項式の次数
                include=['ct'],  # CTのみに適用
                p=probability
            )
        )
    
    if enable_random_gamma:
        transforms.append(
            tio.RandomGamma(
                log_gamma=(-0.3, 0.3),  # ガンマ値の対数範囲
                include=['ct'],  # CTのみに適用
                p=probability
            )
        )
    
    # すべての変換を組み合わせる
    if transforms:
        augment = tio.Compose(transforms)
    else:
        # 拡張なし（恒等変換）
        augment = tio.Compose([])
    
    return augment


def get_minimal_augmentation() -> tio.Compose:
    """
    最小限の拡張（軽い回転と平行移動のみ）を返す
    
    Returns:
        tio.Compose: 最小限の拡張パイプライン
    """
    return tio.Compose([
        tio.RandomAffine(
            scales=(0.98, 1.02),
            degrees=5,
            translation=10,
            p=0.8
        )
    ])


def get_moderate_augmentation() -> tio.Compose:
    """
    中程度の拡張を返す
    
    Returns:
        tio.Compose: 中程度の拡張パイプライン
    """
    return tio.Compose([
        tio.RandomAffine(
            scales=(0.95, 1.05),
            degrees=10,
            translation=20,
            p=0.5
        ),
        tio.RandomNoise(
            std=(0, 15),
            include=['ct'],
            p=0.3
        ),
        tio.RandomBlur(
            std=(0, 0.5),
            include=['ct'],
            p=0.3
        )
    ])


def get_aggressive_augmentation() -> tio.Compose:
    """
    強い拡張を返す（データセットが小さい場合に有用）
    
    Returns:
        tio.Compose: 強い拡張パイプライン
    """
    return get_augmentation_transform(
        enable_random_affine=True,
        enable_random_elastic=True,
        enable_random_noise=True,
        enable_random_blur=True,
        enable_random_bias_field=True,
        enable_random_gamma=True,
        probability=0.8
    )


def apply_augmentation(subject: tio.Subject, augmentation: tio.Compose) -> tio.Subject:
    """
    Subjectに拡張を適用する
    
    Args:
        subject: TorchIOのSubjectオブジェクト
        augmentation: 適用する拡張パイプライン
    
    Returns:
        tio.Subject: 拡張後のSubjectオブジェクト
    """
    return augmentation(subject)


def test_augmentation():
    """
    拡張機能をテストする
    """
    from load_data import load_nifti_data, print_subject_info
    
    # データを読み込み
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    print("Loading original data...")
    subject = load_nifti_data(ct_path, mask_path)
    
    # 最小限の拡張のみテスト（高速化のため）
    augmentation_levels = {
        "Minimal": get_minimal_augmentation(),
    }
    
    for level_name, augmentation in augmentation_levels.items():
        print(f"\n{'='*50}")
        print(f"Testing {level_name} Augmentation")
        print('='*50)
        
        # 拡張を適用
        augmented_subject = apply_augmentation(subject, augmentation)
        
        # 情報を表示
        print(f"\n--- {level_name} Augmentation Applied ---")
        print(f"CT shape: {augmented_subject.ct.shape}")
        print(f"CT value range: [{augmented_subject.ct.data.min():.2f}, {augmented_subject.ct.data.max():.2f}]")
        print(f"Mask shape: {augmented_subject.mask.shape}")
        print(f"Mask unique values: {augmented_subject.mask.data.unique().tolist()}")
        
        # 元のデータとの差分を計算
        if subject.ct.shape == augmented_subject.ct.shape:
            ct_diff = (augmented_subject.ct.data.float() - subject.ct.data.float()).abs().mean()
            mask_diff = (augmented_subject.mask.data.float() - subject.mask.data.float()).abs().mean()
            print(f"Mean absolute CT difference: {ct_diff:.4f}")
            print(f"Mean absolute mask difference: {mask_diff:.4f}")
        else:
            print("Shape changed due to augmentation")
    
    print("\n✓ Augmentation test completed successfully!")


if __name__ == "__main__":
    # テスト実行
    test_augmentation()