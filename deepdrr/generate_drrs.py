#!/usr/bin/env python3
"""
Step 4: 完全なパイプラインスクリプト
md_files/deepdrr.mdの方針に従った実装
TorchIOによる3Dデータ拡張とDiffDRRによる2D投影を統合
"""

import torch
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from diffdrr.drr import DRR
from diffdrr.pose import convert
from typing import Tuple, Optional


class DRRPipeline:
    """DRR生成パイプライン"""
    
    def __init__(self, 
                 ct_path: str,
                 mask_path: str,
                 output_dir: str = "./augmented_data",
                 device: str = "cuda"):
        """
        初期化
        
        Args:
            ct_path: CT画像のNIfTIファイルパス
            mask_path: マスク画像のNIfTIファイルパス
            output_dir: 出力ディレクトリ
            device: 計算デバイス
        """
        self.ct_path = ct_path
        self.mask_path = mask_path
        self.output_dir = Path(output_dir)
        
        # デバイス設定
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = device
        
        # 出力ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "drr").mkdir(exist_ok=True)
        (self.output_dir / "mask").mkdir(exist_ok=True)
        (self.output_dir / "visualization").mkdir(exist_ok=True)
        
        # データを読み込み
        print("Loading NIfTI data...")
        self.initial_subject = self.load_nifti_data()
        
        # 拡張パイプラインを準備
        self.augmentation_pipeline = self.get_augmentation_transform()
    
    def load_nifti_data(self) -> tio.Subject:
        """
        NIfTI形式のCTとマスクファイルを読み込み
        
        Returns:
            tio.Subject: CTとマスクを格納したSubjectオブジェクト
        """
        subject = tio.Subject(
            ct=tio.ScalarImage(self.ct_path),
            mask=tio.LabelMap(self.mask_path),
        )
        return subject
    
    def get_augmentation_transform(self) -> tio.Compose:
        """
        3Dデータ拡張のための変換パイプラインを定義
        ★ 安定性のためにtranslationを小さく制限
        
        Returns:
            tio.Compose: 拡張変換のパイプライン
        """
        return tio.Compose([
            # 空間変換（CTとマスクの両方に適用）
            tio.RandomAffine(
                scales=(0.95, 1.05),
                degrees=10,
                translation=5,  # 20から5に削減して安定性を向上
                isotropic=False,
                center='image',
                default_pad_value='minimum',
                p=0.5
            ),
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=10,  # 20から10に削減
                locked_borders=2,
                p=0.3
            ),
            # 強度変換（CTのみに適用）
            tio.RandomNoise(
                mean=0,
                std=(0, 25),
                include=['ct'],
                p=0.3
            ),
            tio.RandomBlur(
                std=(0, 1),
                include=['ct'],
                p=0.3
            ),
        ])
    
    def setup_drr_generator(self, subject: tio.Subject) -> DRR:
        """
        TorchIO SubjectからDRRジェネレータを正しくセットアップする。
        ★ 胸部X線らしい画像を生成するための最適化版
        
        Args:
            subject: TorchIOのSubjectオブジェクト
            
        Returns:
            DRR: 初期化されたDRRジェネレータ
        """
        # Step 1: TorchIOから必要なデータを抽出
        ct_tensor_hu = subject.ct.data.squeeze(0)  # (D, H, W)
        spacing_xyz = subject.ct.spacing  # (sx, sy, sz)
        affine_matrix = subject.ct.affine  # 4x4 numpy array
        
        # Step 2: HU値を線減弱係数(LAC)に変換
        # より現実的な変換式を使用
        volume_clipped = torch.clamp(ct_tensor_hu, -1024, 3071)
        
        # 組織ごとの減弱係数を考慮したより精密な変換
        mu_water = 0.0192  # [mm^-1] at 60 keV
        mu_air = 0.0004    # [mm^-1]
        mu_bone = 0.0480   # [mm^-1] 骨の減弱係数
        
        # HU値に基づく減弱係数の計算
        # 空気: HU = -1000, 水: HU = 0, 骨: HU = 1000
        volume_lac = torch.zeros_like(volume_clipped)
        
        # 空気領域 (HU < -500)
        air_mask = volume_clipped < -500
        volume_lac[air_mask] = mu_air
        
        # 軟部組織領域 (-500 <= HU <= 200)
        soft_mask = (volume_clipped >= -500) & (volume_clipped <= 200)
        volume_lac[soft_mask] = (volume_clipped[soft_mask] + 1024) * (mu_water / 1000.0)
        
        # 骨領域 (HU > 200) - 骨を強調
        bone_mask = volume_clipped > 200
        # 骨の減弱係数を強調（1.5倍）
        volume_lac[bone_mask] = (volume_clipped[bone_mask] + 1024) * (mu_bone / 1000.0) * 1.5
        
        # デバッグ: LAC変換後の統計情報
        print(f"\nDRR Volume Statistics (after LAC conversion):")
        print(f"  HU range: [{ct_tensor_hu.min().item():.1f}, {ct_tensor_hu.max().item():.1f}]")
        print(f"  LAC range: [{volume_lac.min().item():.6f}, {volume_lac.max().item():.6f}]")
        print(f"  Shape: {volume_lac.shape}")
        print(f"  Spacing: {spacing_xyz}")
        
        # Step 3: DiffDRRに適した形式で4Dテンソルを準備
        volume_4d = volume_lac.unsqueeze(0).to(self.device)  # (1, D, H, W)
        
        # Step 4: DiffDRRSubjectを直接作成（read()関数の処理を回避）
        diffdrr_subject = tio.Subject(
            volume=tio.ScalarImage(tensor=volume_4d, affine=affine_matrix),
            density=tio.ScalarImage(tensor=volume_4d, affine=affine_matrix),  # volumeと同じ
        )
        
        # DiffDRRが期待する追加属性を設定
        # __dict__に直接設定して読み取り専用プロパティを回避
        diffdrr_subject.__dict__['spacing'] = spacing_xyz
        diffdrr_subject.__dict__['reorient'] = torch.eye(4, dtype=torch.float32).to(self.device)
        diffdrr_subject.__dict__['mask'] = None
        diffdrr_subject.__dict__['fiducials'] = None
        
        # Step 5: DRRジェネレータを初期化（胸部X線に最適化）
        drr_generator = DRR(
            subject=diffdrr_subject,
            sdd=1800.0,      # Source-to-detector distance (standard chest X-ray)
            height=512,      # 検出器の高さ（ピクセル）
            width=512,       # 検出器の幅（ピクセル）
            delx=0.7,        # ピクセルサイズX（mm）
            dely=0.7,        # ピクセルサイズY（mm）
        ).to(self.device)
        
        return drr_generator
    
    def get_random_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ランダムなカメラポーズを生成
        ★ Y軸のベースライン位置を-100mmに設定して胸部全体を撮影
        
        Returns:
            rotations: 回転角度（度）
            translations: 平行移動（mm）
        """
        # 回転: 各軸±5度の範囲（わずかなバリエーション）
        rotations = (torch.rand(1, 3) - 0.5) * 10.0  # 度数法
        
        # 平行移動: Y軸のベースラインを調整
        translations = torch.zeros(1, 3)
        translations[:, 0] = (torch.rand(1) - 0.5) * 30.0  # X軸: ±15mm
        translations[:, 1] = -100.0 + (torch.rand(1) - 0.5) * 30.0  # Y軸: -115mm～-85mm（胸部が見える位置）
        translations[:, 2] = (torch.rand(1) - 0.5) * 30.0  # Z軸: ±15mm
        
        return rotations, translations
    
    def setup_mask_drr_generator(self, subject: tio.Subject) -> DRR:
        """
        マスク投影用のDRRジェネレータをセットアップする。
        ★ read()関数を使わず、直接制御する安定した方法
        
        Args:
            subject: TorchIOのSubjectオブジェクト
            
        Returns:
            DRR: マスク用のDRRジェネレータ
        """
        # マスクデータを抽出してfloat型に変換
        mask_tensor = subject.mask.data.squeeze(0).float()  # (D, H, W)
        spacing_xyz = subject.mask.spacing  # (sx, sy, sz)
        affine_matrix = subject.mask.affine  # 4x4 numpy array
        
        # デバッグ: マスクデータの統計情報
        print(f"\nMask Volume Statistics:")
        print(f"  Shape: {mask_tensor.shape}")
        print(f"  Range: [{mask_tensor.min().item():.1f}, {mask_tensor.max().item():.1f}]")
        print(f"  Spacing: {spacing_xyz}")
        
        # マスクを4Dテンソルに変換
        mask_4d = mask_tensor.unsqueeze(0).to(self.device)  # (1, D, H, W)
        
        # DiffDRRSubjectを直接作成
        diffdrr_subject = tio.Subject(
            volume=tio.ScalarImage(tensor=mask_4d, affine=affine_matrix),
            density=tio.ScalarImage(tensor=mask_4d, affine=affine_matrix),  # volumeと同じ
        )
        
        # DiffDRRが期待する追加属性を設定
        # __dict__に直接設定して読み取り専用プロパティを回避
        diffdrr_subject.__dict__['spacing'] = spacing_xyz
        diffdrr_subject.__dict__['reorient'] = torch.eye(4, dtype=torch.float32).to(self.device)
        diffdrr_subject.__dict__['mask'] = None
        diffdrr_subject.__dict__['fiducials'] = None
        
        # マスク用DRRジェネレータを作成
        mask_drr_generator = DRR(
            subject=diffdrr_subject,
            sdd=1800.0,
            height=512,
            width=512,
            delx=0.7,
            dely=0.7,
        ).to(self.device)
        
        return mask_drr_generator
    
    def project_mask(self, subject: tio.Subject, rotations: torch.Tensor, 
                    translations: torch.Tensor) -> torch.Tensor:
        """
        マスクを投影して厚みマップを生成
        
        Args:
            subject: TorchIOのSubjectオブジェクト
            rotations: 回転角度（度）
            translations: 平行移動（mm）
            
        Returns:
            thickness_map: 厚みマップ
        """
        # マスク用のDRRジェネレータをセットアップ
        mask_drr_generator = self.setup_mask_drr_generator(subject)
        
        # RigidTransformを作成
        rotations_rad = torch.deg2rad(rotations).to(self.device)
        translations = translations.to(self.device)
        transform = convert(
            rotations_rad,
            translations,
            parameterization="euler_angles",
            convention="XYZ",
        ).to(self.device)
        
        # 投影
        thickness_map = mask_drr_generator(transform)
        
        return thickness_map
    
    def generate_sample(self, sample_id: int):
        """
        1サンプルのDRRと体積マスクを生成
        
        Args:
            sample_id: サンプルID
        """
        # 3D拡張を適用
        augmented_subject = self.augmentation_pipeline(self.initial_subject)
        
        # DiffDRRジェネレータをセットアップ
        drr_generator = self.setup_drr_generator(augmented_subject)
        
        # ランダムなカメラポーズを生成
        rotations, translations = self.get_random_pose()
        
        # RigidTransformを作成
        rotations_rad = torch.deg2rad(rotations).to(self.device)
        translations = translations.to(self.device)
        transform = convert(
            rotations_rad,
            translations,
            parameterization="euler_angles",
            convention="XYZ",
        ).to(self.device)
        
        # DRRを生成
        drr_image = drr_generator(transform)
        
        # デバッグ: DRR画像の値を確認
        if sample_id == 0:
            print(f"\nDRR Image Statistics:")
            print(f"  Shape: {drr_image.shape}")
            print(f"  Min: {drr_image.min().item():.4f}")
            print(f"  Max: {drr_image.max().item():.4f}")
            print(f"  Mean: {drr_image.mean().item():.4f}")
            print(f"  Std: {drr_image.std().item():.4f}")
        
        # マスクの厚みマップを生成
        thickness_map = self.project_mask(augmented_subject, rotations, translations)
        
        # 定量的スケーリング（厚み -> 体積）
        pixel_area = 0.7 * 0.7  # delx * dely
        volumetric_mask = thickness_map * pixel_area
        
        # 検証（最初のサンプルのみ）
        if sample_id == 0:
            mask_tensor = augmented_subject.mask.data
            voxel_volume = torch.prod(torch.tensor(augmented_subject.ct.spacing))
            original_volume_3d = torch.sum(mask_tensor) * voxel_volume
            projected_volume_2d = torch.sum(volumetric_mask)
            
            print(f"\n=== Volume Validation ===")
            print(f"Original 3D Mask Volume: {original_volume_3d.item():.2f} mm³")
            print(f"Projected 2D Mask Volume Sum: {projected_volume_2d.item():.2f} mm³")
            print(f"Note: 完全な一致は期待できません（投影による情報損失のため）")
        
        # 結果を保存
        self.save_results(drr_image, volumetric_mask, sample_id)
    
    def save_results(self, drr_image: torch.Tensor, volumetric_mask: torch.Tensor, 
                    sample_id: int):
        """
        結果を保存
        
        Args:
            drr_image: DRR画像
            volumetric_mask: 体積マスク
            sample_id: サンプルID
        """
        # TensorをCPUに移動し、Numpy配列に変換
        drr_np = drr_image.squeeze().cpu().numpy()
        mask_np = volumetric_mask.squeeze().cpu().numpy()
        
        # DRR画像を胸部X線らしく処理
        # 1. 対数変換（X線の物理に基づく）
        drr_log = np.log(drr_np + 1e-6)
        
        # 2. 正規化
        drr_normalized = (drr_log - drr_log.min()) / (drr_log.max() - drr_log.min() + 1e-8)
        
        # 3. ガンマ補正（コントラスト強調）
        gamma = 0.4  # より強いコントラスト
        drr_gamma = np.power(drr_normalized, gamma)
        
        # 4. 反転（X線では高密度領域が白く見える）
        drr_inverted = 1.0 - drr_gamma
        
        plt.imsave(
            self.output_dir / "drr" / f"sample_{sample_id:04d}.png",
            drr_inverted,
            cmap='gray'
        )
        
        # 体積マスクを保存（NPY）
        np.save(
            self.output_dir / "mask" / f"sample_{sample_id:04d}.npy",
            mask_np
        )
        
        # 可視化画像を生成（最初の10サンプル）
        if sample_id < 10:
            self.create_visualization(drr_inverted, mask_np, sample_id)
    
    def create_visualization(self, drr: np.ndarray, mask: np.ndarray, sample_id: int):
        """
        可視化画像を作成
        
        Args:
            drr: DRR画像
            mask: 体積マスク
            sample_id: サンプルID
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # DRR画像
        axes[0].imshow(drr, cmap='gray')
        axes[0].set_title('DRR Image')
        axes[0].axis('off')
        
        # 体積マスク
        im = axes[1].imshow(mask, cmap='hot')
        axes[1].set_title('Volumetric Mask (mm³ per pixel)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Volume (mm³)')
        
        plt.suptitle(f'Sample {sample_id:04d}')
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualization" / f"sample_{sample_id:04d}.png")
        plt.close()
    
    def generate_dataset(self, num_samples: int = 10):
        """
        データセットを生成
        
        Args:
            num_samples: 生成するサンプル数
        """
        print(f"\n=== Generating {num_samples} samples ===")
        print(f"Output directory: {self.output_dir}")
        
        for i in tqdm(range(num_samples), desc="Generating Samples"):
            self.generate_sample(i)
        
        print(f"\n✓ Successfully generated {num_samples} samples")
        print(f"  - DRR images: {self.output_dir}/drr/")
        print(f"  - Volumetric masks: {self.output_dir}/mask/")
        print(f"  - Visualizations: {self.output_dir}/visualization/")


def main():
    """メイン関数"""
    # 設定
    CT_NIFTI_PATH = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    MASK_NIFTI_PATH = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    OUTPUT_DIR = "./augmented_data"
    NUM_SAMPLES = 2
    
    # デバイス設定
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # パイプラインを初期化
    pipeline = DRRPipeline(
        ct_path=CT_NIFTI_PATH,
        mask_path=MASK_NIFTI_PATH,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )
    
    # データセットを生成
    pipeline.generate_dataset(num_samples=NUM_SAMPLES)
    
    print("\n=== Pipeline completed successfully! ===")
    print("このパイプラインにより、以下が実現されました：")
    print("1. TorchIOによる3Dボリュームの解剖学的多様性のシミュレーション")
    print("2. DiffDRRによる動的なカメラポーズでの2D投影")
    print("3. 定量的な体積情報を保持した2Dマスクの生成")


if __name__ == "__main__":
    main()