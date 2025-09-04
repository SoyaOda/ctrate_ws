#!/usr/bin/env python3
"""
高忠実度DRR生成スクリプト
gemini1.mdの推奨事項に基づいた実装
- ToCanonicalによるRAS+座標系への標準化
- 区分的線形モデルによる物理的に正確なHU-LAC変換
- 適切なPAビューカメラポーズ設定
"""

import torch
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffdrr.drr import DRR
from diffdrr.pose import convert
import tempfile
import nibabel as nib


def preprocess_ct_volume(ct_path: str) -> tio.Subject:
    """
    NIfTI形式のCTボリュームを読み込み、DiffDRRのための標準化前処理を適用する
    
    1. TorchIO Subjectとして読み込む
    2. ToCanonical変換を適用し、RAS+座標系に標準化する
    3. HU値をクリッピングし、物理的に妥当なLAC値に変換する
    4. LACボリュームで元のボリュームを置き換える
    """
    print(f"Loading CT from: {ct_path}")
    
    # TorchIO Subjectとして画像を読み込む
    subject = tio.Subject(
        ct=tio.ScalarImage(ct_path)
    )
    
    print(f"Original shape: {subject.ct.shape}")
    print(f"Original affine:\n{subject.ct.affine}")
    
    # 1. 座標系の標準化：ToCanonical変換
    # これがアーティファクトを除去し、カメラポーズを予測可能にするための最重要ステップ
    print("\nApplying ToCanonical transform to standardize to RAS+ coordinate system...")
    to_canonical = tio.ToCanonical()
    subject_canonical = to_canonical(subject)
    
    print(f"Canonical shape: {subject_canonical.ct.shape}")
    print(f"Canonical affine:\n{subject_canonical.ct.affine}")
    
    # 2. HUからLACへの変換
    volume_hu = subject_canonical.ct.data.squeeze()
    print(f"\nHU range: [{volume_hu.min():.1f}, {volume_hu.max():.1f}]")
    
    # HU値を物理的に妥当な範囲にクリッピング
    volume_clipped = torch.clamp(volume_hu, -1024, 3071)
    
    # 区分的線形モデルによるHU-LAC変換
    # 70keVのX線エネルギーを想定した場合の水の線減弱係数
    mu_water = 0.02  # mm^-1
    
    # 骨のスケーリング係数（経験的に3-5倍が良好なコントラストを生む）
    bone_scaling_factor = 3.0
    
    # HU <= 0 の領域 (軟部組織)
    lac_soft_tissue = mu_water * (1.0 + volume_clipped / 1000.0)
    
    # HU > 0 の領域 (骨組織)
    lac_bone = mu_water + (volume_clipped / 1000.0) * (mu_water * (bone_scaling_factor - 1.0))
    
    # 2つの領域を結合して最終的なLACボリュームを作成
    volume_lac = torch.where(volume_clipped > 0, lac_bone, lac_soft_tissue)
    
    # 空気以下の領域（HU < -1000）は減衰ゼロとする
    volume_lac[volume_clipped < -1000] = 0.0
    
    print(f"LAC range: [{volume_lac.min():.4f}, {volume_lac.max():.4f}] mm^-1")
    
    # 3. TorchIO Subjectのボリュームを更新
    subject_canonical.ct = tio.ScalarImage(
        tensor=volume_lac.unsqueeze(0),
        affine=subject_canonical.ct.affine
    )
    
    return subject_canonical


def save_subject_for_diffdrr(subject: tio.Subject) -> str:
    """
    DiffDRRのread()関数で読み込めるように一時ファイルとして保存
    """
    temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    
    # Get data and affine, handle both tensor and numpy types
    data = subject.ct.data.squeeze()
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    
    affine = subject.ct.affine
    if torch.is_tensor(affine):
        affine = affine.cpu().numpy()
    
    # NIfTI形式で保存
    nifti_image = nib.Nifti1Image(data, affine)
    nib.save(nifti_image, temp_file.name)
    
    return temp_file.name


def create_drr_generator_direct(subject: tio.Subject, device: str = "cuda") -> tuple:
    """
    前処理済みのLACボリュームから直接DRRジェネレータを作成
    実際にはtempファイル経由のみが動作するため、この関数は使用されない
    """
    # This approach doesn't work with current DiffDRR version
    # Always fall back to temp file method
    raise NotImplementedError("Direct creation not supported, use temp file method")


def generate_chest_xray_views(drr: DRR, sdd: float, output_dir: Path, device: str):
    """
    複数の臨床的ビューのDRRを生成
    """
    # ビュー定義
    views = [
        {
            'name': 'PA_standard',
            'rotation': [0.0, 0.0, 0.0],
            'translation': [0.0, -sdd/2, 0.0],  # Y軸負方向でPA view
            'description': 'Standard PA chest X-ray'
        },
        {
            'name': 'PA_slight_rotation',
            'rotation': [0.0, 0.0, 5.0],  # 5度のZ軸回転
            'translation': [0.0, -sdd/2, 0.0],
            'description': 'PA with 5° rotation'
        },
        {
            'name': 'LAT',
            'rotation': [0.0, 90.0, 0.0],  # Y軸周りに90度回転
            'translation': [-sdd/2, 0.0, 0.0],  # X軸負方向で側面
            'description': 'Lateral view'
        },
    ]
    
    results = []
    
    for view in views:
        print(f"\nGenerating {view['name']} view...")
        print(f"  {view['description']}")
        
        # Create pose
        rotations_deg = torch.tensor([view['rotation']], dtype=torch.float32)
        translations = torch.tensor([view['translation']], dtype=torch.float32)
        
        # Convert to radians and move to device
        rotations_rad = torch.deg2rad(rotations_deg).to(device)
        translations = translations.to(device)
        
        # Create transform
        transform = convert(
            rotations_rad,
            translations,
            parameterization="euler_angles",
            convention="XYZ",
        ).to(device)
        
        # Generate DRR (transmission image)
        img_transmission = drr(transform)
        
        # Post-process for high-fidelity visualization
        # Clamp to physical range [0, 1]
        img_clipped = torch.clamp(img_transmission, 1e-6, 1.0)
        
        # Convert to attenuation (log transform)
        # -log(I/I_0) = integral(mu*dl)
        img_attenuation = -torch.log(img_clipped)
        
        # Convert to numpy and normalize for display
        img_display = img_attenuation.squeeze().cpu().numpy()
        if img_display.max() > img_display.min():
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        
        print(f"  Image stats - Min: {img_display.min():.3f}, Max: {img_display.max():.3f}")
        
        # Save image
        img_path = output_dir / f"{view['name']}.png"
        plt.imsave(img_path, img_display, cmap='gray')
        print(f"  Saved to: {img_path}")
        
        results.append({
            'name': view['name'],
            'description': view['description'],
            'image': img_display,
            'rotation': view['rotation'],
            'translation': view['translation']
        })
    
    return results


def create_comparison_figure(results, output_dir):
    """
    全ビューの比較図を作成
    """
    n_views = len(results)
    fig, axes = plt.subplots(1, n_views, figsize=(6*n_views, 6))
    
    if n_views == 1:
        axes = [axes]
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        ax.imshow(result['image'], cmap='gray')
        ax.set_title(f"{result['name']}\n{result['description']}", fontsize=10)
        ax.axis('off')
        
        # Add pose info
        rot_str = f"R:{result['rotation']}"
        trans_str = f"T:{[int(t) for t in result['translation']]}"
        ax.text(0.5, -0.05, f"{rot_str}\n{trans_str}", 
                transform=ax.transAxes, ha='center', fontsize=8)
    
    plt.tight_layout()
    comparison_path = output_dir / "all_views_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nComparison figure saved to: {comparison_path}")


def main():
    """
    メイン実行関数
    """
    # Paths
    CT_PATH = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    OUTPUT_DIR = Path("./drr_canonical_output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("High-Fidelity DRR Generation with Canonical Preprocessing")
    print("="*60)
    
    # Step 1: Preprocess CT volume
    print("\n[Step 1] Preprocessing CT volume...")
    subject_processed = preprocess_ct_volume(CT_PATH)
    
    # Step 2: Create DRR generator
    print("\n[Step 2] Creating DRR generator...")
    try:
        # Try direct method first
        drr, sdd = create_drr_generator_direct(subject_processed, device)
    except Exception as e:
        print(f"Direct creation failed: {e}")
        print("Falling back to temporary file method...")
        # Fallback: Save and use read()
        temp_path = save_subject_for_diffdrr(subject_processed)
        from diffdrr.data import read
        diffdrr_subject = read(temp_path)
        sdd = 1800.0
        drr = DRR(
            subject=diffdrr_subject,
            sdd=sdd,
            height=512,
            width=512,
            delx=0.7,
            dely=0.7,
        ).to(device)
    
    # Step 3: Generate multiple clinical views
    print("\n[Step 3] Generating clinical views...")
    results = generate_chest_xray_views(drr, sdd, OUTPUT_DIR, device)
    
    # Step 4: Create comparison figure
    print("\n[Step 4] Creating comparison visualization...")
    create_comparison_figure(results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✓ DRR generation completed successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()