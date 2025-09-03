#!/usr/bin/env python3
"""
最小限のDiffDRRテスト
"""

import torch
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from diffdrr.data import load_example_ct


def test_minimal():
    """DiffDRRの最小限のテスト"""
    
    # 例のCTデータを読み込む
    print("Loading example CT...")
    subject = load_example_ct()
    
    print(f"Subject type: {type(subject)}")
    print(f"Subject has 'reorient': {hasattr(subject, 'reorient')}")
    print(f"Subject has 'volume': {hasattr(subject, 'volume')}")
    print(f"Subject has 'density': {hasattr(subject, 'density')}")
    
    # DRRモジュールを作成
    print("\nCreating DRR module...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    drr = DRR(
        subject,
        sdd=1800.0,
        height=256,
        delx=1.0,
    ).to(device)
    
    print(f"DRR module created successfully")
    
    # カメラポーズを設定
    print("\nSetting camera pose...")
    from diffdrr.pose import convert
    
    rotations = torch.tensor([[0.0, 0.0, 0.0]]).to(device)  # ラジアン
    translations = torch.tensor([[0.0, 0.0, 0.0]]).to(device)  # mm
    
    # convertを使ってRigidTransformを作成
    transform = convert(
        rotations,
        translations,
        parameterization="euler_angles",
        convention="XYZ",
    ).to(device)
    
    # DRRを生成
    print("Generating DRR...")
    drr_image = drr(transform)
    
    print(f"DRR shape: {drr_image.shape}")
    print(f"DRR range: [{drr_image.min():.2f}, {drr_image.max():.2f}]")
    print("\n✓ Test successful!")
    

if __name__ == "__main__":
    test_minimal()