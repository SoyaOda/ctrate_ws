#!/usr/bin/env python3
"""
パイプラインの簡易テストスクリプト
"""

from generate_drrs import DRRPipeline

def main():
    """メイン関数"""
    # 設定
    CT_NIFTI_PATH = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    MASK_NIFTI_PATH = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    OUTPUT_DIR = "./test_output"
    NUM_SAMPLES = 2  # テスト用に少数のサンプルのみ
    
    # デバイス設定
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # パイプラインを初期化
    print("Initializing pipeline...")
    pipeline = DRRPipeline(
        ct_path=CT_NIFTI_PATH,
        mask_path=MASK_NIFTI_PATH,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )
    
    # 少数のサンプルでテスト
    print(f"\nGenerating {NUM_SAMPLES} test samples...")
    pipeline.generate_dataset(num_samples=NUM_SAMPLES)
    
    print("\n✓ Test completed successfully!")
    
    # 結果を確認
    from pathlib import Path
    output_path = Path(OUTPUT_DIR)
    drr_files = list((output_path / "drr").glob("*.png"))
    mask_files = list((output_path / "mask").glob("*.npy"))
    viz_files = list((output_path / "visualization").glob("*.png"))
    
    print(f"\n=== Generated Files ===")
    print(f"DRR images: {len(drr_files)}")
    print(f"Mask files: {len(mask_files)}")
    print(f"Visualization images: {len(viz_files)}")


if __name__ == "__main__":
    main()