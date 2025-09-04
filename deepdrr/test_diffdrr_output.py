#!/usr/bin/env python3
"""
DiffDRRの出力を詳細に調査
Beer-Lambert則がどこで適用されているか確認
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
import tempfile
import os


def test_diffdrr_output():
    """DiffDRRの生の出力を確認"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # マスクパス
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    # マスクデータを読み込み
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    mask_binary = (mask_data > 0).astype(np.float32)
    
    print("\n" + "="*60)
    print("Testing DiffDRR Output")
    print("="*60)
    
    # カメラパラメータ
    sdd = 1800.0
    height, width = 512, 512
    delx, dely = 0.7, 0.7
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    # テスト1: 値を変えたマスク
    test_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    for val in test_values:
        print(f"\n{'='*40}")
        print(f"Testing with mask value = {val}")
        print('='*40)
        
        # マスクをスケール
        mask_scaled = mask_binary * val
        
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
            mask_nifti_scaled = nib.Nifti1Image(mask_scaled, mask_nifti.affine)
            nib.save(mask_nifti_scaled, temp_file.name)
            temp_path = temp_file.name
        
        try:
            # DiffDRRで投影
            mask_subject = read(temp_path, bone_attenuation_multiplier=1.0)
            
            # デバッグ: 読み込まれた密度を確認
            print(f"  Density range after read: [{mask_subject.density.data.min():.6f}, {mask_subject.density.data.max():.6f}]")
            
            drr = DRR(
                mask_subject,
                sdd=sdd,
                height=height,
                width=width,
                delx=delx,
                dely=dely,
            ).to(device)
            
            # 投影を実行
            output = drr(rotations, translations,
                        parameterization="euler_angles",
                        convention="ZXY")
            
            output_np = output.squeeze().cpu().numpy()
            
            print(f"  DRR output range: [{output_np.min():.6f}, {output_np.max():.6f}]")
            print(f"  Output mean: {output_np.mean():.6f}")
            print(f"  Output std: {output_np.std():.6f}")
            
            # 出力の性質を調べる
            non_zero = output_np[output_np > 1e-6]
            if len(non_zero) > 0:
                # 線形性をチェック
                # 線形なら、値を2倍にすれば出力も2倍になるはず
                print(f"  Non-zero pixels: {len(non_zero)}")
                print(f"  Non-zero mean: {non_zero.mean():.6f}")
                
                # ログスケールで出力を見る
                log_output = np.log(np.maximum(output_np, 1e-10))
                print(f"  Log output range: [{log_output.min():.3f}, {log_output.max():.3f}]")
            
        finally:
            os.unlink(temp_path)
    
    # テスト2: 生の積分値を確認
    print("\n" + "="*60)
    print("Analyzing Output Nature")
    print("="*60)
    
    # val = 1.0での詳細分析
    val = 1.0
    mask_scaled = mask_binary * val
    
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
        mask_nifti_scaled = nib.Nifti1Image(mask_scaled, mask_nifti.affine)
        nib.save(mask_nifti_scaled, temp_file.name)
        temp_path = temp_file.name
    
    try:
        mask_subject = read(temp_path, bone_attenuation_multiplier=1.0)
        drr = DRR(mask_subject, sdd=sdd, height=height, width=width, delx=delx, dely=dely).to(device)
        
        # renderメソッドに直接アクセス（可能なら）
        source = torch.tensor([[0.0, -850.0, 0.0]], device=device)  # カメラ位置
        target = torch.tensor([[[0.0, 850.0, 0.0]]], device=device)  # 検出器位置（簡略化）
        
        output = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
        output_np = output.squeeze().cpu().numpy()
        
        # 出力が透過率（exp(-積分)）なのか、単なる積分なのかを判定
        print("\nOutput Analysis:")
        print(f"  Range: [{output_np.min():.6f}, {output_np.max():.6f}]")
        
        # もし透過率なら、値は[0, 1]の範囲にあるはず
        if output_np.min() >= 0 and output_np.max() <= 1:
            print("  ✓ Output is in [0, 1] range - likely transmittance (exp(-integral))")
            
            # 積分値を逆算
            integral = -np.log(np.maximum(output_np, 1e-10))
            print(f"  Reconstructed integral range: [{integral.min():.3f}, {integral.max():.3f}]")
        else:
            print("  ✗ Output is NOT in [0, 1] range")
            
            # 値が1より大きい場合、何か別の処理がされている
            if output_np.max() > 1:
                print("  Output > 1 detected - not standard transmittance")
    
    finally:
        os.unlink(temp_path)
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("DiffDRR output is likely transmittance: I = exp(-∫μ dx)")
    print("This is why Beer-Lambert law cannot be avoided by setting μ=0")
    print("The exponential is applied AFTER the line integral")


if __name__ == "__main__":
    test_diffdrr_output()