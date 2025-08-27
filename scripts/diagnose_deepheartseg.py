#!/usr/bin/env python3
"""
DeepHeartSeg出力の診断スクリプト
"""
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json

def diagnose_step1():
    """Step1出力を診断"""
    print("\n[Step1 Diagnosis]")
    print("-"*50)
    
    step1_dir = Path("/home/soya/ctrate_ws/tools/DeepHeartSeg/data/test_data/step1_heartloc")
    
    # 各ディレクトリの確認
    dirs_to_check = [
        "curated", "model_output_nrrd", "model_output", 
        "model_input", "resampled", "bbox"
    ]
    
    for d in dirs_to_check:
        dir_path = step1_dir / d
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            print(f"  {d}: {len(files)} files")
            if files and len(files) <= 3:
                for f in files[:3]:
                    print(f"    - {f.name}")
        else:
            print(f"  {d}: NOT FOUND")
    
    # Step1の出力マスクを確認
    step1_mask = step1_dir / "model_output_nrrd/patient001_pred.nrrd"
    if step1_mask.exists():
        img = sitk.ReadImage(str(step1_mask))
        arr = sitk.GetArrayFromImage(img)
        
        print(f"\n  Step1 Output Mask:")
        print(f"    Shape: {arr.shape}")
        print(f"    Spacing: {img.GetSpacing()}")
        print(f"    Unique labels: {np.unique(arr)}")
        print(f"    Non-zero voxels: {np.sum(arr > 0):,}")
        
        # ラベルごとのボクセル数
        for label in np.unique(arr):
            if label == 0:
                continue
            count = np.sum(arr == label)
            print(f"    Label {label}: {count:,} voxels")

def diagnose_step2():
    """Step2出力を診断"""
    print("\n[Step2 Diagnosis]")
    print("-"*50)
    
    step2_dir = Path("/home/soya/ctrate_ws/tools/DeepHeartSeg/data/test_data/step2_heartseg")
    
    # 各ディレクトリの確認
    dirs_to_check = [
        "bbox", "cropped", "model_input", 
        "model_output", "model_output_nrrd"
    ]
    
    for d in dirs_to_check:
        dir_path = step2_dir / d
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            print(f"  {d}: {len(files)} files")
            if files and len(files) <= 3:
                for f in files[:3]:
                    print(f"    - {f.name}")
        else:
            print(f"  {d}: NOT FOUND")
    
    # BBoxファイルの確認
    bbox_file = step2_dir / "bbox/bbox.pkl"
    if bbox_file.exists():
        print(f"\n  BBox file exists: {bbox_file}")
        print(f"    Size: {bbox_file.stat().st_size} bytes")
    
    # Step2の出力マスクを確認
    step2_mask = step2_dir / "model_output_nrrd/patient001_pred.nrrd"
    if step2_mask.exists():
        img = sitk.ReadImage(str(step2_mask))
        arr = sitk.GetArrayFromImage(img)
        
        print(f"\n  Step2 Output Mask:")
        print(f"    Shape: {arr.shape}")
        print(f"    Spacing: {img.GetSpacing()}")
        print(f"    Origin: {img.GetOrigin()}")
        print(f"    Direction: {img.GetDirection()}")
        print(f"    Unique labels: {np.unique(arr)}")
        print(f"    Non-zero voxels: {np.sum(arr > 0):,}")
        
        # ラベルごとのボクセル数と位置
        for label in np.unique(arr):
            if label == 0:
                continue
            mask = (arr == label)
            count = np.sum(mask)
            
            # 重心を計算
            indices = np.where(mask)
            if len(indices[0]) > 0:
                center = [np.mean(indices[i]) for i in range(3)]
                print(f"    Label {label}: {count:,} voxels, center: {[int(c) for c in center]}")

def check_step3():
    """Step3の存在を確認"""
    print("\n[Step3 Check]")
    print("-"*50)
    
    # Step3関連のファイルを探す
    src_dir = Path("/home/soya/ctrate_ws/tools/DeepHeartSeg/src")
    
    # Step3のソースファイル
    step3_files = list(src_dir.glob("*step3*"))
    step3_dirs = list(src_dir.glob("step3_*"))
    
    if step3_files:
        print("  Step3 source files found:")
        for f in step3_files:
            print(f"    - {f.name}")
    
    if step3_dirs:
        print("  Step3 directories found:")
        for d in step3_dirs:
            print(f"    - {d.name}/")
            # ディレクトリ内のファイルも表示
            py_files = list(d.glob("*.py"))[:3]
            for f in py_files:
                print(f"      - {f.name}")
    
    # 設定ファイルを確認
    config_dir = src_dir / "config"
    step3_configs = list(config_dir.glob("*step3*")) + list(config_dir.glob("*peri*"))
    
    if step3_configs:
        print("\n  Step3 config files found:")
        for c in step3_configs:
            print(f"    - {c.name}")
    
    # 重みファイルを確認
    weights_dir = Path("/home/soya/ctrate_ws/tools/DeepHeartSeg/data")
    step3_weights = list(weights_dir.glob("**/step3*/model_weights/*"))
    
    if step3_weights:
        print("\n  Step3 weight files found:")
        for w in step3_weights:
            print(f"    - {w}")

def compare_with_totalseg():
    """TotalSegmentatorの心臓マスクと比較"""
    print("\n[Comparison with TotalSegmentator]")
    print("-"*50)
    
    # TotalSegmentatorの心臓マスク
    totalseg_dir = Path("/home/soya/ctrate_ws/outputs/totalseg_output/test_valid_2_a_1")
    heart_parts = [
        "heart_myocardium",
        "heart_ventricle_left", 
        "heart_ventricle_right",
        "heart_atrium_left",
        "heart_atrium_right"
    ]
    
    total_voxels = 0
    for part in heart_parts:
        part_file = totalseg_dir / f"{part}.nii.gz"
        if part_file.exists():
            img = sitk.ReadImage(str(part_file))
            arr = sitk.GetArrayFromImage(img)
            voxels = np.sum(arr > 0)
            total_voxels += voxels
            print(f"  {part}: {voxels:,} voxels")
    
    print(f"  Total TotalSegmentator heart: {total_voxels:,} voxels")
    
    # DeepHeartSegと比較
    dhs_mask = Path("/home/soya/ctrate_ws/outputs/deepheartseg_test/step2_heartseg.nii.gz")
    if dhs_mask.exists():
        img = sitk.ReadImage(str(dhs_mask))
        arr = sitk.GetArrayFromImage(img)
        dhs_voxels = np.sum(arr > 0)
        print(f"  DeepHeartSeg heart: {dhs_voxels:,} voxels")
        
        if total_voxels > 0:
            ratio = dhs_voxels / total_voxels * 100
            print(f"  Ratio (DHS/TotalSeg): {ratio:.1f}%")
            
            if ratio < 50:
                print("  ⚠ WARNING: DeepHeartSeg segmentation is much smaller than expected!")

def main():
    print("="*60)
    print("DeepHeartSeg Diagnostic Report")
    print("="*60)
    
    diagnose_step1()
    diagnose_step2()
    check_step3()
    compare_with_totalseg()
    
    print("\n" + "="*60)
    print("Diagnostic Summary")
    print("="*60)
    
    # 問題の可能性を示唆
    print("\n[Potential Issues]")
    
    # Step2の出力確認
    step2_mask_path = Path("/home/soya/ctrate_ws/tools/DeepHeartSeg/data/test_data/step2_heartseg/model_output_nrrd/patient001_pred.nrrd")
    if step2_mask_path.exists():
        img = sitk.ReadImage(str(step2_mask_path))
        arr = sitk.GetArrayFromImage(img)
        unique_labels = np.unique(arr)
        
        if len(unique_labels) <= 2:  # 0と1だけ
            print("  ⚠ Step2 output has only binary labels (expected multi-label)")
            print("    -> Weight file mismatch or wrong model configuration")
        
        non_zero = np.sum(arr > 0)
        if non_zero < 100000:  # 10万ボクセル未満は異常に小さい
            print("  ⚠ Step2 segmentation is abnormally small")
            print("    -> BBox calculation or cropping issue")
    
    print("\n[Recommendations]")
    print("  1. Check if Step1 output is correct (should be heart localization)")
    print("  2. Verify BBox calculation between Step1 and Step2")
    print("  3. Consider using Step3 for pericardium segmentation")
    print("  4. Alternative: Use TotalSegmentator-based approach (more stable)")

if __name__ == "__main__":
    main()