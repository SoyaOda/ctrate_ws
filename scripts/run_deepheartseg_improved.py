#!/usr/bin/env python3
"""
DeepHeartSeg改善版実行スクリプト
AIのアドバイスに基づいた修正:
1. Step1の出力を適切にクロップ
2. Step2への正しい入力
3. 心膜の疑似生成
"""
import os
import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import subprocess
import json

def prepare_improved_input(ct_path, output_base):
    """
    改善された入力データの準備
    """
    print("[INFO] Preparing improved input for DeepHeartSeg...")
    
    # CTデータを読み込み
    ct_img = sitk.ReadImage(ct_path)
    ct_arr = sitk.GetArrayFromImage(ct_img)
    spacing = ct_img.GetSpacing()
    
    # 胸部領域の自動検出（HUベース）
    # 肺野: -1000 ~ -500 HU
    # 軟部組織: -100 ~ 100 HU
    lung_mask = (ct_arr > -1000) & (ct_arr < -500)
    soft_tissue = (ct_arr > -100) & (ct_arr < 100)
    
    # 胸部の大まかな範囲を特定
    z_proj = np.sum(lung_mask, axis=(1, 2))
    chest_slices = np.where(z_proj > 1000)[0]  # 肺が存在するスライス
    
    if len(chest_slices) > 0:
        z_start = max(0, chest_slices[0] - 10)
        z_end = min(ct_arr.shape[0], chest_slices[-1] + 10)
        print(f"  Chest region: slices {z_start} to {z_end}")
        
        # 胸部領域のみ抽出
        chest_ct = ct_arr[z_start:z_end, :, :]
        
        # 新しいSimpleITK画像を作成
        chest_img = sitk.GetImageFromArray(chest_ct)
        chest_img.SetSpacing(spacing)
        origin = list(ct_img.GetOrigin())
        origin[2] += z_start * spacing[2]
        chest_img.SetOrigin(origin)
        chest_img.SetDirection(ct_img.GetDirection())
        
        # NRRDとして保存
        output_nrrd = output_base / "improved_input.nrrd"
        sitk.WriteImage(chest_img, str(output_nrrd))
        print(f"  Saved cropped chest CT to: {output_nrrd}")
        
        return str(output_nrrd)
    else:
        # フォールバック：元のデータをそのまま使用
        output_nrrd = output_base / "original_input.nrrd"
        sitk.WriteImage(ct_img, str(output_nrrd))
        return str(output_nrrd)

def fix_deepheartseg_config():
    """
    DeepHeartSeg設定ファイルの修正
    """
    config_path = Path("/home/soya/ctrate_ws/tools/DeepHeartSeg/data/test_config_improved.yaml")
    
    config = """io:
  path_to_data_folder: /opt/DeepHeartSeg/data
  raw_data_folder_name: improved_data/raw
  heartloc_data_folder_name: improved_data/step1_heartloc
  heartseg_data_folder_name: improved_data/step2_heartseg
  curated_data_folder_name: curated
  qc_curated_data_folder_name: curated_qc
  resampled_data_folder_name: resampled
  model_input_folder_name: model_input
  model_weights_folder_name: /opt/DeepHeartSeg/data/step1_heartloc/model_weights
  model_output_folder_name: model_output
  upsampled_data_folder_name: model_output_nrrd
  step1_inferred_data_folder_name: model_output_nrrd
  bbox_folder_name: bbox
  cropped_data_folder_name: cropped
  seg_metrics_folder_name: model_output_metrics

processing:
  has_manual_seg: false
  fill_mask_holes: true
  export_png: false
  create_test_set: "All"
  use_gpu: false
  gpu_ids: [0]
  multiprocessing: false
  num_cores: 1
  # Step1用の設定 - より標準的な値に
  curated_size: [512, 512, 0]
  curated_spacing: [0.7, 0.7, 2.5]
  model_input_size: 112
  model_input_spacing: 3.0
  # Step2用の追加設定
  use_inferred_masks: true
  inter_size: [384, 384, 80]
  training_size: [128, 128, 112]
  final_size: [128, 128, 80]
  final_spacing: [0, 0, 2.5]

model:
  pool_size: [2, 2, 2]
  conv_size: [3, 3, 3]
  down_steps: 4
  extended: false
  weights_file_name: GPU_1.hdf5
"""
    
    config_path.write_text(config)
    print(f"[INFO] Created improved config at: {config_path}")
    return str(config_path)

def run_deepheartseg_step1_improved():
    """
    改善されたStep1実行
    """
    print("\n[Running DeepHeartSeg Step1 - Improved]")
    print("="*60)
    
    script = """
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, '/opt/DeepHeartSeg/src')
from step1_heartloc import export_data, downsample_data, input_data_prep, run_inference, upsample_results

# Load config
with open('/opt/DeepHeartSeg/data/test_config_improved.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Starting Improved DeepHeartSeg Step 1")
print("="*60)

io_config = config['io']
proc_config = config['processing']
model_config = config['model']

# Paths
data_folder = io_config['path_to_data_folder']
raw_folder = os.path.join(data_folder, io_config['raw_data_folder_name'])
heartloc_folder = os.path.join(data_folder, io_config['heartloc_data_folder_name'])

# Create directories
for d in [heartloc_folder]:
    if not os.path.exists(d):
        os.makedirs(d)

curated_dir = os.path.join(heartloc_folder, io_config['curated_data_folder_name'])
qc_curated_dir = os.path.join(heartloc_folder, io_config['qc_curated_data_folder_name'])
resampled_dir = os.path.join(heartloc_folder, io_config['resampled_data_folder_name'])
model_input_dir = os.path.join(heartloc_folder, io_config['model_input_folder_name'])
model_output_dir = os.path.join(heartloc_folder, io_config['model_output_folder_name'])
model_output_nrrd_dir = os.path.join(heartloc_folder, io_config['upsampled_data_folder_name'])

for d in [curated_dir, qc_curated_dir, resampled_dir, model_input_dir, model_output_dir, model_output_nrrd_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

print("Paths configured")

# Execute steps
try:
    print("\\n1. Exporting data...")
    export_data.export_data(
        raw_data_dir_path=raw_folder,
        curated_dir_path=curated_dir,
        qc_curated_dir_path=qc_curated_dir,
        curated_size=proc_config['curated_size'],
        curated_spacing=proc_config['curated_spacing'],
        num_cores=proc_config['num_cores'],
        export_png=proc_config['export_png'],
        has_manual_seg=proc_config['has_manual_seg']
    )
    print("  Export completed")
except Exception as e:
    print("  Error:", str(e))

try:
    print("\\n2. Downsampling...")
    downsample_data.downsample_data(
        curated_dir_path=curated_dir,
        resampled_dir_path=resampled_dir,
        model_input_dir_path=model_input_dir,
        crop_size=proc_config['model_input_size'],
        new_spacing=proc_config['model_input_spacing'],
        has_manual_seg=proc_config['has_manual_seg'],
        num_cores=proc_config['num_cores']
    )
    print("  Downsampling completed")
except Exception as e:
    print("  Error:", str(e))

try:
    print("\\n3. Input prep...")
    input_data_prep.input_data_prep(
        resampled_dir_path=resampled_dir,
        model_input_dir_path=model_input_dir,
        create_test_set=proc_config['create_test_set'],
        crop_size=proc_config['model_input_size'],
        new_spacing=proc_config['model_input_spacing'],
        has_manual_seg=proc_config['has_manual_seg'],
        fill_mask_holes=proc_config['fill_mask_holes']
    )
    print("  Input prep completed")
except Exception as e:
    print("  Error:", str(e))

try:
    print("\\n4. Running inference...")
    weights_path = io_config['model_weights_folder_name']
    run_inference.run_inference(
        model_output_dir_path=model_output_dir,
        model_input_dir_path=model_input_dir,
        model_weights_dir_path=weights_path,
        crop_size=proc_config['model_input_size'],
        use_gpu=proc_config['use_gpu'],
        gpu_ids=proc_config['gpu_ids'],
        export_png=proc_config['export_png'],
        model_down_steps=model_config['down_steps'],
        extended=model_config['extended'],
        has_manual_seg=proc_config['has_manual_seg'],
        weights_file_name=model_config['weights_file_name']
    )
    print("  Inference completed")
except Exception as e:
    print("  Error:", str(e))

try:
    print("\\n5. Upsampling...")
    upsample_results.upsample_results(
        curated_dir_path=curated_dir,
        resampled_dir_path=resampled_dir,
        model_output_dir_path=model_output_dir,
        model_output_nrrd_dir_path=model_output_nrrd_dir,
        num_cores=proc_config['num_cores']
    )
    print("  Upsampling completed")
except Exception as e:
    print("  Error:", str(e))

print("\\nStep 1 completed!")
if os.path.exists(model_output_nrrd_dir):
    files = os.listdir(model_output_nrrd_dir)
    print("Output files:", files)
"""
    
    # スクリプトを保存
    script_path = Path("/home/soya/ctrate_ws/scripts/run_dhs_step1_improved.py")
    script_path.write_text(script)
    
    # 実行
    cmd = [
        "docker", "run", "--rm",
        "-v", "/home/soya/ctrate_ws/tools/DeepHeartSeg:/opt/DeepHeartSeg",
        "-v", "/home/soya/ctrate_ws/scripts:/scripts",
        "-w", "/opt/DeepHeartSeg/src",
        "deepheartseg:py27",
        "python2", "/scripts/run_dhs_step1_improved.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def create_pericardium_from_heart(heart_mask_path, output_path, thickness_mm=4.0):
    """
    心臓マスクから心膜を疑似生成
    """
    print("\n[Creating Pseudo-Pericardium]")
    print("="*60)
    
    # 心臓マスクを読み込み
    heart_img = sitk.ReadImage(heart_mask_path)
    heart_arr = sitk.GetArrayFromImage(heart_img)
    spacing = heart_img.GetSpacing()
    
    print(f"  Heart mask shape: {heart_arr.shape}")
    print(f"  Spacing: {spacing}")
    
    # バイナリ化
    heart_binary = heart_arr > 0
    
    # 距離変換で心膜領域を作成
    from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure
    
    # 外側への距離
    outside = ~heart_binary
    dist_outside = distance_transform_edt(outside, sampling=spacing[::-1])
    
    # 内側への距離
    dist_inside = distance_transform_edt(heart_binary, sampling=spacing[::-1])
    
    # 心膜: 心臓表面から内側2mm、外側4mmの領域
    pericardium = ((dist_outside > 0) & (dist_outside <= thickness_mm)) | \
                  ((dist_inside > 0) & (dist_inside <= 2.0))
    
    # 形態学的処理で滑らかに
    struct = generate_binary_structure(3, 2)
    pericardium = binary_dilation(pericardium, structure=struct, iterations=1)
    
    # 保存
    peri_img = sitk.GetImageFromArray(pericardium.astype(np.uint8))
    peri_img.CopyInformation(heart_img)
    sitk.WriteImage(peri_img, output_path)
    
    print(f"  Pericardium saved to: {output_path}")
    
    # 統計
    peri_voxels = np.sum(pericardium)
    peri_volume = peri_voxels * np.prod(spacing[::-1]) / 1000.0
    print(f"  Pericardium voxels: {peri_voxels:,}")
    print(f"  Pericardium volume: {peri_volume:.2f} ml")
    
    return pericardium

def main():
    # パス設定
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
    output_base = Path("/home/soya/ctrate_ws/outputs/deepheartseg_improved")
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DeepHeartSeg Improved Pipeline")
    print("="*60)
    
    # 1. 改善された入力データの準備
    improved_data_dir = Path("/home/soya/ctrate_ws/tools/DeepHeartSeg/data/improved_data")
    improved_data_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = improved_data_dir / "raw" / "patient001"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 改善された入力を準備
    improved_nrrd = prepare_improved_input(ct_path, output_base)
    
    # DeepHeartSegの入力として配置
    import shutil
    target_nrrd = raw_dir / "img.nrrd"
    shutil.copy(improved_nrrd, target_nrrd)
    print(f"[INFO] Copied improved input to: {target_nrrd}")
    
    # 2. 設定ファイルの修正
    fix_deepheartseg_config()
    
    # 3. Step1の実行
    if run_deepheartseg_step1_improved():
        print("[SUCCESS] Step1 completed")
        
        # Step1の出力を確認
        step1_output = Path("/home/soya/ctrate_ws/tools/DeepHeartSeg/data/improved_data/step1_heartloc/model_output_nrrd/patient001_pred.nrrd")
        
        if step1_output.exists():
            # 4. 心膜の疑似生成
            peri_output = output_base / "pseudo_pericardium.nii.gz"
            create_pericardium_from_heart(str(step1_output), str(peri_output))
            
            # 5. 結果の変換と保存
            print("\n[Converting Results]")
            
            # Step1出力をNIfTIに変換
            step1_img = sitk.ReadImage(str(step1_output))
            heart_output = output_base / "heart_mask_improved.nii.gz"
            sitk.WriteImage(step1_img, str(heart_output))
            print(f"  Heart mask saved to: {heart_output}")
            
            print("\n[RESULTS SUMMARY]")
            print("="*60)
            print(f"  Heart mask: {heart_output}")
            print(f"  Pericardium mask: {peri_output}")
            print(f"  Output directory: {output_base}")
            print("\n  Next step: Visualize these masks to verify quality")
            
    else:
        print("[ERROR] Step1 failed")

if __name__ == "__main__":
    main()