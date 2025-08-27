
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
    print("\n1. Exporting data...")
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
    print("\n2. Downsampling...")
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
    print("\n3. Input prep...")
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
    print("\n4. Running inference...")
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
    print("\n5. Upsampling...")
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

print("\nStep 1 completed!")
if os.path.exists(model_output_nrrd_dir):
    files = os.listdir(model_output_nrrd_dir)
    print("Output files:", files)
