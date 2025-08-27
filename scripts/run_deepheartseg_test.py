#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Run DeepHeartSeg Step 1 - Heart Localization
"""

import os
import sys
import yaml

# Set matplotlib backend before importing DeepHeartSeg modules
import matplotlib
matplotlib.use('Agg')

# Add DeepHeartSeg src to path
sys.path.insert(0, '/opt/DeepHeartSeg/src')

# Import DeepHeartSeg modules
from step1_heartloc import export_data, downsample_data, input_data_prep, run_inference, upsample_results

def main():
    # Load config
    with open('/opt/DeepHeartSeg/data/test_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Starting DeepHeartSeg Step 1 - Heart Localization")
    print("="*60)
    
    # Extract config parameters
    io_config = config['io']
    proc_config = config['processing']
    model_config = config['model']
    
    # Set paths
    data_folder = io_config['path_to_data_folder']
    raw_folder = os.path.join(data_folder, io_config['raw_data_folder_name'])
    heartloc_folder = os.path.join(data_folder, io_config['heartloc_data_folder_name'])
    
    # Create output directories
    if not os.path.exists(heartloc_folder):
        os.makedirs(heartloc_folder)
    
    curated_dir = os.path.join(heartloc_folder, io_config['curated_data_folder_name'])
    qc_curated_dir = os.path.join(heartloc_folder, io_config['qc_curated_data_folder_name'])
    resampled_dir = os.path.join(heartloc_folder, io_config['resampled_data_folder_name'])
    model_input_dir = os.path.join(heartloc_folder, io_config['model_input_folder_name'])
    model_output_dir = os.path.join(heartloc_folder, io_config['model_output_folder_name'])
    model_output_nrrd_dir = os.path.join(heartloc_folder, io_config['upsampled_data_folder_name'])
    
    for d in [curated_dir, qc_curated_dir, resampled_dir, model_input_dir, model_output_dir, model_output_nrrd_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    print("Paths configured:")
    print("  Raw data:", raw_folder)
    print("  Output:", heartloc_folder)
    
    # Step 1: Data preparation
    print("\n1. Exporting/preprocessing data...")
    try:
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
        print("  Data export completed")
    except Exception as e:
        print("  Error in export_data:", str(e))
        # Continue anyway
    
    # Step 2: Downsampling
    print("\n2. Downsampling data...")
    try:
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
        print("  Error in downsample_data:", str(e))
    
    # Step 3: Input data preparation
    print("\n3. Preparing model input...")
    try:
        input_data_prep.input_data_prep(
            resampled_dir_path=resampled_dir,
            model_input_dir_path=model_input_dir,
            create_test_set=proc_config['create_test_set'],
            crop_size=proc_config['model_input_size'],
            new_spacing=proc_config['model_input_spacing'],
            has_manual_seg=proc_config['has_manual_seg'],
            fill_mask_holes=proc_config['fill_mask_holes']
        )
        print("  Input preparation completed")
    except Exception as e:
        print("  Error in input_data_prep:", str(e))
    
    # Step 4: Run inference
    print("\n4. Running model inference...")
    try:
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
        print("  Error in run_inference:", str(e))
    
    # Step 5: Upsampling results
    print("\n5. Upsampling results...")
    try:
        upsample_results.upsample_results(
            curated_dir_path=curated_dir,
            resampled_dir_path=resampled_dir,
            model_output_dir_path=model_output_dir,
            model_output_nrrd_dir_path=model_output_nrrd_dir,
            num_cores=proc_config['num_cores']
        )
        print("  Upsampling completed")
    except Exception as e:
        print("  Error in upsample_results:", str(e))
    
    print("\n" + "="*60)
    print("DeepHeartSeg Step 1 completed")
    print("Output directory:", heartloc_folder)
    
    # List output files
    if os.path.exists(model_output_nrrd_dir):
        files = os.listdir(model_output_nrrd_dir)
        if files:
            print("Output files:", files)
        else:
            print("No output files found")

if __name__ == "__main__":
    main()