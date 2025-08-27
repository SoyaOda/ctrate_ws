#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Run DeepHeartSeg Step 2 - Heart Segmentation
"""

import os
import sys
import yaml

# Set matplotlib backend before importing DeepHeartSeg modules
import matplotlib
matplotlib.use('Agg')

# Add DeepHeartSeg src to path
sys.path.insert(0, '/opt/DeepHeartSeg/src')

# Import DeepHeartSeg Step2 modules
from step2_heartseg import compute_bbox, crop_data, input_data_prep, run_inference, upsample_results

def main():
    # Load config
    with open('/opt/DeepHeartSeg/data/test_config_step2.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Starting DeepHeartSeg Step 2 - Heart Segmentation")
    print("="*60)
    
    # Extract config parameters
    io_config = config['io']
    proc_config = config['processing']
    model_config = config['model']
    
    # Set paths
    data_folder = io_config['path_to_data_folder']
    heartloc_folder = os.path.join(data_folder, io_config['heartloc_data_folder_name'])
    heartseg_folder = os.path.join(data_folder, io_config['heartseg_data_folder_name'])
    
    # Create output directories
    if not os.path.exists(heartseg_folder):
        os.makedirs(heartseg_folder)
    
    # Subdirectories
    bbox_dir = os.path.join(heartseg_folder, io_config['bbox_folder_name'])
    cropped_dir = os.path.join(heartseg_folder, io_config['cropped_data_folder_name'])
    model_input_dir = os.path.join(heartseg_folder, io_config['model_input_folder_name'])
    model_output_dir = os.path.join(heartseg_folder, io_config['model_output_folder_name'])
    model_output_nrrd_dir = os.path.join(heartseg_folder, io_config['upsampled_data_folder_name'])
    seg_metrics_dir = os.path.join(heartseg_folder, io_config['seg_metrics_folder_name'])
    
    for d in [bbox_dir, cropped_dir, model_input_dir, model_output_dir, model_output_nrrd_dir, seg_metrics_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    print("Paths configured:")
    print("  Input (Step1):", heartloc_folder)
    print("  Output (Step2):", heartseg_folder)
    
    # Get paths from Step1
    curated_dir = os.path.join(heartloc_folder, io_config['curated_data_folder_name'])
    step1_output_dir = os.path.join(heartloc_folder, io_config['step1_inferred_data_folder_name'])
    
    # Step 1: Compute bounding box
    print("\n1. Computing bounding boxes...")
    try:
        compute_bbox.compute_bbox(
            curated_dir_path=curated_dir,
            seg_dir_path=step1_output_dir,
            bbox_dir_path=bbox_dir,
            has_manual_seg=proc_config['has_manual_seg'],
            use_inferred_masks=proc_config['use_inferred_masks'],
            export_png=proc_config['export_png']
        )
        print("  Bounding boxes computed")
    except Exception as e:
        print("  Error in compute_bbox:", str(e))
    
    # Step 2: Crop data
    print("\n2. Cropping data...")
    try:
        crop_data.crop_data(
            curated_dir_path=curated_dir,
            bbox_dir_path=bbox_dir,
            cropped_dir_path=cropped_dir,
            inter_size=proc_config['inter_size'],
            training_size=proc_config['training_size'],
            final_size=proc_config['final_size'],
            final_spacing=proc_config['final_spacing'],
            curated_size=proc_config['curated_size'],
            curated_spacing=proc_config['curated_spacing'],
            num_cores=proc_config['num_cores'],
            has_manual_seg=proc_config['has_manual_seg']
        )
        print("  Data cropping completed")
    except Exception as e:
        print("  Error in crop_data:", str(e))
    
    # Step 3: Input data preparation
    print("\n3. Preparing model input...")
    try:
        input_data_prep.input_data_prep(
            cropped_dir_path=cropped_dir,
            model_input_dir_path=model_input_dir,
            has_manual_seg=proc_config['has_manual_seg']
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
            cropped_dir_path=cropped_dir,
            model_output_dir_path=model_output_dir,
            model_output_nrrd_dir_path=model_output_nrrd_dir,
            has_manual_seg=proc_config['has_manual_seg'],
            num_cores=proc_config['num_cores']
        )
        print("  Upsampling completed")
    except Exception as e:
        print("  Error in upsample_results:", str(e))
    
    print("\n" + "="*60)
    print("DeepHeartSeg Step 2 completed")
    print("Output directory:", heartseg_folder)
    
    # List output files
    if os.path.exists(model_output_nrrd_dir):
        files = os.listdir(model_output_nrrd_dir)
        if files:
            print("Output files:", files)
        else:
            print("No output files found")

if __name__ == "__main__":
    main()