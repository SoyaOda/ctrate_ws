#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Run DeepHeartSeg Step 2 - Heart Segmentation (Fixed parameters)
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
    
    # Determine run mode
    if proc_config['has_manual_seg']:
        run = "Test" if proc_config['use_inferred_masks'] else "Train"
    else:
        run = "Test"
    
    # Step 1: Compute bounding box
    print("\n1. Computing bounding boxes...")
    try:
        compute_bbox.compute_bbox(
            cur_dir=curated_dir,
            pred_dir=step1_output_dir,
            output_dir=bbox_dir,
            num_cores=proc_config['num_cores'],
            has_manual_seg=proc_config['has_manual_seg'],
            run=run
        )
        print("  Bounding boxes computed")
    except Exception as e:
        print("  Error in compute_bbox:", str(e))
    
    # Step 2: Crop data
    print("\n2. Cropping data...")
    try:
        crop_data.crop_data(
            bb_calc_dir=bbox_dir,
            output_dir=cropped_dir, 
            network_dir=model_input_dir,
            inter_size=proc_config['inter_size'],
            final_size=proc_config['final_size'],
            final_spacing=proc_config['final_spacing'],
            num_cores=proc_config['num_cores']
        )
        print("  Data cropping completed")
    except Exception as e:
        print("  Error in crop_data:", str(e))
    
    # Step 3: Input data preparation
    print("\n3. Preparing model input...")
    try:
        input_data_prep.input_data_prep(
            input_dir=cropped_dir,
            output_dir=model_input_dir,
            run=run,
            fill_holes=proc_config['fill_mask_holes'],
            final_size=proc_config['final_size']
        )
        print("  Input preparation completed")
    except Exception as e:
        print("  Error in input_data_prep:", str(e))
    
    # Step 4: Run inference
    print("\n4. Running model inference...")
    try:
        weights_path = io_config['model_weights_folder_name']
        run_inference.run_inference(
            model_weights_dir_path=weights_path,
            data_dir=model_input_dir,
            output_dir=model_output_dir,
            weights_file_name=model_config['weights_file_name'],
            use_gpu=proc_config['use_gpu'],
            gpu_ids=proc_config['gpu_ids'],
            export_png=proc_config['export_png'],
            final_size=proc_config['final_size'],
            training_size=proc_config['training_size'],
            down_steps=model_config['down_steps']
        )
        print("  Inference completed")
    except Exception as e:
        print("  Error in run_inference:", str(e))
    
    # Step 5: Upsampling results
    print("\n5. Upsampling results...")
    try:
        upsample_results.upsample_results(
            cur_input=curated_dir,
            crop_input=cropped_dir,
            network_dir=model_input_dir,
            test_dir=model_output_dir,
            output_dir=model_output_nrrd_dir,
            inter_size=proc_config['inter_size'],
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