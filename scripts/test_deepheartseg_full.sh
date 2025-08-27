#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "DeepHeartSeg Full Test"
echo "============================================"

# Input and output paths
CT_INPUT="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/deepheartseg_full_test"
DEEPHEARTSEG_ROOT="/home/soya/ctrate_ws/tools/DeepHeartSeg"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/raw/patient001"

echo "Step 1: Converting NIfTI to NRRD..."
docker run --rm \
    -v "$CT_INPUT":/input.nii.gz \
    -v "$OUTPUT_DIR":/output \
    deepheartseg:py27 python2 -c "
import SimpleITK as sitk
img = sitk.ReadImage('/input.nii.gz')
sitk.WriteImage(img, '/output/raw/patient001/img.nrrd')
print('Converted to NRRD format successfully')
"

echo ""
echo "Step 2: Creating custom config for heart localization..."
cat > "$OUTPUT_DIR/heart_loc_config.yaml" <<'EOF'
io:
  path_to_data_folder: "/workspace/data"
  raw_data_folder_name: "raw"
  heartloc_data_folder_name: "step1_heartloc"
  curated_data_folder_name: "curated"
  curated_qc_folder_name: "curated_qc"
  resampled_data_folder_name: "resampled"
  model_input_folder_name: "model_input"
  model_weights_folder_name: "model_weights"
  model_output_folder_name: "model_output"
  model_output_nrrd_folder_name: "model_output_nrrd"

processing:
  has_manual_seg: false
  export_png: false
  multiprocessing: false

network:
  spacing: [5.0, 5.0, 5.0]
  padding: [[42, 42], [34, 34], [42, 42]]
  name: "heartloc"
  vnet_depth: 3
  filters_init: 16
  kernel_initializer: "he_uniform"
  activation: "relu"
  batch_norm: false
  dropout_rate: 0.5
  batch_size: 1
EOF

echo ""
echo "Step 3: Running DeepHeartSeg Step 1 (Heart Localization)..."
docker run --rm \
    -v "$OUTPUT_DIR":/workspace/data \
    -v "$DEEPHEARTSEG_ROOT/data/step1_heartloc/model_weights":/workspace/data/step1_heartloc/model_weights \
    -v "$DEEPHEARTSEG_ROOT/src":/opt/DeepHeartSeg/src \
    -w /opt/DeepHeartSeg/src \
    deepheartseg:py27 \
    python2 run_step1_heart_localization.py --conf /workspace/data/heart_loc_config.yaml

echo ""
echo "Step 4: Checking output..."
ls -la "$OUTPUT_DIR/step1_heartloc/"

echo ""
echo "Step 5: Converting output back to NIfTI..."
docker run --rm \
    -v "$OUTPUT_DIR":/workspace \
    deepheartseg:py27 python2 -c "
import SimpleITK as sitk
import os
import glob

# Find output NRRD files
nrrd_files = glob.glob('/workspace/step1_heartloc/model_output_nrrd/*.nrrd')
if nrrd_files:
    for nrrd_file in nrrd_files:
        img = sitk.ReadImage(nrrd_file)
        base_name = os.path.basename(nrrd_file).replace('.nrrd', '.nii.gz')
        output_path = os.path.join('/workspace', base_name)
        sitk.WriteImage(img, output_path)
        print('Converted %s to NIfTI format' % base_name)
else:
    print('No output NRRD files found')
"

echo ""
echo "Final output:"
ls -la "$OUTPUT_DIR/"

echo "Test completed."