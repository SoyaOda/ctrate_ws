#!/bin/bash
# すべての症例に対してTotalSegmentatorを実行するスクリプト
# Docker Desktop for WindowsのWSL2統合が必要

set -e  # エラーが発生したら停止

# 色付き出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== TotalSegmentator Setup for All Cases ===${NC}"
echo ""

# Docker確認
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not available in WSL2${NC}"
    echo "Please ensure Docker Desktop is running and WSL2 integration is enabled:"
    echo "1. Open Docker Desktop"
    echo "2. Go to Settings > Resources > WSL Integration"
    echo "3. Enable integration with your WSL2 distro"
    echo "4. Restart Docker Desktop and WSL2"
    exit 1
fi

# 処理する症例のリスト
declare -a CASES=(
    "valid_1/valid_1_a/valid_1_a_1"
    "valid_1/valid_1_a/valid_1_a_2"
    "valid_2/valid_2_a/valid_2_a_1"
)

# TotalSegmentatorのバージョン
TOTALSEG_VERSION="2.10.0"

# 各症例を処理
for CASE in "${CASES[@]}"; do
    # パスを分解
    CASE_NAME=$(echo $CASE | sed 's/\//_/g')
    INPUT_FILE="/workspace/data/CT-RATE-v2/dataset/valid_fixed/${CASE}.nii.gz"
    OUTPUT_DIR_TOTAL="/workspace/outputs/batch_results/${CASE_NAME}/segmentation"
    OUTPUT_DIR_TISSUE="/workspace/outputs/batch_results/${CASE_NAME}/segmentation_tissue"
    OUTPUT_DIR_MERGED="/home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation_merged"
    
    echo -e "${YELLOW}Processing: ${CASE}${NC}"
    echo "  Case name: ${CASE_NAME}"
    
    # 既存のmergedディレクトリを確認
    if [ -d "${OUTPUT_DIR_MERGED}" ] && [ "$(ls -A ${OUTPUT_DIR_MERGED} 2>/dev/null)" ]; then
        echo -e "  ${GREEN}✓ Already processed (merged directory exists with files)${NC}"
        echo "    Skipping..."
        echo ""
        continue
    fi
    
    # 1. Total task (臓器セグメンテーション) - fastオプションなし
    echo "  Running total segmentation task..."
    docker run --rm \
        -v /home/soya/ctrate_ws:/workspace \
        wasserth/totalsegmentator:${TOTALSEG_VERSION} \
        TotalSegmentator \
        -i ${INPUT_FILE} \
        -o ${OUTPUT_DIR_TOTAL} \
        --task total
    
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Total segmentation completed${NC}"
    else
        echo -e "  ${RED}✗ Total segmentation failed${NC}"
        continue
    fi
    
    # 2. Tissue types task (脂肪組織)
    echo "  Running tissue types task..."
    docker run --rm \
        -v /home/soya/ctrate_ws:/workspace \
        wasserth/totalsegmentator:${TOTALSEG_VERSION} \
        TotalSegmentator \
        -i ${INPUT_FILE} \
        -o ${OUTPUT_DIR_TISSUE} \
        --task tissue_types
    
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Tissue segmentation completed${NC}"
    else
        echo -e "  ${YELLOW}⚠ Tissue segmentation failed (may need license key)${NC}"
        echo "    Continuing with organ masks only..."
    fi
    
    # 3. マスクをマージ
    echo "  Merging masks..."
    mkdir -p ${OUTPUT_DIR_MERGED}
    
    # Total taskの結果をコピー
    TOTAL_DIR="/home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation"
    if [ -d "${TOTAL_DIR}" ]; then
        cp ${TOTAL_DIR}/*.nii.gz ${OUTPUT_DIR_MERGED}/ 2>/dev/null || true
        echo "    Copied organ masks"
    fi
    
    # Tissue taskの結果をコピー（存在する場合）
    TISSUE_DIR="/home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation_tissue"
    if [ -d "${TISSUE_DIR}" ]; then
        cp ${TISSUE_DIR}/*.nii.gz ${OUTPUT_DIR_MERGED}/ 2>/dev/null || true
        echo "    Copied tissue masks"
    fi
    
    # 結果を確認
    FILE_COUNT=$(ls -1 ${OUTPUT_DIR_MERGED}/*.nii.gz 2>/dev/null | wc -l)
    if [ ${FILE_COUNT} -gt 0 ]; then
        echo -e "  ${GREEN}✓ Successfully merged ${FILE_COUNT} mask files${NC}"
    else
        echo -e "  ${RED}✗ No mask files found in merged directory${NC}"
    fi
    
    echo ""
done

echo -e "${GREEN}=== Processing Complete ===${NC}"
echo ""

# 最終確認
echo "Summary of merged directories:"
for CASE in "${CASES[@]}"; do
    CASE_NAME=$(echo $CASE | sed 's/\//_/g')
    OUTPUT_DIR_MERGED="/home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation_merged"
    
    if [ -d "${OUTPUT_DIR_MERGED}" ]; then
        FILE_COUNT=$(ls -1 ${OUTPUT_DIR_MERGED}/*.nii.gz 2>/dev/null | wc -l)
        if [ ${FILE_COUNT} -gt 0 ]; then
            echo -e "  ${GREEN}✓${NC} ${CASE_NAME}: ${FILE_COUNT} files"
            
            # 重要なファイルの存在確認
            REQUIRED_FILES=("heart" "lung_left" "lung_right" "liver" "stomach" "spleen" "torso_fat")
            MISSING_FILES=""
            for FILE in "${REQUIRED_FILES[@]}"; do
                if [ ! -f "${OUTPUT_DIR_MERGED}/${FILE}.nii.gz" ]; then
                    MISSING_FILES="${MISSING_FILES} ${FILE}"
                fi
            done
            
            if [ -n "${MISSING_FILES}" ]; then
                echo -e "    ${YELLOW}⚠ Missing:${MISSING_FILES}${NC}"
            fi
        else
            echo -e "  ${RED}✗${NC} ${CASE_NAME}: No files found"
        fi
    else
        echo -e "  ${RED}✗${NC} ${CASE_NAME}: Directory not found"
    fi
done

echo ""
echo "Next steps:"
echo "1. Run EAT+PAT extraction v5.3 for each case:"
echo ""
for CASE in "${CASES[@]}"; do
    CASE_NAME=$(echo $CASE | sed 's/\//_/g')
    echo "python scripts/extract_eat_pat_improved_v5_3.py \\"
    echo "  /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/${CASE}.nii.gz \\"
    echo "  /home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation_merged \\"
    echo "  /home/soya/ctrate_ws/outputs/eat_pat_v5_3_${CASE_NAME} \\"
    echo "  --verbose"
    echo ""
done