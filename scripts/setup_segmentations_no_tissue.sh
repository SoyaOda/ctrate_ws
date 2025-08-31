#!/bin/bash
# すべての症例に対してTotalSegmentator（臓器のみ）を実行するスクリプト
# tissue_typesはライセンスが必要なためスキップ

set -e  # エラーが発生したら停止

# 色付き出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== TotalSegmentator Setup (Organs Only) ===${NC}"
echo -e "${YELLOW}Note: Tissue types task requires license and will be skipped${NC}"
echo ""

# Docker確認
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not available in WSL2${NC}"
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
    OUTPUT_DIR_MERGED="/home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation_merged"
    
    echo -e "${YELLOW}Processing: ${CASE}${NC}"
    echo "  Case name: ${CASE_NAME}"
    
    # 既存のsegmentationディレクトリを確認
    SEGMENTATION_DIR="/home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation"
    if [ -d "${SEGMENTATION_DIR}" ] && [ "$(ls -A ${SEGMENTATION_DIR}/*.nii.gz 2>/dev/null | wc -l)" -gt 10 ]; then
        echo -e "  ${GREEN}✓ Already processed (segmentation exists)${NC}"
        
        # マージディレクトリが存在しない場合は作成
        if [ ! -d "${OUTPUT_DIR_MERGED}" ]; then
            echo "  Creating merged directory..."
            mkdir -p ${OUTPUT_DIR_MERGED}
            cp ${SEGMENTATION_DIR}/*.nii.gz ${OUTPUT_DIR_MERGED}/
            echo -e "  ${GREEN}✓ Copied to merged directory${NC}"
        fi
        echo ""
        continue
    fi
    
    # Total task (臓器セグメンテーション) - fastオプションを追加して高速化
    echo "  Running total segmentation task (with --fast option for speed)..."
    docker run --rm \
        -v /home/soya/ctrate_ws:/workspace \
        wasserth/totalsegmentator:${TOTALSEG_VERSION} \
        TotalSegmentator \
        -i ${INPUT_FILE} \
        -o ${OUTPUT_DIR_TOTAL} \
        --task total \
        --fast
    
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Total segmentation completed${NC}"
    else
        echo -e "  ${RED}✗ Total segmentation failed${NC}"
        continue
    fi
    
    # マスクをマージディレクトリにコピー
    echo "  Creating merged directory..."
    mkdir -p ${OUTPUT_DIR_MERGED}
    
    if [ -d "${SEGMENTATION_DIR}" ]; then
        cp ${SEGMENTATION_DIR}/*.nii.gz ${OUTPUT_DIR_MERGED}/ 2>/dev/null || true
        echo "    Copied organ masks"
    fi
    
    # 結果を確認
    FILE_COUNT=$(ls -1 ${OUTPUT_DIR_MERGED}/*.nii.gz 2>/dev/null | wc -l)
    if [ ${FILE_COUNT} -gt 0 ]; then
        echo -e "  ${GREEN}✓ Successfully copied ${FILE_COUNT} mask files${NC}"
    else
        echo -e "  ${RED}✗ No mask files found in merged directory${NC}"
    fi
    
    echo ""
done

echo -e "${GREEN}=== Processing Complete ===${NC}"
echo ""

# torso_fatファイルを別途コピー（valid_1_a_2から他の症例へ）
echo "Copying torso_fat from valid_1_a_2 to other cases..."
SOURCE_FAT="/home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_valid_1_a_2/segmentation_merged/torso_fat.nii.gz"

if [ -f "${SOURCE_FAT}" ]; then
    for CASE in "${CASES[@]}"; do
        CASE_NAME=$(echo $CASE | sed 's/\//_/g')
        if [ "${CASE_NAME}" != "valid_1_valid_1_a_valid_1_a_2" ]; then
            DEST_DIR="/home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation_merged"
            if [ -d "${DEST_DIR}" ] && [ ! -f "${DEST_DIR}/torso_fat.nii.gz" ]; then
                echo -e "  ${YELLOW}⚠ Note: torso_fat.nii.gz needs to be generated separately for ${CASE_NAME}${NC}"
                echo "    (Tissue types task requires license)"
            fi
        fi
    done
else
    echo -e "${YELLOW}⚠ torso_fat.nii.gz not found in valid_1_a_2${NC}"
fi

echo ""

# 最終確認
echo "Summary of segmentation directories:"
for CASE in "${CASES[@]}"; do
    CASE_NAME=$(echo $CASE | sed 's/\//_/g')
    OUTPUT_DIR_MERGED="/home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation_merged"
    
    if [ -d "${OUTPUT_DIR_MERGED}" ]; then
        FILE_COUNT=$(ls -1 ${OUTPUT_DIR_MERGED}/*.nii.gz 2>/dev/null | wc -l)
        if [ ${FILE_COUNT} -gt 0 ]; then
            echo -e "  ${GREEN}✓${NC} ${CASE_NAME}: ${FILE_COUNT} files"
            
            # 重要なファイルの存在確認
            REQUIRED_FILES=("heart" "lung_left" "lung_right" "liver" "stomach" "spleen")
            MISSING_FILES=""
            for FILE in "${REQUIRED_FILES[@]}"; do
                if [ ! -f "${OUTPUT_DIR_MERGED}/${FILE}.nii.gz" ]; then
                    MISSING_FILES="${MISSING_FILES} ${FILE}"
                fi
            done
            
            if [ -n "${MISSING_FILES}" ]; then
                echo -e "    ${YELLOW}⚠ Missing:${MISSING_FILES}${NC}"
            fi
            
            # torso_fatの確認
            if [ ! -f "${OUTPUT_DIR_MERGED}/torso_fat.nii.gz" ]; then
                echo -e "    ${RED}⚠ Missing: torso_fat.nii.gz (required for EAT+PAT extraction)${NC}"
            fi
        else
            echo -e "  ${RED}✗${NC} ${CASE_NAME}: No files found"
        fi
    else
        echo -e "  ${RED}✗${NC} ${CASE_NAME}: Directory not found"
    fi
done

echo ""
echo -e "${YELLOW}Important: torso_fat.nii.gz is required for EAT+PAT extraction${NC}"
echo "If missing, you need to:"
echo "1. Get academic license from: https://backend.totalsegmentator.com/license-academic/"
echo "2. Run tissue_types task with license key"
echo "OR"
echo "3. Use alternative fat segmentation method"