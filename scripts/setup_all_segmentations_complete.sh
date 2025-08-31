#!/bin/bash
# すべての症例に対してTotalSegmentator（全タスク）を実行するスクリプト
# total, tissue_types, heartchambers_highresを含む完全セットアップ

set -e  # エラーが発生したら停止

# 色付き出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Complete TotalSegmentator Setup for All Cases ===${NC}"
echo ""

# ライセンスキーを.envから読み込み
if [ -f "/home/soya/ctrate_ws/.env" ]; then
    source /home/soya/ctrate_ws/.env
    if [ -z "$TOTALSEG_LICENSE_KEY" ]; then
        echo -e "${RED}Error: TOTALSEG_LICENSE_KEY not found in .env${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ License key loaded: ${TOTALSEG_LICENSE_KEY}${NC}"
else
    echo -e "${RED}Error: .env file not found${NC}"
    exit 1
fi

# Docker確認
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not available in WSL2${NC}"
    echo "Please ensure Docker Desktop is running and WSL2 integration is enabled"
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

echo ""
echo -e "${BLUE}=== Processing Configuration ===${NC}"
echo "  TotalSegmentator version: ${TOTALSEG_VERSION}"
echo "  License key: ${TOTALSEG_LICENSE_KEY}"
echo "  Cases to process: ${#CASES[@]}"
echo ""

# 各症例を処理
for CASE in "${CASES[@]}"; do
    # パスを分解 - パスの最初と最後の部分を組み合わせ（例: valid_1_valid_1_a_1）
    FIRST_PART=$(echo $CASE | cut -d'/' -f1)
    LAST_PART=$(basename $CASE)
    CASE_NAME="${FIRST_PART}_${LAST_PART}"
    INPUT_FILE="/workspace/data/CT-RATE-v2/dataset/valid_fixed/${CASE}.nii.gz"
    OUTPUT_DIR_TOTAL="/workspace/outputs/batch_results/${CASE_NAME}/segmentation"
    OUTPUT_DIR_TISSUE="/workspace/outputs/batch_results/${CASE_NAME}/segmentation_tissue"
    OUTPUT_DIR_HEART="/workspace/outputs/batch_results/${CASE_NAME}/segmentation_heart"
    OUTPUT_DIR_MERGED="/home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation_merged"
    
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Processing: ${CASE}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "  Case name: ${CASE_NAME}"
    
    # 既存のmergedディレクトリを確認
    if [ -d "${OUTPUT_DIR_MERGED}" ]; then
        # 必要なファイルの存在確認
        REQUIRED_FILES=(
            "heart_myocardium.nii.gz"
            "heart_atrium_left.nii.gz"
            "heart_atrium_right.nii.gz"
            "heart_ventricle_left.nii.gz"
            "heart_ventricle_right.nii.gz"
            "torso_fat.nii.gz"
            "liver.nii.gz"
            "stomach.nii.gz"
            "spleen.nii.gz"
            "lung_left.nii.gz"
            "lung_right.nii.gz"
        )
        
        ALL_PRESENT=true
        for FILE in "${REQUIRED_FILES[@]}"; do
            if [ ! -f "${OUTPUT_DIR_MERGED}/${FILE}" ]; then
                ALL_PRESENT=false
                echo -e "  ${YELLOW}⚠ Missing: ${FILE}${NC}"
            fi
        done
        
        if [ "$ALL_PRESENT" = true ]; then
            echo -e "  ${GREEN}✓ Already completely processed (all required files present)${NC}"
            echo ""
            continue
        else
            echo "  Some required files missing, reprocessing..."
        fi
    fi
    
    # mergedディレクトリを作成
    mkdir -p ${OUTPUT_DIR_MERGED}
    
    # 1. Total task (臓器セグメンテーション)
    echo ""
    echo -e "${BLUE}[1/3] Running total segmentation task...${NC}"
    
    TOTAL_DIR="/home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation"
    if [ -d "${TOTAL_DIR}" ] && [ "$(ls -A ${TOTAL_DIR}/*.nii.gz 2>/dev/null | wc -l)" -gt 50 ]; then
        echo -e "  ${GREEN}✓ Total segmentation already exists${NC}"
    else
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
    fi
    
    # 2. Tissue types task (脂肪組織) - ライセンス付き
    echo ""
    echo -e "${BLUE}[2/3] Running tissue types task (with license)...${NC}"
    
    TISSUE_DIR="/home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation_tissue"
    if [ -f "${TISSUE_DIR}/torso_fat.nii.gz" ]; then
        echo -e "  ${GREEN}✓ Tissue segmentation already exists${NC}"
    else
        docker run --rm \
            -v /home/soya/ctrate_ws:/workspace \
            wasserth/totalsegmentator:${TOTALSEG_VERSION} \
            TotalSegmentator \
            -i ${INPUT_FILE} \
            -o ${OUTPUT_DIR_TISSUE} \
            --task tissue_types \
            -l ${TOTALSEG_LICENSE_KEY}
        
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}✓ Tissue segmentation completed${NC}"
        else
            echo -e "  ${YELLOW}⚠ Tissue segmentation failed${NC}"
            echo "    This might be due to an invalid license key"
        fi
    fi
    
    # 3. Heart chambers highres task (心臓詳細) - ライセンス付き
    echo ""
    echo -e "${BLUE}[3/3] Running heart chambers highres task...${NC}"
    
    HEART_DIR="/home/soya/ctrate_ws/outputs/batch_results/${CASE_NAME}/segmentation_heart"
    if [ -f "${HEART_DIR}/heart_myocardium.nii.gz" ] || [ -f "${HEART_DIR}/myocardium.nii.gz" ]; then
        echo -e "  ${GREEN}✓ Heart chambers segmentation already exists${NC}"
    else
        docker run --rm \
            -v /home/soya/ctrate_ws:/workspace \
            wasserth/totalsegmentator:${TOTALSEG_VERSION} \
            TotalSegmentator \
            -i ${INPUT_FILE} \
            -o ${OUTPUT_DIR_HEART} \
            --task heartchambers_highres \
            -l ${TOTALSEG_LICENSE_KEY}
        
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}✓ Heart chambers segmentation completed${NC}"
        else
            echo -e "  ${YELLOW}⚠ Heart chambers segmentation failed${NC}"
            echo "    Trying without license key..."
            
            # ライセンスなしで再試行
            docker run --rm \
                -v /home/soya/ctrate_ws:/workspace \
                wasserth/totalsegmentator:${TOTALSEG_VERSION} \
                TotalSegmentator \
                -i ${INPUT_FILE} \
                -o ${OUTPUT_DIR_HEART} \
                --task heartchambers_highres
            
            if [ $? -eq 0 ]; then
                echo -e "  ${GREEN}✓ Heart chambers segmentation completed (no license needed)${NC}"
            else
                echo -e "  ${RED}✗ Heart chambers segmentation failed${NC}"
            fi
        fi
    fi
    
    # 4. マスクをマージ
    echo ""
    echo -e "${BLUE}Merging all masks...${NC}"
    
    # Total taskの結果をコピー
    if [ -d "${TOTAL_DIR}" ]; then
        cp ${TOTAL_DIR}/*.nii.gz ${OUTPUT_DIR_MERGED}/ 2>/dev/null || true
        echo "  Copied organ masks from total task"
    fi
    
    # Tissue taskの結果をコピー
    if [ -d "${TISSUE_DIR}" ]; then
        cp ${TISSUE_DIR}/*.nii.gz ${OUTPUT_DIR_MERGED}/ 2>/dev/null || true
        echo "  Copied tissue masks"
    fi
    
    # Heart chambers taskの結果をコピー（heart_プレフィックスを追加）
    if [ -d "${HEART_DIR}" ]; then
        # プレフィックスなしのファイルがある場合
        for file in ${HEART_DIR}/*.nii.gz; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                # heart_プレフィックスがない場合は追加
                if [[ ! "$filename" =~ ^heart_ ]]; then
                    # myocardium.nii.gz -> heart_myocardium.nii.gz
                    cp "$file" "${OUTPUT_DIR_MERGED}/heart_${filename}" 2>/dev/null || true
                else
                    cp "$file" "${OUTPUT_DIR_MERGED}/" 2>/dev/null || true
                fi
            fi
        done
        echo "  Copied heart chamber masks (with heart_ prefix)"
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

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}=== Processing Complete ===${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
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
            CRITICAL_FILES=(
                "heart_myocardium.nii.gz"
                "torso_fat.nii.gz"
                "liver.nii.gz"
                "lung_left.nii.gz"
                "lung_right.nii.gz"
            )
            
            MISSING_FILES=""
            for FILE in "${CRITICAL_FILES[@]}"; do
                if [ ! -f "${OUTPUT_DIR_MERGED}/${FILE}" ]; then
                    MISSING_FILES="${MISSING_FILES} ${FILE}"
                fi
            done
            
            if [ -n "${MISSING_FILES}" ]; then
                echo -e "    ${YELLOW}⚠ Missing critical files:${MISSING_FILES}${NC}"
            else
                echo -e "    ${GREEN}✓ All critical files present${NC}"
            fi
        else
            echo -e "  ${RED}✗${NC} ${CASE_NAME}: No files found"
        fi
    else
        echo -e "  ${RED}✗${NC} ${CASE_NAME}: Directory not found"
    fi
done

echo ""
echo -e "${BLUE}Next steps: Run EAT+PAT extraction v5.3 for each case:${NC}"
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