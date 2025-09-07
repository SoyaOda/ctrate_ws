#!/bin/bash
# DeepDRRマスク投射をDocker環境で実行

echo "🐳 DeepDRRマスク投射Docker実行"
echo "📋 README.md準拠のCUDA 11.8環境使用"
echo "="*60

# パスをDockerマウント用に調整
sed -i 's|/home/soya/ctrate_ws/data/CT-RATE-v2|/workspace/data/CT-RATE-v2|g' /home/soya/ctrate_ws/mask_projection/src/mask_deepdrr_projection.py
sed -i 's|/home/soya/ctrate_ws/outputs|/home/soya/ctrate_ws/outputs|g' /home/soya/ctrate_ws/mask_projection/src/mask_deepdrr_projection.py
sed -i 's|/home/soya/ctrate_ws/mask_projection/outputs|/workspace/mask_projection/outputs|g' /home/soya/ctrate_ws/mask_projection/src/mask_deepdrr_projection.py

echo "🔄 パス調整完了"

# Docker実行コマンド（README.md準拠）
docker run --rm --gpus all \
  -v /home/soya/ctrate_ws/data:/workspace/data \
  -v /home/soya/ctrate_ws/deepdrr:/workspace \
  -v /home/soya/ctrate_ws/mask_projection:/workspace/mask_projection \
  -v /home/soya/ctrate_ws/outputs:/home/soya/ctrate_ws/outputs \
  deepdrr-japan-mirror:latest bash -c "
    pip3 install torch torchvision pydicom opencv-python --no-cache-dir -i https://pypi.python.org/simple/ > /dev/null 2>&1 &&
    echo '🧪 DeepDRRマスク投射実行開始' &&
    cd /workspace &&
    python3 mask_projection/src/mask_deepdrr_projection.py
  "

echo "🏁 DeepDRRマスク投射Docker実行完了"