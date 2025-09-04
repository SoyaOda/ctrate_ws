# DeepDRR使用方法ガイド

このドキュメントでは、CT-RATE v2データからDeepDRRを使用して高品質なDRR（Digitally Reconstructed Radiograph）を生成する方法を説明します。

## 📋 目次

- [実行環境](#実行環境)
- [クイックスタート](#クイックスタート)
- [段階別実装](#段階別実装)
- [トラブルシューティング](#トラブルシューティング)
- [カスタマイズ](#カスタマイズ)

## 🚀 実行環境

### 必要な環境
- **GPU**: NVIDIA RTX 3090以上推奨（11GB+ VRAM）
- **Docker**: GPUサポート有効
- **データ**: CT-RATE v2 NIfTIファイル

### 環境確認
```bash
# GPU確認
nvidia-smi

# Dockerでのテスト
cd /home/soya/ctrate_ws/deepdrr
./run_deepdrr_docker.sh
```

## ⚡ クイックスタート

### 1. 最高品質（臨床仕様版）
**実際の医療機器と同等の品質**

```bash
# Docker環境で実行
docker exec deepdrr-test python3 test_deepdrr_clinical.py
```

**出力:**
- `deepdrr_PA_clinical.png` (9MB) - PA view PNG
- `deepdrr_LAT_clinical.png` (10MB) - LAT view PNG  
- `deepdrr_PA_clinical.dcm` (15MB) - PA view DICOM
- `deepdrr_LAT_clinical.dcm` (15MB) - LAT view DICOM
- `deepdrr_clinical_comparison.png` - 比較画像

**仕様:**
- 検出器: 35×43cm, 0.14mm/pixel
- 解像度: 2500×3072 pixels
- DICOM: MONOCHROME2, ViewPosition準拠
- 撮影法: PA（標準）+ 左側面LAT

### 2. 標準品質（最終修正版）
**問題解決済みの高品質版**

```bash
docker exec deepdrr-test python3 test_deepdrr_final.py
```

**出力:**
- `deepdrr_PA_final.png` (2.7MB) - PA view
- `deepdrr_LAT_final.png` (2.8MB) - LAT view
- 解像度: 2048×2048 pixels

## 📚 段階別実装

### レベル1: 基本テスト
```bash
python3 test_deepdrr_cuda11.py
```
- DeepDRRの動作確認
- GPU環境テスト

### レベル2: 改善版  
```bash
python3 test_deepdrr_fixed.py
```
- SID=180cm, 高解像度
- パーセンタイル窓によるコントラスト改善

### レベル3: 問題解決版
```bash
python3 test_deepdrr_final.py
```
- 白黒・左右・拡大の3つの問題を解決
- Energy表示による正しい階調

### レベル4: 臨床仕様版 ⭐
```bash
python3 test_deepdrr_clinical.py
```
- 実機準拠の35×43cm検出器
- DICOM出力対応
- 医療グレード品質

## 🔧 各実装の特徴比較

| 実装版 | 解像度 | 品質 | 特徴 | 用途 |
|--------|--------|------|------|------|
| **臨床仕様版** | 2500×3072 | 医療グレード | DICOM対応、実機仕様 | 研究・臨床応用 |
| 最終修正版 | 2048×2048 | 高品質 | 問題解決済み | 一般的な研究用途 |
| 改善版 | 2048×2048 | 標準 | 基本的な改善 | 開発・テスト |
| 基本版 | 1536×1536 | 基本 | 動作確認用 | 環境テスト |

## 🛠️ トラブルシューティング

### よくある問題と解決策

#### 1. CUDA 12.9環境で動作しない
**エラー:** `identifier 'texture' is undefined`

**解決策:**
```bash
# Docker環境を使用（推奨）
./run_deepdrr_docker.sh

# または conda環境
./setup_conda_env.sh
conda activate deepdrr-cuda11
```

#### 2. GPU メモリ不足
**エラー:** `CUDA out of memory`

**解決策:**
```python
# 解像度を下げる
sensor_width=1536, sensor_height=1536
pixel_size=0.20  # 0.14 → 0.20
```

#### 3. 画像が黒い・白い
**原因:** パーセンタイル窓の設定

**解決策:**
```python
# 窓を調整
energy_to_uint16(img, window=(0.5, 99.7), gamma=1.2)
```

#### 4. 左右が逆
**解決策:**
```python
# C-arm設定を変更
rotate_camera_left=False  # True ⟷ False
# または
img = img[:, ::-1]  # 後処理で反転
```

#### 5. 拡大率が大きい
**解決策:**
```python
# SADをSIDに近づける
make_cxr_carm_portrait(sid_mm=1800, oid_mm=60)  # OIDを小さく
```

## 🎛️ カスタマイズ

### パラメータ調整

#### 撮影幾何
```python
# SID/SAD/OID調整
SID = 1800.0  # Source-Image Distance
OID = 80.0    # Object-Image Distance  
SAD = SID - OID  # Source-Axis Distance

# 倍率 = SID/SAD
magnification = SID / SAD  # 1.047
```

#### 検出器仕様
```python
# 解像度・サイズ調整
pixel_size = 0.14        # mm/pixel
plate_size = (350, 430)  # mm (width, height)
sensor_size = (2500, 3072)  # pixels
```

#### 画質調整
```python
# コントラスト・階調
window = (1.0, 99.5)  # パーセンタイル範囲
gamma = 1.0           # ガンマ補正（>1でソフト、<1でハード）
```

### 撮影角度

#### 標準角度
```python
# PA (Posterior-Anterior): 正面
alpha=0.0, beta=0.0

# LAT (Lateral): 側面
alpha=90.0, beta=0.0   # 左側面
alpha=-90.0, beta=0.0  # 右側面

# AP (Anterior-Posterior): 前後
alpha=180.0, beta=0.0
```

#### カスタム角度
```python
# 斜位撮影
alpha=45.0, beta=0.0   # 45度斜位

# 頭尾方向傾斜
alpha=0.0, beta=15.0   # 15度頭側傾斜
```

### DICOM設定

#### メタデータカスタマイズ
```python
# 患者情報
ds.PatientName = "YOUR^PATIENT"
ds.PatientID = "YOUR_ID_0001"

# 撮影情報
ds.ViewPosition = "PA"  # "PA", "LATERAL", "AP"
ds.DistanceSourceToDetector = 1800.0
ds.DistanceSourceToPatient = 1720.0
```

## 📖 技術詳細

### DeepDRR出力の解釈
- **出力値**: Energy deposited（エネルギー堆積）
- **表示方法**: -log変換なしの線形表示を推奨
- **理由**: DeepDRRの物理モデルに基づく

### 幾何学的考慮事項
- **PA vs AP**: PAは心拡大を抑制（患者を検出器に近づける）
- **倍率**: M = SID/SOD, 低倍率にはOIDを小さく
- **焦点サイズ**: 1.0-1.2mmが一般的

### DICOM準拠
- **PhotometricInterpretation**: MONOCHROME2（値↑=白）
- **ViewPosition**: PA/LATERAL/AP明記
- **PresentationIntentType**: FOR PRESENTATION

## 📞 サポート

### 問題報告
- GitHub Issues: https://github.com/SoyaOda/ctrate_ws/issues
- 実行ログと環境情報を含めてください

### 参考資料
- [DeepDRR公式](https://deepdrr.readthedocs.io/)
- [CT-RATE データセット](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- `md_files/answers/` - 問題解決の詳細解説

---

## 🏁 最後に

この実装により、CT-RATE v2データから医療グレードのDRRが生成できます。研究用途から臨床応用まで、幅広い用途にご活用ください。

**推奨実行順序:**
1. 環境テスト: `test_deepdrr_cuda11.py`
2. 品質確認: `test_deepdrr_final.py`  
3. 本格運用: `test_deepdrr_clinical.py` ⭐

質問や問題があれば、遠慮なくお聞かせください！