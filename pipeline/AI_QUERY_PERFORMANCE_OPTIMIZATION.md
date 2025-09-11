# AI Query: CTスキャン処理パイプラインの飛躍的速度向上

## 背景・目的
数万例のCTスキャンを処理するパイプラインの実行速度を飛躍的に向上させたい。現在の実装では1例あたり15-30分かかり、数万例処理に数ヶ月〜年単位の時間が必要となってしまう。最終成果物（EAT/PATマスク、DRR画像、マスク投影画像）の品質は維持したまま、処理速度を大幅に改善する方法を知りたい。

## 現在の実装詳細

### システム環境
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **CPU**: 4コア
- **Memory**: 31GB
- **OS**: Linux (WSL2)
- **Docker**: GPU サポート有効

### 処理パイプライン構成（3ステップ）

#### Step 1: PAT+EAT抽出 (推定5-15分/例)

**TotalSegmentator実行 (2段階)**:
```bash
# 1. Total task（臓器セグメンテーション）
docker run --rm --gpus all \
  -v /home/soya/ctrate_ws:/workspace \
  wasserth/totalsegmentator:2.10.0 \
  TotalSegmentator \
  -i /workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz \
  -o /workspace/pipeline/test_outputs/single_case_test/step1_eat_pat/totalseg \
  --task total

# 2. Tissue types task（脂肪セグメンテーション）- ライセンス必要
docker run --rm --gpus all \
  -v /home/soya/ctrate_ws:/workspace \
  wasserth/totalsegmentator:2.10.0 \
  TotalSegmentator \
  -i /workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz \
  -o /workspace/pipeline/test_outputs/single_case_test/step1_eat_pat/totalseg_tissue \
  --task tissue_types -ta <LICENSE_KEY>

# 3. EAT/PAT抽出スクリプト実行
python3 scripts/extract_eat_pat_improved_v5_4.py \
  <CT_PATH> <TOTALSEG_DIR> <OUTPUT_DIR> --verbose
```

**EAT/PAT抽出スクリプト詳細 (scripts/extract_eat_pat_improved_v5_4.py)**:
```python
#!/usr/bin/env python3
"""
改善版v5.4 EAT+PAT抽出：Z軸処理改善版
主要な改善点：
1. 接触判定と禁止帯の順序を入れ替え（最重要）
2. シードだけZ範囲で制限し、伝播は3D全域を許可
3. 心膜近傍との連結性を採用条件に追加
4. 横隔膜の小孔をもう少し確実に塞ぐ
5. Z軸モードの追加（RPA→横隔膜、RPA→心尖、現行モード）
6. 横隔膜近似の実装による正確なZ範囲設定
"""

import numpy as np, nibabel as nib
from pathlib import Path
from scipy.ndimage import (
    binary_dilation, generate_binary_structure,
    distance_transform_edt, label, binary_propagation, binary_closing
)

# 重要パラメータ
DILATION_RADIUS_MM = 15.0  # 心臓マスク膨張半径 [mm] (EAT+PATを含む範囲)
DEFAULT_INFERIOR_BAND_MM = 50.0  # 下部帯域の高さ [mm]
DEFAULT_LUNG_CONTACT_MM = 1.0    # 肺接触判定の許容距離 [mm] (δ)
DEFAULT_ABDOMEN_GAP_MM = 1.0     # 腹部臓器除外の膨張距離 [mm] (γ)
DEFAULT_MAX_HEART_DIST_MM = 20.0 # 心臓からの最大距離 [mm] (ρ)

def check_required_masks(totalseg_dir):
    """必要なマスクファイルの存在を確認"""
    required_masks = {
        'heart': ['heart_myocardium.nii.gz', 'myocardium.nii.gz', 'heart.nii.gz'],
        'lung_left': ['lung_left.nii.gz'],
        'lung_right': ['lung_right.nii.gz'],
        'torso_fat': ['torso_fat.nii.gz'],  # tissue_types taskで生成
        'liver': ['liver.nii.gz'],
        'stomach': ['stomach.nii.gz'],
        'spleen': ['spleen.nii.gz']
    }
    # マスク存在チェック処理...
    
def extract_eat_pat_v5_4(ct_path, totalseg_dir, output_dir, **params):
    """v5.4改善版EAT+PAT抽出メイン処理"""
    # 1. 必要マスク読み込み
    # 2. 心臓Shell作成 (15mm膨張)
    # 3. 脂肪候補抽出 (torso_fat使用)
    # 4. 肺接触判定 (ILAM: Inferior Lung-Adjacency Mode)
    # 5. 腹部臓器除外
    # 6. Z軸範囲制限 (RPA→横隔膜モード)
    # 7. 連結性解析・最終選択
```

**使用技術**:
- TotalSegmentator 2.10.0 (深層学習ベースの臓器セグメンテーション) - **最重要ボトルネック**
- scipy.ndimage (形態学的処理、距離変換、ラベリング)
- nibabel (NIfTI画像処理)
- 複雑な3D形態学処理 (binary_dilation, distance_transform_edt)

#### Step 2: DRR作成 (推定2-5分/例)
```bash
docker run --rm --gpus all \
  -v /home/soya/ctrate_ws:/workspace \
  deepdrr-japan-mirror:latest bash -c "
    pip3 install torch torchvision pydicom opencv-python --no-cache-dir -i https://pypi.python.org/simple/ > /dev/null 2>&1 && 
    cd /workspace && 
    python3 deepdrr/test_deepdrr_final_pa_fixed.py"
```

**DeepDRRスクリプト詳細 (deepdrr/test_deepdrr_final_pa_fixed.py)**:
```python
#!/usr/bin/env python3
"""
DeepDRR PA左右反転修正版（final_correct.py改良）
技術仕様:
- 座標系: LPS（配列とaffine同期変換）
- 撮影角度: PA=α0°,β0°,γ90°、LAT=α0°,β±90°,γ90°
- PA左右修正: radiographic view（患者の左=画像の右）対応
- 投影制御: 幾何更新後2回実行（DeepDRR標準）
"""

from pathlib import Path
import numpy as np, imageio.v2 as imageio, nibabel as nib
from nibabel import orientations as nio
from nibabel.affines import voxel_sizes
import matplotlib; matplotlib.use('Agg')
from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector

def load_volume_LPS(nifti: Path) -> Volume:
    """配列とaffineの同期変換でLPS正規化"""
    img = nib.load(str(nifti))
    in_ornt = nio.io_orientation(img.affine)
    out_ornt = nio.axcodes2ornt(('L','P','S'))
    to_lps = nio.ornt_transform(in_ornt, out_ornt)
    
    data_lps = nio.apply_orientation(img.get_fdata().astype(np.float32), to_lps)
    aff_lps = img.affine @ nio.inv_ornt_aff(to_lps, img.shape)
    spacing = voxel_sizes(aff_lps)[:3]
    
    shp = np.array(data_lps.shape, dtype=float)
    origin = (-0.5 * (shp - 1.0) * spacing).tolist()
    
    return Volume.from_hu(
        hu_values=data_lps, origin=origin, spacing=spacing.tolist(),
        anatomical_coordinate_system="LPS",
    )

def make_carm(isocenter, sid=1800.0, oid=80.0, px=0.14, plate_mm=(350.0, 430.0)):
    """実機相当C-arm設定"""
    sad = sid - oid
    W = int(round(plate_mm[0] / px)) & ~1
    H = int(round(plate_mm[1] / px)) & ~1
    return MobileCArm(
        isocenter=isocenter, rotate_camera_left=True,
        source_to_detector_distance=sid,
        source_to_isocenter_vertical_distance=sad,
        pixel_size=px, sensor_width=W, sensor_height=H,
        min_alpha=-720, max_alpha=720, min_beta=-720, max_beta=720,
    )

def project(ct: Volume, carm: MobileCArm, alpha=0., beta=0., gamma=0., label=""):
    """投影実行（幾何更新後2回呼び出し）"""
    with Projector(ct, carm=carm) as projector:
        carm.move_to(alpha=alpha, beta=beta, gamma=gamma, degrees=True)
        _ = projector()  # warm-up
        img = projector().astype(np.float32)
    # PA左右反転修正
    if alpha == 0 and beta == 0:  # PA
        img = img[:, ::-1].copy()
    return img

# メイン処理: PA/LAT投影生成
def main():
    nifti = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    ct = load_volume_LPS(nifti)
    carm = make_carm(ct.center_in_world)
    ct.place_center(carm.isocenter_in_world)
    
    # PA（正面）+ LAT（側面）投影
    img_pa = project(ct, carm, alpha=0, beta=0, gamma=90, label="PA正面")
    img_lat = project(ct, carm, alpha=0, beta=90, gamma=90, label="LAT側面")
```

**使用技術**:
- DeepDRR (PyTorch, CUDA) - **GPU集約的**
- nibabel orientations (座標系変換)
- LPS座標系統一処理
- Projector warm-up（初回は時間がかかる）

#### Step 3: マスク投影 (推定3-10分/例)
```bash
docker run --rm --gpus all \
  -v /home/soya/ctrate_ws:/workspace \
  deepdrr-japan-mirror:latest bash -c "
    pip3 install torch torchvision pydicom opencv-python scipy --no-cache-dir -i https://pypi.python.org/simple/ > /dev/null 2>&1 && 
    cd /workspace && 
    python3 mask_projection/src/mask_deepdrr_projection_thickness_gpt5pro8.py"
```

**マスク投影スクリプト詳細 (mask_projection/src/mask_deepdrr_projection_thickness_gpt5pro8.py)**:
```python
#!/usr/bin/env python3
"""
Plan-A: 単色"実効"モデルで画素値=厚み[mm] (gpt5pro8物理補正版)
- I_air (flat-field) を同一幾何で取得 + 健全性チェック
- G = -log( I_mask / I_air ) with valid condition
- μ_eff を体積一致で自己較正
- T_mm = G / μ_eff（Ω=ロバストシルエット内、外は0）
- ∑T_mm * p^2 = V_3D を常に満たす

根拠: Beer–Lambert (mono): -log(I/I0) = μ * L → 厚みに線形
"""

from pathlib import Path
import numpy as np, nibabel as nib
from nibabel import orientations as nio
from nibabel.affines import voxel_sizes
import imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing
from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector

def load_volume_LPS(nifti: Path) -> Volume:
    """LPS座標系正規化"""
    img = nib.load(str(nifti))
    in_ornt = nio.io_orientation(img.affine)
    out_ornt = nio.axcodes2ornt(("L","P","S"))
    to_lps = nio.ornt_transform(in_ornt, out_ornt)
    data_lps = nio.apply_orientation(img.get_fdata().astype(np.float32), to_lps)
    # 以下省略...

def load_mask_volume_LPS(mask_nifti: Path) -> Volume:
    """マスクをLPS座標系に変換"""
    img_mask = nib.load(str(mask_nifti))
    # 座標系変換...
    # gpt5pro8改良: 現実的なHU（背景=-1000, 内部=-50）
    hu = np.full(mask_lps.shape, -1000.0, dtype=np.float32)
    hu[mask_lps > 0.5] = -50.0  # 脂肪組織の現実的なHU値
    # Volume作成...

def render_energy(vol: Volume, carm: MobileCArm, alpha=0, beta=0, gamma=90, flip_pa=True):
    """同一幾何のエナジー画像（生成）"""
    with Projector(vol, carm=carm) as projector:
        carm.move_to(alpha=alpha, beta=beta, gamma=gamma, degrees=True)
        _ = projector()           # warm-up
        img = projector().astype(np.float32)  # [0..1] transmission-like
    if flip_pa:
        img = img[:, ::-1].copy() # radiographic view: 患者左=画像右
    return img

def get_flat_field(vol: Volume, carm: MobileCArm, view, outdir, flip_pa=True):
    """Flat-field (I_air) 生成・キャッシュ"""
    # 空気Volume作成
    air_vol = make_air_volume_like(vol)
    # 同一幾何で投影
    I_air = render_energy(air_vol, carm, *view, flip_pa=flip_pa)
    return I_air

def thickness_estimation_plan_a(I_mask, I_air, mask_vol: Volume, carm: MobileCArm):
    """Plan-A: 単色実効モデルによる厚み推定"""
    # 1. 有効性チェック
    valid = (I_air > 1e-8) & (I_mask > 1e-8) & (I_mask <= I_air * 1.01)
    
    # 2. G = -log(I_mask / I_air)
    G = np.zeros_like(I_mask)
    G[valid] = -np.log(I_mask[valid] / I_air[valid])
    
    # 3. μ_eff 自己較正（体積一致）
    V_3D = np.sum(mask_vol.data > 0.5) * np.prod(mask_vol.spacing)  # [mm³]
    A_iso, px_iso, M = pixel_area_at_isocenter(carm)
    S_2D = np.sum(G > 0) * A_iso  # [mm²]
    
    if S_2D > 0:
        mu_eff = np.sum(G) / (V_3D / px_iso)  # 体積一致較正
        T_mm = G / mu_eff
    else:
        T_mm = np.zeros_like(G)
    
    return T_mm, mu_eff

# メイン処理
def main():
    ct_nifti   = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    mask_nifti = Path("/workspace/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz")
    
    # 1. ボリューム読み込み
    ct_vol   = load_volume_LPS(ct_nifti)
    mask_vol = load_mask_volume_LPS(mask_nifti)
    carm = make_carm(ct_vol.center_in_world)
    
    # 2. 投影画像生成
    view = (0.0, 0.0, 90.0)  # PA
    I_mask = render_energy(mask_vol, carm, *view, flip_pa=True)
    I_air  = get_flat_field(mask_vol, carm, view, outdir, flip_pa=True)
    
    # 3. 厚み推定
    T_mm, mu_eff = thickness_estimation_plan_a(I_mask, I_air, mask_vol, carm)
    
    # 4. 可視化・保存
    save_thickness_map(T_mm, outdir / "thickness_map_pa.png")
```

**使用技術**:
- DeepDRR投影エンジン - **GPU集約的**
- Beer-Lambert則による厚み推定
- 単色実効モデル（μ_eff自己較正）
- Flat-field正規化（I_air生成）
- 複雑な物理計算（体積一致較正）

### 現在のパイプライン実装コード

```python
class CTProcessingPipeline:
    def run_step1_eat_pat_extraction(self) -> bool:
        # TotalSegmentator呼び出し
        project_root = Path("/home/soya/ctrate_ws")
        cmd1 = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{project_root}:/workspace",
            "wasserth/totalsegmentator:2.10.0",
            "TotalSegmentator",
            "-i", f"/workspace/{self.ct_path.relative_to(project_root)}",
            "-o", f"/workspace/{totalseg_dir.relative_to(project_root)}",
            "--task", "total"
        ]
        result = subprocess.run(cmd1, capture_output=True, text=True)
        
        # EAT/PAT抽出スクリプト実行
        cmd2 = [
            "python3", str(eat_pat_script),
            str(self.ct_path), str(totalseg_dir), str(self.step1_output),
            "--verbose" if self.verbose else ""
        ]
        result = subprocess.run(cmd2, capture_output=True, text=True)

    def run_step2_drr_generation(self) -> bool:
        # 動的スクリプト生成 (パス置換)
        temp_script = self.step2_output / "temp_deepdrr_script.py"
        script_content = script_content.replace(old_path, new_path)
        
        # Docker実行
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", "/home/soya/ctrate_ws:/workspace",
            "deepdrr-japan-mirror:latest", "bash", "-c",
            "pip3 install ... && python3 temp_script.py"
        ]

    def run_step3_mask_projection(self) -> bool:
        # 同様の動的スクリプト生成とDocker実行
```

### 性能ボトルネック分析（詳細）

1. **TotalSegmentator** (最大ボトルネック):
   - 深層学習推論: nnUNet-based 3D CNN、GPU VRAM 20GB+使用
   - **2つのタスク実行**: total + tissue_types（ライセンス必要）
   - 単一症例で10-15分、モデルロード・推論・後処理含む

2. **Docker起動オーバーヘッド**:
   - 各ステップで新コンテナ起動（30-60秒）
   - pip install毎回実行（torch, torchvision等大容量）
   - イメージpull・コンテナ初期化コスト

3. **DeepDRR投影処理**:
   - Projector warm-up（初回起動コスト）
   - 高解像度投影（2500x3070ピクセル）
   - 複数ビュー生成（PA + LAT）
   - CUDA メモリ転送・線積分計算

4. **データI/O**:
   - 大容量NIfTI読み書き（300MB+/ファイル）
   - 中間ファイル生成・保存（マスク、投影画像）
   - 座標系変換処理（LPS正規化）

5. **複雑な後処理**:
   - scipy.ndimage演算（距離変換、形態学処理）
   - 3D連結性解析・ラベリング
   - マスク投影の物理計算（Beer-Lambert、μ_eff較正）

6. **シーケンシャル処理**:
   - ステップ間の厳密な依存関係
   - GPU アイドル時間（CPU処理中）
   - メモリ・ストレージアクセス競合

### リソース使用状況
```json
{
  "gpu_usage": [{"memory_percent": 95.0, "gpu_utilization": 95}],
  "cpu_percent": 62.1,
  "memory_percent": 33.4
}
```

## 最適化要求

### 目標
- **現在**: 15-30分/例 → **目標**: 1-5分/例 (3-10倍高速化)
- 数万例処理を現実的な時間内（数週間〜数ヶ月）で完了
- 最終成果物の品質維持

### 制約
- 単一GPU環境 (RTX 3090)
- 最終成果物（マスク、DRR、投影画像）の品質は同等レベル維持
- 既存のDockerイメージ・ライブラリ構成の大幅変更は避けたい

## 質問内容

### A. TotalSegmentatorの最適化（最重要）
1. **バッチ推論**: 複数CTを同時処理してGPU効率向上は可能？
2. **モデル軽量化**: nnUNet-based 3D CNNの高速版やDistillation版は？
3. **タスク統合**: total + tissue_typesの同時実行で時間短縮は？
4. **代替セグメンテーション**: より高速なCT臓器セグメンテーション手法は？
5. **メモリ最適化**: VRAM 24GB制約での推論最適化手法は？

### B. Docker・環境最適化
1. **永続化コンテナ**: デーモン化してセットアップコスト削減は？
2. **事前ビルドイメージ**: torch等事前インストール済みイメージ作成は？  
3. **マルチステージビルド**: 各ステップ専用の軽量イメージ構築は？
4. **GPU共有**: 単一コンテナ内での全ステップ実行は可能？

### C. パイプライン・並列化戦略  
1. **ステップ間パイプライン**: Step1完了時にStep2開始する並列実行は？
2. **GPU利用率最大化**: TotalSegmentator実行中の他処理は？
3. **CPU-GPU協調**: CPU集約処理とGPU処理の効率的分散は？
4. **メモリ内データ渡し**: ディスクI/O削減でGPU VRAM直接連携は？

### D. データI/O・ストレージ最適化
1. **NIfTI読み込み高速化**: メモリマップ、遅延読み込み、圧縮活用は？
2. **中間データ形式**: HDF5、Zarr等の高速フォーマット移行は？  
3. **キャッシュ戦略**: 座標変換結果、セグメンテーション結果の再利用は？
4. **ストレージ階層**: SSD、RAM disk活用での I/O ボトルネック解消は？

### E. アルゴリズム・計算最適化
1. **DeepDRR高速化**: 投影解像度動的調整、近似計算手法は？
2. **座標変換最適化**: LPS変換の事前計算・キャッシュ化は？
3. **形態学処理高速化**: scipy.ndimage代替のGPU実装は？
4. **物理計算簡略化**: Beer-Lambert計算の近似・ルックアップテーブル化は？

### F. バッチ・スケール処理戦略
1. **インテリジェントバッチング**: 類似症例グルーピングで処理効率化は？
2. **動的リソース管理**: GPU・CPUリソースの症例別最適配分は？
3. **段階的品質設定**: 重要症例は高品質、通常症例は高速処理の使い分けは？
4. **分散処理**: 複数GPUノード、クラウド並列処理への拡張は？

### G. 品質維持との両立
1. **品質評価指標**: 高速化による品質劣化の定量評価方法は？
2. **段階的最適化**: Step1高速化 → Step2最適化の優先順位は？
3. **検証・テスト**: 最適化版と原版の結果同等性確認手法は？

これらについて、公式実装を参考にした解決策を教えてください。