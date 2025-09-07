# DiffDRRマスク投影パフォーマンス問題の包括的解決策Query

## 背景と問題詳細

### プロジェクト概要
- **目標**: 既存のDeepDRR PA投影画像(`/home/soya/ctrate_ws/deepdrr/deepdrr/outputs/deepdrr_PA_final_pa_fixed.png`)と完全位置一致するマスク投影を実現
- **手法**: DiffDRRを使用してマスク(`/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz`)をPA投影し、DeepDRR結果とオーバーレイ表示
- **現在の状況**: DiffDRRでの高解像度マスク投影が非常に重く、実用的でない

### 具体的な実装と問題点

#### 1. 現在の実装コード概要

**A. pure_mask_projector.py の重要部分:**
```python
# DiffDRRセットアップ（2500x3070高解像度）
width = int(round(plate_w / pixel_size)) & ~1   # 2500
height = int(round(plate_h / pixel_size)) & ~1  # 3070

self.mask_drr = DRR(
    self.mask_subject,
    sdd=1800.0,               # Source-to-Detector Distance
    height=height,            # 3070
    width=width,              # 2500
    delx=pixel_size,          # 0.194mm
    dely=pixel_size,          # 0.194mm
    x0=0.0, y0=0.0, z0=0.0,   # 検出器中心
    device=torch.device("cpu"),
    center_volume=True,
    pose_transforms=[], 
    render_depth=True,
    backend="siddon"          # Siddon線積分（CPU処理）
)
```

**B. 角度変換ロジック:**
```python
def deepdrr_to_diffdrr_pose(self, alpha_deg, beta_deg, gamma_deg):
    """DeepDRR角度をDiffDRR角度に変換"""
    # DeepDRR: α(ヨー), β(ピッチ), γ(ロール) → DiffDRR角度
    alpha_rad = np.radians(alpha_deg)  # ヨー
    beta_rad = np.radians(beta_deg)    # ピッチ
    gamma_rad = np.radians(gamma_deg)  # ロール
    
    # PA投影: α=0°, β=0°, γ=90° → DiffDRR変換
    angles_diffdrr = torch.tensor([
        [alpha_rad, beta_rad + np.pi/2, gamma_rad - np.pi/2]
    ], dtype=torch.float32)
    
    return angles_diffdrr
```

**C. 投影実行:**
```python
# PA正面投影実行
angles_diffdrr = self.deepdrr_to_diffdrr_pose(alpha_deg=0, beta_deg=0, gamma_deg=90.0)
mask_projection = self.mask_drr(angles_diffdrr)  # ← ここで数分間停止
```

#### 2. 試行した解決策

**A. adaptive_resolution_projector.py:**
- 段階的解像度アップ（0.1倍から1.0倍まで）
- 結果: 0.1倍（250x307）でも処理が重い

**B. 複数の最適化試行:**
```python
# 試行1: 解像度削減
resolution_scale = 0.1  # 250x307
# 試行2: デバイス最適化
device=torch.device("cpu")  # GPU使用不可
# 試行3: メモリクリア
gc.collect(), torch.cuda.empty_cache()
```

### 詳細な技術分析

#### DiffDRRの処理フロー問題点

1. **Siddon線積分アルゴリズム**: CPU処理で非常に重い
2. **高解像度処理**: 2500x3070 = 768万ピクセルでの線積分計算
3. **マスクデータ特性**: バイナリ（0/1）でも線積分計算は必要
4. **メモリ使用量**: 大きなボリュームデータの処理

#### DeepDRR vs DiffDRRの違い

**DeepDRR (成功例):**
- GPU最適化されたレンダラー
- CTデータの効率的な処理
- test_deepdrr_final_pa_fixed.pyで正常動作

**DiffDRR (問題):**
- CPU処理のSiddon線積分
- 高解像度での処理が非常に重い
- マスクデータでも処理負荷大

### 実行環境詳細

- **OS**: WSL2 Linux (Linux 6.6.87.2-microsoft-standard-WSL2)
- **Python**: 3.12
- **GPU**: 利用可能だがDiffDRRでGPU使用時エラー
- **メモリ**: 十分な容量
- **ディスク**: 十分な容量

### 求める解決策

以下の観点から包括的な解決策を求めます：

#### 1. パフォーマンス最適化手法
- DiffDRRでの高速化手法（GPU利用、アルゴリズム最適化）
- マスクデータ特有の最適化手法
- 効率的な解像度削減手法

#### 2. 代替技術手法
- DiffDRR以外の軽量なマスク投影ライブラリ
- DeepDRRでマスク投影する手法
- 他のX線投影ライブラリ（PyTorch3D、等）

#### 3. 実装アプローチ改善
- 投影解像度の適切な削減方法
- バッチ処理や分割処理での高速化
- メモリ効率的な実装パターン

#### 4. 位置一致保証手法
- DeepDRRとDiffDRRの完全な幾何学一致方法
- 座標系変換の正確性検証
- ピクセル単位での位置一致確認手法

### 制約条件

1. **既存DeepDRR結果の利用**: 新たにDeepDRRを実行せず、既存結果を使用
2. **完全位置一致**: マスクとCTの投影が完全に一致する必要
3. **実用的な処理時間**: 数分以内での処理完了が必要
4. **環境制約**: WSL2環境での動作が必要

### 期待する回答内容

1. **根本原因の特定**: なぜDiffDRRが重いのか、技術的詳細
2. **具体的な解決策**: 実装可能な高速化手法
3. **代替手法の提案**: DiffDRR以外の選択肢
4. **実装コード例**: 修正版コードの具体例
5. **ベストプラクティス**: マスク投影の効率的な実装方法

これらについて、公式実装を参考にした解決策を教えてください。