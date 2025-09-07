# DiffDRRマスク投影が完全にゼロになる根本原因解析Query

## 問題の詳細状況

### 実行結果の詳細
```
INFO:__main__:✅ 投影完了: 形状torch.Size([3070, 2500]), 範囲[0.000, 0.000]
INFO:__main__:📊 マスク厚み統計: max=0.000mm, mean=0.000mm
INFO:__main__:📈 最大厚み: 0.00mm
```

**重大な問題**: DiffDRRでマスク投影を実行しても**全ピクセルが完全にゼロ**になる

### 実装詳細と設定値

#### 1. マスクデータの詳細分析
```python
# マスクファイル分析結果
形状: (1024, 1024, 251)
データ型: float64
値の範囲: 0.0-1.0
ユニーク値: [0. 1.]
非ゼロ数: 796,876/263,192,576 (0.30%) # 十分なマスクデータが存在
spacing: (0.34179688, 0.34179688, 1.5)
orientation: ('L', 'P', 'S')
affine:
[[ -0.34179688   0.           0.         166.        ]
 [  0.          -0.34179688   0.          -9.        ]
 [  0.           0.           1.5         56.29986572]
 [  0.           0.           0.           1.        ]]
```

#### 2. DiffDRR設定の詳細
```python
# 現在の実装 (patch_mask_thickness_drr.py)
subject = read(
    volume="/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz",
    orientation="PA",
    center_volume=False       # Query回答準拠: 原点ズレ防止
)

drr = DRR(
    subject,
    sdd=1800.0,               # DeepDRRと同じSID
    height=3070,              # DeepDRRと同じ高さ
    width=2500,               # DeepDRRと同じ幅
    delx=0.194,               # DeepDRRと同じピクセルサイズ
    dely=0.194,
    renderer="siddon",        # 正確な線積分
    reducefn="sum",           # Beer–Lambert無し
    reverse_x_axis=True,      # radiographic view
    patch_size=10,            # メモリ節約
).to(device=cuda, dtype=float32)
```

#### 3. 角度変換の実装
```python
# Query回答準拠の公式角度変換
# DeepDRR PA: α=0°, β=0°, γ=90°
alpha = np.deg2rad(0.0)           # 0°
beta  = np.deg2rad(90.0 - 0.0)    # 90°
gamma = np.deg2rad(-90.0)         # -90°

angles = torch.tensor([[alpha, beta, gamma]], device=cuda, dtype=float32)
translations = torch.zeros((1, 3), device=cuda, dtype=float32)

# 投影実行
img = drr(angles, translations, parameterization="euler_angles", convention="YZX")
```

### 比較: DeepDRR vs DiffDRRの設定差異

#### DeepDRR (成功例) の重要設定
```python
# deepdrr/test_deepdrr_final_pa_fixed.py での成功設定
volume = Volume.from_nifti(ct_path)  # CTデータ使用
device = geo.CArm(
    alpha=0, beta=0, gamma=90,       # PA設定
    pixel_size=0.194,
    source_to_detector_distance=1800,
    height=3070, width=2500
)
# 最終的に水平反転: img = img[:, ::-1]
```

#### DiffDRR (問題発生) の設定
```python
# マスクデータ使用 (0/1バイナリ)
# center_volume=False (Query回答準拠)
# 公式角度変換使用
# 結果: 完全にゼロ
```

### 潜在的な根本原因候補

#### 1. 座標系・原点の不整合
- **center_volume=False**: NIfTI実世界座標を維持するが、DeepDRRとの原点が一致しない可能性
- **affine変換**: マスクのaffine行列による座標変換が不適切
- **orientation**: ('L', 'P', 'S') vs DeepDRRの座標系の違い

#### 2. 角度変換の問題
- **公式変換の適用ミス**: α_diff=α, β_diff=90°-β, γ_diff=-γ が実際には不適切
- **convention="YZX"**: 回転順序がDeepDRRと異なる
- **parameterization="euler_angles"**: DeepDRRと異なる角度表現

#### 3. 幾何学的配置の問題
- **検出器位置**: DiffDRRの検出器がマスクの投影範囲外に配置
- **source位置**: X線源の位置設定が不適切
- **視野**: マスクが検出器の視野に入っていない

#### 4. データ処理の問題
- **マスクの値域**: 0/1バイナリが線積分で適切に処理されていない
- **precision**: float32 vs float64の精度問題
- **device**: GPU処理での数値計算の問題

### 検証が必要な具体的項目

#### A. 座標系整合性の検証
1. DeepDRRとDiffDRRでの同一マスクの原点・中心位置比較
2. affine行列による座標変換の影響確認
3. center_volume=True vs False での結果差異

#### B. 角度設定の検証
1. 複数角度（AP、LAT等）での投影テスト
2. 角度=0（正面）での単純投影テスト
3. DeepDRRの実際の角度パラメータとの詳細比較

#### C. 幾何学的配置の検証
1. 検出器とマスクの相対位置関係の可視化
2. source-detector-object の幾何学配置確認
3. 投影レイがマスク領域を通過するかの確認

#### D. データ・処理の検証
1. 中間解像度（500x500等）での動作確認
2. trilinearレンダラーでの結果比較
3. マスクデータの前処理（補間、リサンプリング）の影響

### 期待する解決策

#### 1. 根本原因の特定
- 座標系、角度、幾何学配置のうち、どれが主原因か
- Query回答の実装で見落とした重要な設定は何か

#### 2. 段階的デバッグ手法
- 問題を段階的に切り分ける具体的手順
- 各段階で確認すべき数値・可視化方法

#### 3. 修正版実装コード
- 根本原因に対応した具体的なコード修正
- DeepDRRとの完全整合を保証する実装方法

#### 4. 代替アプローチ
- DiffDRR以外での同等機能実現方法
- より確実な位置一致保証手法

### 制約条件

- 既存のDeepDRR PA画像との完全位置一致が必須
- マスクデータ（0/1バイナリ）での厚み投影が必要
- 実用的な処理時間（数分以内）での動作が必要
- WSL2 + CUDA環境での動作が必要

**なぜQuery回答準拠の実装で全てゼロになるのか、その根本原因と解決策を具体的な実装コードと共に教えてください。**