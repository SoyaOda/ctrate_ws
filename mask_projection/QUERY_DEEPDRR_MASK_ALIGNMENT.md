# DeepDRRとマスク投影の完全位置一致に関するQuery

## 背景と問題

### 現在の状況
`test_deepdrr_final_pa_fixed.py`は正常に動作しており、PA投影（α=0°, β=0°, γ=90°）で適切なDRR画像を生成しています。しかし、DiffDRRを使用したマスク投影との位置一致を実現しようとした際に以下の問題が発生しています。

### 問題詳細

#### 1. CUDAコンパイルエラー
DeepDRRのProjectorを直接使用する際に以下のエラーが発生：
```
nvcc compilation failed - texture identifier undefined errors
```

#### 2. 位置一致の不確実性
現在のDiffDRR実装では以下を実行していますが、完全位置一致が保証されているか不明：

**DeepDRR側（test_deepdrr_final_pa_fixed.py）**:
- LPS厳密変換: `nio.apply_orientation()` + `nio.inv_ornt_aff()`
- 体積中心設定: `origin = (-0.5 * (shp - 1.0) * spacing).tolist()`
- isocenter設定: `carm.isocenter = ct.center_in_world`
- PA左右反転: `img_pa = img_pa[:, ::-1].copy()`
- C-arm設定: `rotate_camera_left=True`, SID=1800mm, ピクセル=0.14mm

**DiffDRR側（現在の実装）**:
- Subject読み込み: `diffdrr.data.read(volume, center_volume=True, orientation="PA")`
- 角度変換: α′=α, β′=90°−β, γ′=−γ (gpt5pro3.md準拠)
- 平行移動: `[0, 0, 0]` (center_volume=Trueのため)
- PA左右反転: `thickness_mm[:, ::-1].copy()`

#### 3. 検証方法の不足
両者の位置一致を検証する明確な方法が確立されていません。

## 質問事項

### 1. 座標系統一の確実性について
`test_deepdrr_final_pa_fixed.py`のLPS変換と`diffdrr.data.read(center_volume=True)`によるLPS処理は、**完全に同一の座標変換**を行っていると保証できますか？

特に以下の点について：
- DeepDRRの`origin = (-0.5 * (shp - 1.0) * spacing).tolist()`
- DiffDRRの`center_volume=True`による内部処理
- 両者のisocenter/world原点の一致

### 2. 角度変換の正確性確認
gpt5pro3.mdの角度変換式：
```
α′ = α
β′ = 90° - β  
γ′ = -γ
```

この変換により、DeepDRRの`move_to(alpha=0, beta=0, gamma=90)`とDiffDRRの変換後角度が、**完全に同一の投影ジオメトリ**を表現していることを確認できる方法はありますか？

### 3. PA左右反転の一致性について
両者でPA投影に`[:, ::-1]`を適用していますが：
- DeepDRR: `rotate_camera_left=True` + 最終反転
- DiffDRR: `reverse_x_axis=True` + 最終反転

この組み合わせで**ピクセル単位での完全一致**が実現されますか？

### 4. 検出器パラメータの精密一致について
現在の設定：
```python
# DeepDRR (test_deepdrr_final_pa_fixed.py)
sid=1800.0, oid=80.0, px=0.14
W = int(round(350.0 / 0.14)) & ~1  # 2500
H = int(round(430.0 / 0.14)) & ~1  # 3070

# DiffDRR
sdd=1800.0, delx=0.14, dely=0.14
width=2500, height=3070
```

これらが**数値誤差なし**でピクセル座標まで一致することを保証できますか？

### 5. 代替検証方法について
CUDAコンパイル問題によりDeepDRRを直接実行できない場合、以下の方法でマスク投影の位置精度を検証できますか？

**提案A: 既存結果との比較**
1. `test_deepdrr_final_pa_fixed.py`を実行してPA DRR画像を生成
2. DiffDRRでマスク投影を生成  
3. 両画像の特徴点（例：心臓輪郭）を比較して位置一致を検証

**提案B: ファントム検証**
1. 既知形状のマスク（例：立方体、球体）で投影テスト
2. 解析的に計算可能な厚み分布と比較
3. 位置ずれを定量的に測定

**提案C: Docker環境の活用**
`deepdrr-cuda11:latest`などの既存Docker環境でDeepDRRを実行し、ホスト側のDiffDRRと結果比較

## 期待する回答

### 1. 確実な位置一致実現方法
上記の疑問点を解決し、**test_deepdrr_final_pa_fixed.pyのDRRと完全位置一致するマスク投影**を実現するための具体的実装方法

### 2. 検証手順
両者の位置一致を客観的に確認できる検証方法とその実装

### 3. 可視化機能
DRR・マスク・オーバーレイを含む統合可視化の実装方法

## 現在の実装状況
- DiffDRRによるマスク投影: ✅ 動作
- 角度変換: ✅ 実装済み
- PA左右反転: ✅ 実装済み  
- DeepDRR直接実行: ❌ CUDAエラー
- 位置一致検証: ❌ 未実装

これらについて、公式実装を参考にした解決策を教えてください。