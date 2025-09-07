# 元解像度(2500×3070)でのDiffDRRメモリ最適化Query

## 現状と要求

### 達成された成果
- ✅ **根本原因解決**: `center_volume=True + by=-SOD` でゼロ投影問題を完全解決
- ✅ **低解像度動作確認**: 256×256で69.214mm最大厚み、3.7%非ゼロピクセルを達成
- ✅ **DeepDRR準拠設定**: SID=1800, SOD=1720, 角度変換、radiographic viewを完全実装

### 残る課題
- ❌ **元解像度(2500×3070)でのメモリ不足**: 42.89GB要求 vs 24GB GPU制限
- ❌ **実用的処理時間**: 2分でタイムアウト（パッチ分割でも）

### ユーザー要求
**"できる限り元の解像度で行いたい"** - 2500×3070での実用的な動作を実現

## 環境詳細

### ハードウェア仕様
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **利用可能メモリ**: 23.7GB
- **プラットフォーム**: WSL2 Linux

### 現在のメモリ使用量推定
```
Target: 2500×3070 = 7.7M pixels
Memory estimates (float32):
- Volume data: ~1GB (1024×1024×251)
- Output image: 0.03GB  
- Intermediate: 0.06GB (2x buffer)
- GPU overhead: ~2-4GB
Total estimate: 4.1GB ← 十分に24GB以内のはず
```

**矛盾**: 理論上4.1GBなのに実際は42.89GB要求される

## 現在の実装詳細

### 成功している設定（256×256）
```python
# quick_test_drr.py - 成功例
subject = read(
    volume=mask_path,
    orientation="PA", 
    center_volume=True,  # ✅
)

drr = DRR(
    subject, sdd=1800.0, height=256, width=256,
    delx=2.0, dely=2.0, renderer="siddon",
    reducefn="sum", reverse_x_axis=True,
).to(device="cuda", dtype=torch.float32)

# 角度・平行移動
rot = torch.tensor([[0.0, np.deg2rad(90.0), np.deg2rad(-90.0)]])
xyz = torch.tensor([[0.0, -1720.0, 0.0]])  # ✅ 修正済み
pose = convert(rot, xyz, parameterization="euler_angles", convention="YZX")

# 結果: 69.214mm最大厚み、3.7%非ゼロ ← 完全成功
```

### 失敗している設定（2500×3070）
```python
# fixed_mask_thickness_drr.py - メモリ不足
SID, SOD = 1800.0, 1720.0
PX = 0.14  # DeepDRR準拠
W, H = 2500, 3070  # 元解像度

drr = DRR(
    subject, sdd=SID, height=H, width=W,
    delx=PX, dely=PX, renderer="siddon", 
    reducefn="sum", reverse_x_axis=True,
    patch_size=5,  # 最小パッチサイズでも不足
)

# エラー: CUDA out of memory. Tried to allocate 42.89 GiB
```

## 試行した最適化手法とその結果

### 1. パッチ分割
```python
patch_candidates = [64, 50, 25, 20, 10, 5]
patch_size = 5  # 2500×3070 を 500×614 の2500パッチに分割

# 結果: 42.89GB要求で失敗
```

### 2. データ型最適化
```python
dtype = torch.float32  # vs float16
# float16テストは未実施
```

### 3. Trilinear vs Siddon
```python
renderer="trilinear"  # 高速だが近似
# vs
renderer="siddon"     # 正確だが重い

# Trilinearでも同じメモリ不足エラー
```

### 4. 解像度段階削減
```python
# 1/2解像度 (1250×1535) でも2分タイムアウト
# 1/8解像度 (313×384) は未テスト
```

## 根本的な疑問と仮説

### 疑問1: メモリ使用量の不一致
**理論**: 4.1GB vs **実際**: 42.89GB (10倍差)

**可能な原因**:
1. **Siddon線積分の内部実装**: 大量の中間バッファが必要？
2. **パッチ分割の実装問題**: 全体を一度にロードしている？
3. **DiffDRRの座標変換**: 高解像度で指数的にメモリ増加？
4. **PyTorchの自動微分**: 不要なgradient計算？

### 疑問2: パッチ分割の効果不足
**期待**: 500×614パッチなら1/25のメモリ使用量
**実際**: 全体メモリ使用量は変わらない

### 疑問3: 256×256成功 vs 2500×3070失敗
**ピクセル数比**: 256×256 = 65K vs 2500×3070 = 7.7M (118倍)
**メモリ使用量**: 理論的には118倍だが、実際は桁違いの差

## 求める解決策

### 1. メモリ使用量の詳細分析
- DiffDRR内部でのメモリ使用量の段階的解析
- Siddonレンダラーの具体的なメモリ要件
- パッチ分割が実際に機能しているかの検証

### 2. 高解像度最適化手法
```python
# 求める最適化アプローチ
# A案: メモリ効率的なパッチ分割実装
# B案: より小さなパッチサイズ (例: patch_size=2)
# C案: float16 + mixed precision
# D案: CPU-GPU間でのデータ転送最適化
# E案: 段階的サイズアップ (512→1024→2048→2500)
```

### 3. 代替レンダリング手法
- **Trilinear最適化**: サンプル数を動的調整
- **カスタムSiddon**: メモリ効率版の実装
- **ハイブリッド**: 低解像度Trilinear + 高解像度Siddon upsampling

### 4. 具体的な実装コード
- 24GB制限内で2500×3070を実現する完全なコード
- メモリ使用量のリアルタイム監視
- 段階的デバッグのためのメモリプロファイリング

### 5. パフォーマンス最適化
- 2分以内での処理完了
- リアルタイム進捗表示
- 中断・再開機能

## 制約条件

1. **メモリ制限**: 24GB GPU VRAM厳守
2. **品質維持**: Siddon線積分による正確な厚みmm値
3. **位置一致**: DeepDRRとの完全な幾何学的整合性
4. **設定維持**: 修正済み設定 (center_volume=True, by=-SOD) の継続使用

## 期待する回答内容

### 1. 根本原因の特定
- なぜ4.1GB推定が42.89GB要求になるのか
- DiffDRRの内部メモリ使用パターンの詳細

### 2. 実践的解決策
- 24GB以内で2500×3070を確実に処理する具体的手法
- メモリ効率最適化の段階的アプローチ

### 3. 最適化実装コード
```python
# 期待するコード形式
def memory_efficient_drr_2500x3070():
    # 24GB制限内での確実な実装
    # メモリ監視付き
    # エラーハンドリング付き
    pass
```

### 4. フォールバック戦略
- 2500×3070が不可能な場合の最大解像度
- 品質とパフォーマンスのトレードオフ指針

**現在の修正済み設定で、できる限り元解像度(2500×3070)に近い高品質マスク厚みDRRを24GB GPU制限内で実現する方法を、具体的な実装コードと共に教えてください。**