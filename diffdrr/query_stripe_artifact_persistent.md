# DiffDRR：ToCanonical適用後も横縞アーティファクトが残存する問題

## 環境
- DiffDRR v0.5.1
- TorchIO 0.20.21
- PyTorch 2.8.0
- CUDA 12.0

## 実装内容（gemini1.mdの推奨事項に基づく）

### 1. ToCanonicalによる座標系標準化
```python
# CTをRAS+座標系に標準化
subject = tio.Subject(ct=tio.ScalarImage(ct_path))
to_canonical = tio.ToCanonical()
subject_canonical = to_canonical(subject)

# 確認
print(f"Original orientation: {subject.ct.orientation}")  # ('L', 'P', 'S')
print(f"Canonical orientation: {subject_canonical.ct.orientation}")  # ('R', 'A', 'S')
```

### 2. 標準化後のボリュームをDiffDRRで読み込み
```python
# 標準化したHUボリュームを一時ファイルに保存
nifti_image = nib.Nifti1Image(canonical_hu_data, canonical_affine)
nib.save(nifti_image, temp_file.name)

# DiffDRRのread()で読み込み
subject = read(temp_file.name, bone_attenuation_multiplier=2.5)
```

## 問題

ToCanonicalでRAS+座標系に標準化したにも関わらず、生成されるDRR画像に横縞（水平ストライプ）アーティファクトが残存しています。

### 観察される症状
1. **規則的な横縞パターン**: 画像全体に水平な縞模様
2. **ブロック状の構造**: 特に斜め視点で顕著
3. **解剖学的構造の不明瞭さ**: 肺野、心臓、肋骨が識別困難

### 試した対処法と結果
1. ✅ ToCanonicalでRAS+座標系に標準化 → アーティファクト残存
2. ✅ 明示的なHU-LAC変換 → DiffDRRが受け付けない
3. ✅ カメラポーズの調整 → 部分的改善のみ
4. ✅ 後処理（ヒストグラム均等化、ガンマ補正）→ アーティファクトは消えない

## デバッグ情報

```
Volume shape after ToCanonical: (1, 1024, 1024, 251)
Canonical orientation: ('R', 'A', 'S')
DiffDRR volume range: [-1024, 3071] (HU values)
DiffDRR density range: [0.0, 1.0]
Generated DRR range: [1.29, 1.53] (very narrow)
```

## 推測される原因

1. **DiffDRRの内部処理の問題**
   - read()関数の内部での軸の再配置
   - Siddonアルゴリズムのメモリアクセスパターン

2. **ボリュームの形状やサイズの問題**
   - 1024x1024x251という大きなボリューム
   - メモリレイアウトの最適化不足

3. **レイキャスティングの精度問題**
   - サンプリング間隔が不適切
   - 補間方法の問題

## 質問

1. **DiffDRR v0.5.1でToCanonical適用後も横縞が残る原因は？**
   - 座標系以外の要因がある？
   - DiffDRRの特定のバージョンの問題？

2. **正しい前処理パイプラインは？**
   - ToCanonical以外に必要な前処理は？
   - ボリュームのリサイズやパディングが必要？

3. **代替アプローチ**
   - DiffDRR以外のDRR生成ライブラリの推奨
   - より低レベルでのレイキャスティング実装

これらについて、公式実装を参考にした解決策を教えてください。