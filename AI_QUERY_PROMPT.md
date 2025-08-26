# AIクエリプロンプト：HARTA EATセグメンテーション座標系問題の解決

## 技術的背景
医療画像処理パイプラインでHARTA（Epicardial Adipose Tissue segmentation tool）を使用してCT画像からEAT（心外膜脂肪組織）を抽出しています。HARTAはDICOM入力のみ対応の非深層学習ベースのツールです。

## 現在の処理フロー
1. NIfTI形式のCTデータをDICOMに変換（Pythonスクリプト）
2. HARTAでDICOMを処理し、2D PNGマスクを生成
3. 2D PNGマスクを3D NIfTIに再構成
4. セグメンテーション結果を可視化

## 発生している問題
セグメンテーション結果のNIfTIファイルが、オリジナルCTと同じような画像になっており、以下の問題があります：
- 位置がずれている
- 向きが異なる
- 解像度/粗さが変わっている
- 座標系が不整合

## 技術的詳細

### 問題のあるコード部分

#### 1. NIfTI→DICOM変換での座標系情報損失
```python
# 現在の問題のある実装
ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # 固定値
ds.ImagePositionPatient = [float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3] + i * spacing[2])]
```

#### 2. 3D再構成時の方向行列未設定
```python
# SimpleITK使用時
img = sitk.GetImageFromArray(mask_vol.transpose(2, 0, 1))
img.SetSpacing(spacing)
img.SetOrigin(origin)
# SetDirection()が未設定
```

#### 3. アフィン変換行列の不整合
- オリジナルNIfTI: 完全な4x4アフィン変換行列
- DICOM変換: 簡略化された座標系（軸平行と仮定）
- 出力NIfTI: デフォルトの単位行列

## 質問

### メイン質問
NIfTI → DICOM → 2D処理 → NIfTIの変換チェーンで、医療画像の座標系（アフィン変換、方向コサイン、原点）を正確に保持する方法を教えてください。

### 具体的な実装方法
1. **NIfTI→DICOM変換**: nibabelのアフィン行列からDICOMのImageOrientationPatientとImagePositionPatientを正確に計算する方法

2. **座標系情報の保存**: DICOM変換時の座標系メタデータを保存し、後で復元する最適な方法

3. **3D再構成**: 2D PNGマスクからオリジナルと同じ座標系を持つ3D NIfTIを生成する方法

4. **SimpleITKでの実装**: SimpleITKでDirection matrixを含む完全な座標系を設定する方法

### 参考にすべきライブラリ/ツール
- nibabel、SimpleITK、pydicom、ITK
- TotalSegmentatorやnnU-Netなどの医療画像処理での座標系処理
- DICOMとNIfTI間の座標系変換のベストプラクティス

### コード例の要望
以下の完全な実装例を提供してください：
1. NIfTIのアフィン行列からDICOMの正確な座標系パラメータを計算
2. 座標系メタデータの保存と復元
3. オリジナルと同じ座標系でマスクNIfTIを生成

## 期待する解決策
- オリジナルCTとピクセル単位で位置が一致するセグメンテーション結果
- 医療画像ビューアで正しく重ね合わせ表示できる出力
- DICOM/NIfTI座標系変換の完全性を保証する実装

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。