# HARTA実装 問題分析レポート

## 現状の実装内容

### 実装済みスクリプト
1. **nifti_to_dicom.sh**: NIfTI → DICOM変換
2. **run_harta_auto.py**: HARTA自動実行ラッパー
3. **harta_masks_to_nifti.py**: 2Dマスク → 3D NIfTI変換
4. **batch_process_harta.sh**: バッチ処理統合スクリプト

### 処理フロー
1. NIfTIファイル → DICOM変換（Docker内でPython実装）
2. HARTA実行（_automatic_.py使用）
3. 2D PNGマスク → 3D NIfTI変換
4. ボリューム計算

## 発見された問題

### 主要問題：セグメンテーション結果の座標系不整合
- **症状**: セグメンテーション画像がオリジナルCTと同じような見た目だが、位置・向き・粗さが異なる
- **原因箇所**: 複数の変換ステップで座標系情報が失われている

### 具体的な問題点

#### 1. NIfTI → DICOM変換の問題
```python
# 現在の実装（nifti_to_dicom.sh内）
ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # 固定値（軸方向）
ds.ImagePositionPatient = [float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3] + i * spacing[2])]
```
- **問題**: ImageOrientationPatientが固定値で、元のNIfTIの回転情報が失われている
- **影響**: DICOMの向きが常に軸方向になり、斜め切りのCTデータで位置ずれが発生

#### 2. HARTA出力形式の問題
```
HARTA出力:
- fat/CTRATE001_0_fat.png: RGBA形式 (512, 512, 4)
- 各スライスが独立したPNGファイル
- スライス番号のみで位置情報なし
```
- **問題**: HARTAの出力PNGにはDICOMの座標系情報が含まれない
- **影響**: 3D再構成時に正確な位置合わせができない

#### 3. 2D → 3D NIfTI変換の問題
```python
# 現在の実装（harta_masks_to_nifti.py）
img = sitk.GetImageFromArray(mask_vol.astype(np.uint8).transpose(2, 0, 1))
img.SetSpacing((spacing[0], spacing[1], spacing[2]))
img.SetOrigin(origin)
# SetDirectionが未設定
```
- **問題**: 方向コサイン（Direction matrix）が設定されていない
- **影響**: デフォルトの単位行列が使用され、回転情報が失われる

#### 4. PNGマスクの読み込み問題
```python
# HARTAのfatマスクがRGBA形式で保存されている
shape=(512, 512, 4), dtype=uint8  # 4チャンネル
```
- **問題**: RGBAの4チャンネルから適切にバイナリマスクを抽出していない可能性
- **影響**: マスクの値が不正確になる

## 問題の根本原因

### 座標系変換チェーンの崩壊
```
オリジナルNIfTI（正しい座標系）
    ↓ [変換1: アフィン行列の一部のみ使用]
DICOM（簡略化された座標系）
    ↓ [HARTA処理]
2D PNGマスク（座標情報なし）
    ↓ [変換2: 不完全な座標系復元]
出力NIfTI（座標系不整合）
```

## 必要な修正

1. **NIfTI → DICOM変換の改善**
   - オリジナルのアフィン変換行列から正確なImageOrientationPatientを計算
   - スライス位置の正確な計算

2. **座標系情報の保持**
   - DICOM変換時の座標系情報を別ファイルに保存
   - 3D再構成時にその情報を使用

3. **3D NIfTI生成の改善**
   - オリジナルNIfTIのアフィン変換行列を直接コピー
   - Direction matrixの適切な設定

4. **マスク処理の改善**
   - RGBA → バイナリマスクの適切な変換
   - しきい値処理の確認