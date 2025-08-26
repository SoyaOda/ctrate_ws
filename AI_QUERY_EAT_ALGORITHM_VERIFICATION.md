# TotalSegmentatorとHARTAを組み合わせたEAT抽出アルゴリズムの検証

## 背景と問題
HARTAの心膜境界検出が失敗し、体全体の脂肪を検出してしまう問題（5,385ml vs 期待値100-300ml）を解決するため、TotalSegmentatorの心臓マスクを使用してHARTA出力をクリップする実装を行った。

## 現在の実装アルゴリズム

### 1. 全体フロー
```
1. HARTA PNG (fat/)から3Dボリューム再構成
2. TotalSegmentatorで心臓構造取得
3. EDT（ユークリッド距離変換）で心膜周囲の薄い殻（0-6mm）生成
4. 最終EAT = HARTA脂肪 ∧ HU値(-190～-30) ∧ 殻 ∧ ¬心筋
```

### 2. 詳細実装コード

#### 2.1 HARTA PNGからバイナリマスク抽出
```python
def load_mask_from_png(png_path, rgb_threshold=127):
    """
    HARTAのPNG（RGBA形式）からバイナリマスクを抽出
    アルファチャンネルは常に255なので使用せず、RGBチャンネルで判定
    """
    im = Image.open(png_path).convert("RGBA")
    arr = np.array(im)  # (H, W, 4)
    rgb = arr[..., :3].astype(np.int16)
    # R=G=B=255が脂肪、0が背景。閾値を超えたら1
    mask = (rgb > rgb_threshold).any(axis=-1).astype(np.uint8)
    return mask
```

#### 2.2 3Dボリューム再構成
```python
def reconstruct_3d_from_harta(harta_png_dir, slice_map_json):
    """HARTA PNGから3Dボリュームを再構成"""
    with open(slice_map_json, 'r') as f:
        smap = json.load(f)
    
    geometry = smap["geometry"]
    size = tuple(geometry["size"])  # (X, Y, Z)
    meta = smap["meta"]
    
    # fat/ディレクトリのPNGファイルを収集
    hdir = Path(harta_png_dir) / "fat"
    masks_dict = {}
    
    for p in hdir.glob("*.png"):
        fname = p.stem  # CTRATE001_100_fat -> 100を抽出
        parts = fname.split('_')
        if len(parts) >= 2:
            try:
                slice_idx = int(parts[1])
                if slice_idx < len(meta):
                    masks_dict[slice_idx] = load_mask_from_png(str(p))
            except ValueError:
                continue
    
    # Z順にスタック
    stacks = []
    for z in range(size[2]):
        if z in masks_dict:
            stacks.append(masks_dict[z])
        else:
            stacks.append(np.zeros((size[1], size[0]), dtype=np.uint8))
    
    volume = np.stack(stacks, axis=2).astype(np.uint8)  # (X, Y, Z)
    return volume, geometry
```

#### 2.3 心臓ROIと薄い殻の生成（EDT使用）
```python
def create_heart_roi_with_shell(totalseg_dir, ct_nifti_path, shell_thickness_mm=6.0):
    """
    TotalSegmentatorの心臓マスクからEDTを使って
    心膜周囲の薄い殻を作成
    """
    # CT画像からスペーシング情報取得
    ct_img = nib.load(ct_nifti_path)
    spacing = ct_img.header.get_zooms()[:3]  # (sx, sy, sz) in mm
    
    # 心臓構造を読み込み
    heart_parts = []
    components = [
        "heart_myocardium",      # 心筋
        "heart_ventricle_left",  # 左心室
        "heart_ventricle_right", # 右心室
        "heart_atrium_left",     # 左心房
        "heart_atrium_right"     # 右心房
    ]
    
    for comp in components:
        comp_path = Path(totalseg_dir) / f"{comp}.nii.gz"
        if comp_path.exists():
            mask = nib.load(str(comp_path)).get_fdata() > 0
            heart_parts.append(mask)
    
    # すべての心臓構造を結合
    heart = np.logical_or.reduce(heart_parts)
    
    # 心筋マスクを取得（ない場合は心臓全体を使用）
    myo_path = Path(totalseg_dir) / "heart_myocardium.nii.gz"
    if myo_path.exists():
        myo = nib.load(str(myo_path)).get_fdata() > 0
    else:
        myo = heart
    
    # ユークリッド距離変換で心筋表面からの距離を計算
    outside = ~myo
    dist_mm = distance_transform_edt(outside, sampling=spacing)
    
    # EAT領域：心筋表面から0-6mmの薄い殻
    shell_eat = (dist_mm > 0) & (dist_mm <= shell_thickness_mm)
    
    # 安全のため、膨張させた心臓領域内に制限
    from scipy.ndimage import binary_dilation, generate_binary_structure
    struct = generate_binary_structure(3, 2)
    shell_eat &= binary_dilation(heart, structure=struct, iterations=2)
    
    return shell_eat, myo
```

#### 2.4 最終EAT抽出
```python
def extract_eat(ct_nifti_path, harta_volume, shell_eat, myo, 
                hu_min=-190, hu_max=-30):
    """
    最終的なEATを抽出
    EAT = HARTA脂肪 ∧ HU値範囲 ∧ 殻 ∧ ¬心筋
    """
    # CT画像を読み込み
    ct_img = nib.load(ct_nifti_path)
    ct_data = ct_img.get_fdata()
    
    # HU値による脂肪領域の制限（文献標準：-190～-30 HU）
    hu_fat = (ct_data >= hu_min) & (ct_data <= hu_max)
    
    # HARTA脂肪マスク
    harta_fat = harta_volume > 0
    
    # 最終EAT = すべての条件のAND演算
    # 1. HARTA脂肪候補
    # 2. HU値が脂肪範囲
    # 3. 心膜周囲の殻内
    # 4. 心筋ではない
    eat = harta_fat & hu_fat & shell_eat & (~myo)
    
    # ボリューム計算
    spacing = ct_img.header.get_zooms()[:3]
    voxel_volume_ml = np.prod(spacing) / 1000.0  # mm³ to ml
    eat_volume_ml = np.sum(eat) * voxel_volume_ml
    
    return eat, eat_volume_ml
```

## 検証すべき点

### 1. アルゴリズムの妥当性
- **EDT（ユークリッド距離変換）の使用は適切か？**
  - 心筋表面からの正確な距離計算に使用
  - アニソトロピックボクセル（異方性）に対応（spacing考慮）
  
- **6mmの殻厚は妥当か？**
  - 文献的にEATは心外膜から数mm程度
  - 5mm, 6mm, 8mm, 10mmで感度解析を実施予定

### 2. HU値範囲の妥当性
- **-190～-30 HUは適切か？**
  - 文献（BMC Cardiovasc Disord 2022等）で標準的な範囲
  - 脂肪組織の典型的なHU値範囲

### 3. 論理演算の順序と組み合わせ
```python
eat = harta_fat & hu_fat & shell_eat & (~myo)
```
- この4条件のAND演算は適切か？
- 心筋を除外（~myo）は必要か？

### 4. TotalSegmentatorの使用上の問題

#### 現在直面している問題：
```python
# エラー1: heartchambers_highresタスクはライセンス必要
ValueError: task heartchambers_highres requires license

# エラー2: roi_subsetで心臓構造を指定するとKeyError
KeyError: 'heart_myocardium'
```

#### 質問：
1. TotalSegmentatorの標準タスク（task=total）で心臓構造は取得可能か？
2. 心臓構造のクラス名は正しいか？
   - heart_myocardium
   - heart_atrium_left/right
   - heart_ventricle_left/right

### 5. 座標系の整合性
- HARTAの2D PNG（512×512）
- CT NIfTI（512×512×217）
- TotalSegmentatorの出力
これらの座標系は正しく一致しているか？

### 6. 心膜境界の定義
- 現在の実装：心筋（myocardium）の外側0-6mmを心膜領域と仮定
- 医学的に正確か？
- visceral pericardiumとparietal pericardiumの区別は必要か？

## 期待される結果
- **EATボリューム**: 50-200ml（健常者の典型値）
- **現在の問題**: HARTAのみだと5,385ml（明らかに過大）

## 質問

1. **アルゴリズム全体の妥当性**
   - このアプローチ（HARTA＋TotalSegmentator＋EDT）は医学的に正しいか？
   - より良い方法はあるか？

2. **TotalSegmentatorの正しい使用方法**
   - 心臓構造を取得する最適な方法は？
   - ライセンス不要で心臓セグメンテーションを得る方法は？

3. **パラメータの最適化**
   - 殻厚（6mm）は適切か？
   - HU値範囲（-190～-30）は適切か？

4. **実装の改善点**
   - 計算効率の改善方法は？
   - メモリ使用量の削減方法は？

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。