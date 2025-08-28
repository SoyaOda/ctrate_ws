すごく良い進化です。ログと各スライスを見る限り、**ILAM**（肺接触ベース）の導入で横隔膜近傍の偽陰性はかなり抑えられています。一方で、ユーザメモのとおり「肺に接する torso\_fat の**接触部分だけ**が黄色で拾われ、塊全体が拾われていない」現象が残っています。
原因と対処、その他の改善ポイントをまとめます。

---

## なぜ接触部分だけが EAT+PAT になるのか

v4 の ILAM 部分では

```python
cand_contact = (
    fat_clipped & mediastinum & lungs_dilated &
    near_heart & ~abdomen_forbidden
)
```

という\*\*“局所条件の積”**で候補を作っています。`lungs_dilated`（肺を δ mm 膨張）に**当たっている体脂肪のボクセルだけ**が通るため、肺に触れている“縁”の細い帯しか残りません。
本来やりたいのは「**肺と接触している塊（連結成分）を丸ごと採用**」なので、**“接触ボクセルをシードにして、torso\_fat の中を連結伝播\*\*」させる必要があります。

> これには **geodesic（マスク制約付き）な伝播**が適しており、`scipy.ndimage.binary_propagation` がそのための関数です（“morphological geodesic dilation of an image under a mask”）。一方、`binary_dilation` は構造要素と反復回数ベースの単純膨張で、距離の等方性は保証されません。([docs.scipy.org][1], [scipy.github.io][2])
> また **`distance_transform_edt` の `sampling`** を使うとボクセルの非等方性を補正した距離（mm）での判定が可能です。([scipy.github.io][3], [tedboy.github.io][4])

---

## 直し方（最小パッチ）

### 1) mm 正確化のための等方膨張ヘルパ

反復回数ではなく EDT を使った「半径 r\[mm] 膨張」に統一します。

```python
from scipy.ndimage import distance_transform_edt

def iso_dilate(mask: np.ndarray, radius_mm: float, spacing) -> np.ndarray:
    """等方(距離)膨張：mask から半径 radius_mm 以内を True"""
    if radius_mm <= 0:
        return mask.copy()
    dist = distance_transform_edt(~mask, sampling=spacing)
    return dist <= radius_mm
```

* `lungs_dilated = iso_dilate(lungs, lung_contact_mm, spacing)`
* `abdomen_forbidden = iso_dilate(abdominal_organs, abdomen_gap_mm, spacing)`
* 心膜ブリッジや大動脈PVAT除外も同様に EDT 版に。

> ※ `binary_dilation` の反復はボクセルサイズに依存し、Z 方向に厚い撮像では mm 換算がズレます。EDT なら `sampling=spacing` で mm ベースになります。([scipy.github.io][2])

### 2) 連結成分を**塊ごと**採る（推奨：ラベル法）

接触ボクセルを含む連結成分だけを残すシンプルで速い実装です。

```python
from scipy.ndimage import label, generate_binary_structure

# 下部帯域（Z）は z_spacing で計算
z_spacing = float(spacing[2])
band_slices = int(np.ceil(inferior_band_mm / z_spacing))
z_switch = min(z_min + band_slices, z_max)

# 連結性の定義（6:直交, 18, 26:対角も許可）
conn = generate_binary_structure(3, 2)  # 26 連結

# “ドメイン”：伝播してよい領域
domain = (fat_clipped & (~lungs) & ~abdomen_forbidden)
if max_heart_dist_mm is not None:
    dist_heart = distance_transform_edt(~heart_mask, sampling=spacing)
    domain &= (dist_heart <= max_heart_dist_mm)

# Z クリップ
zmask = np.zeros_like(domain, bool); zmask[:, :, z_min:z_switch+1] = True
domain &= zmask

# 遠位大動脈 PVAT の禁止帯があるなら domain からも除外しておく
# domain &= ~distal_aorta_forbidden

# “シード”：肺に δ mm 以内で接触している torso_fat
seed = domain & iso_dilate(lungs, lung_contact_mm, spacing)

# ラベル付けして、シードに触れるラベルのみ残す
lbl, n = label(domain, structure=conn)
touch_ids = np.unique(lbl[seed])
ilam_components = (lbl > 0) & np.isin(lbl, touch_ids)
```

これで **肺に触れる torso\_fat の“塊”**が下部帯域でまるごと取れます。
最後はこれを shell ベースと**和集合**に：

```python
pericardial_fat_mask = eat_pat_shell | ilam_components
```

（実験的に置換したい場合は、z\_min〜z\_switch だけ `eat_pat_shell` を上書き）

### 3) 代替：geodesic 伝播（`binary_propagation`）

同じ発想をモルフォロジーの**マスク付き伝播**で書くとこうなります：

```python
from scipy.ndimage import binary_propagation

mask_for_growth = domain  # 上で作った domain をそのままマスクに
reached = binary_propagation(seed=seed, mask=mask_for_growth, structure=conn)
ilam_components = reached   # = 肺接触から domain 内を連結伝播した領域
```

`binary_propagation` は「シードからマスクの中を連結に広がる」演算です。([docs.scipy.org][1])
ラベル法と同等の結果になりますが、`mask` でさらに細かい制御（例えば地理的制約）を掛けたい時に便利です。

---

## そのほかの改善ポイント（画像とログを踏まえたレビュー）

1. **下部帯域の厚さ計算**
   　`band_slices = int(np.ceil(inferior_band_mm / min(spacing)))` になっており、**Z 方向ではなく最小ボクセルサイズ**で割っています。
   　→ **`inferior_band_mm / spacing[2]`** で計算してください（本ケースでは 25/0.75 ≒ 34 スライス、現在は 37）。

2. **等方距離で統一**
   　肺・腹部臓器・大動脈の膨張、心膜ブリッジなど**全部 EDT ベース**にすると、Z 厚が大きい撮像でも mm パラメータの意味が一貫します。([scipy.github.io][3])

3. **“near\_heart”（ρ）の扱い**
   　連結塊の末端が ρ を超えて切られることがあります。
   　→ `max_heart_dist_mm=None` を許容し、**無効化できるように**しておくとよいです（初期検証は ρ なし→必要なら付与）。

4. **マスクの診断出力**
   　`ilam_addition = ilam_components & ~eat_pat_shell` を **NIfTI と PNG**で保存し、
   　- 追加体積（ml）
   　- 追加された連結成分数、最大成分の体積
   　をログに出すと、過学習/過小抽出の早期発見に役立ちます。

5. **安全弁（オプション）**
   　- **最大厚み制限**：`distance_transform_edt(~seed)` の geodesic 距離で塊の外側へ“伸びすぎ”を抑制（例：連結伝播後に seed からの geodesic 距離 > 20 mm を除外）。
   　- **骨/胸骨クランプ**：胸骨/肋骨 2–3 mm の禁止帯は、前胸壁近接例での混入を確実に減らします（TotalSegmentator の `sternum`/`rib_*` は利用可）。

---

## 置き換え部分の完成コード（ILAM 節）

以下だけ差し替えれば動きます（EDT膨張・ラベル法・Z帯域の z-spacing 修正込み）。既存の `aorta` PVAT 除外もそのまま併用可能です。

```python
# --- ILAM (Inferior Lung-Adjacency Mode) ---
print("\n14. Applying ILAM (Inferior Lung-Adjacency Mode)...")
print(f"  Parameters: inferior_band={inferior_band_mm} mm, "
      f"lung_contact={lung_contact_mm} mm, abdomen_gap={abdomen_gap_mm} mm, "
      f"max_heart_dist={max_heart_dist_mm} mm")

def iso_dilate(mask, radius_mm, spacing):
    if radius_mm <= 0: return mask.copy()
    dist = distance_transform_edt(~mask, sampling=spacing)
    return dist <= radius_mm

# 下部帯域は z_spacing で
z_spacing = float(spacing[2])
band_slices = int(np.ceil(inferior_band_mm / z_spacing))
z_switch = min(z_min + band_slices, z_max)
print(f"  ILAM Z-range: {z_min}..{z_switch} ({z_switch - z_min + 1} slices)")

# 等方膨張で禁止帯・接触帯を作る
lungs_touch = iso_dilate(lungs, lung_contact_mm, spacing)
abdomen_forbidden = iso_dilate(abdominal_organs, abdomen_gap_mm, spacing)

# 伝播ドメイン（torso_fat ∩ (~lungs) ∩ ~腹部臓器 ∩ [任意]心臓からの距離）
domain = (fat_clipped & (~lungs) & ~abdomen_forbidden)
if max_heart_dist_mm is not None:
    if 'dist_heart' not in locals():
        dist_heart = distance_transform_edt(~heart_mask, sampling=spacing)
    domain &= (dist_heart <= max_heart_dist_mm)

# Z で帯域制限
zmask = np.zeros_like(domain, bool); zmask[:, :, z_min:z_switch+1] = True
domain &= zmask

# シード：肺接触している torso_fat（帯域内）
seed = domain & lungs_touch

# 連結成分の“塊”を丸ごと採る
conn = generate_binary_structure(3, 2)  # 26-connectivity
lbl, n = label(domain, structure=conn)
touch_ids = np.unique(lbl[seed])
ilam_components = (lbl > 0) & np.isin(lbl, touch_ids)

# （任意）遠位大動脈 PVAT 除外
# ilam_components &= ~distal_aorta_forbidden

# 和集合（推奨）
pericardial_fat_mask = eat_pat_shell | ilam_components

# ログ
ilam_voxels = int(np.sum(ilam_components & ~eat_pat_shell))
print(f"  ILAM added voxels (beyond shell): {ilam_voxels:,}")
```

---

## 期待される見え方

この修正で、添付の 141–165 mm スライスの**肺近傍の縦隔脂肪が“帯”ではなく塊ごと**黄色で塗られるはずです。
もし追加される体積が大きすぎる場合は、次の順で抑えてください：

1. `inferior_band_mm` を 25 → 20 mm
2. `lung_contact_mm` を 1.5 → 1.0–1.2 mm
3. `max_heart_dist_mm` を 30 → 20–25 mm（または None→制限）
4. （必要なら）大動脈 PVAT の禁止帯を 2.5 → 3–4 mm

---

### 参考（API の意味づけ）

* **binary\_propagation** は「シードからマスク内を**連結に**広げる」演算＝マスク制約付き geodesic 膨張。今回の“塊ごと採る”に相性がよいです。([docs.scipy.org][1])
* **binary\_dilation** は構造要素と反復回数による近傍膨張で、mm を直接扱えません（非等方ボクセルでは歪みます）。([scipy.github.io][2])
* **distance\_transform\_edt** は `sampling=spacing` で物理スケール（mm）に正しく変換できます。今回の等方膨張や距離閾に必須です。([scipy.github.io][3], [tedboy.github.io][4])

---

必要ならこのパッチを組み込んだ **v4.1** の完全スクリプト体裁でも出せます。まずは上の差分で動かしてみてください。挙動が変わるはずです。

[1]: https://docs.scipy.org/doc/scipy-1.2.3/reference/generated/scipy.ndimage.binary_propagation.html "scipy.ndimage.binary_propagation — SciPy v1.2.3 Reference Guide"
[2]: https://scipy.github.io/devdocs/reference/generated/scipy.ndimage.binary_dilation.html "binary_dilation — SciPy v1.17.0.dev Manual"
[3]: https://scipy.github.io/devdocs/reference/generated/scipy.ndimage.distance_transform_edt.html "distance_transform_edt — SciPy v1.17.0.dev Manual"
[4]: https://tedboy.github.io/scipy/generated/scipy.ndimage.distance_transform_edt.html?utm_source=chatgpt.com "4.3.5.11. scipy.ndimage.distance_transform_edt"
