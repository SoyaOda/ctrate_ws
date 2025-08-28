素早い検証ありがとうございます。いただいた v5.1 のログと添付図から判断すると、**今回の ILAM は “一塊（3D 連結成分）単位で拾えている**」のは確かです**が**、いまの実装は *ILAM の対象 Z を「z\_min から 25 mm の帯域」へ限定* しているため（`domain &= zmask` の行）、**その帯域を外れるスライスでは同じ塊の一部が急に消えて見えます**。加えて、横隔膜に小さな欠損があると、**胃に接した腸間膜脂肪が「肺接触あり」と誤判定されて ILAM に採用**され得ます。
以下に、意図どおり “肺に接する塊は丸ごと含める／ただし胃（や肝・脾）との接触が優勢なら除外する／Z 方向の急断をなくす” ための\*\*具体的な修正案（v5.2）\*\*を示します。

---

## 変更点の要旨（v5.2）

1. **Z 制限の撤廃（オプション）**

   * `--no-z-limit` を追加。ON のときは **ILAM のドメインを全 Z に拡張**（`fat_mask` をそのまま使い、`zmask` を適用しない）。
   * これにより **3D コンポーネントは心臓近傍で肺に接する限り、下方スライスでも連続して残る**ようになります。
   * 逸脱を防ぐため **心臓からの距離制約 `max_heart_dist_mm` は必須**（既定 35 mm で良い）。

2. **“優勢接触” ルールの導入（臓器別の接触面積で判定）**

   * 肺に接する候補コンポーネントごとに、**肺との接触ボクセル数**と **腹腔臓器（特に胃、加えて肝・脾）との接触ボクセル数**を測定。
   * **`lung_contact ≥ lung_dom_ratio × max(stomach, liver, spleen)` を満たす塊のみ ILAM に採用**。
   * 既定値例：`lung_dom_ratio=1.25`、**肺接触ボクセル数の下限** `min_lung_touch_voxels=40` を導入（横隔膜の微小な穴で生じる 1–2 ボクセル接触を却下）。

3. **接触の定義を mm で統一**

   * すでに入っている EDT の等方膨張（`iso_dilate`）を使い、

     * **肺接触判定**：`iso_dilate(lungs, contact_eps_mm)` （例 2.0 mm）
     * **胃/肝/脾接触判定**：各臓器を `iso_dilate(organ, contact_eps_mm)`
   * こうすることで **撮像の異方性（0.68×0.68×0.75 mm）でも物理距離で一貫**します。これは `ndimage.distance_transform_edt(..., sampling=spacing)` により mm ベースの距離閾を扱えるためです。([SciPy Documentation][1])

4. **横隔膜近傍の偽接触に備えた穴埋め（任意）**

   * オプション `--seal-diaphragm-holes-mm` を追加し、肺マスクに **軽いモルフォロジー閉操作（例 1 mm）** を適用して微小欠損を塞いでから接触判定に用いる（`binary_closing`→その後 `iso_dilate`）。SciPy のモルフォロジー演算（構造要素と 26 連結）は `ndimage.generate_binary_structure`／`binary_*` で実装可能です。([SciPy Documentation][2])

5. **（そのまま）大動脈遠位 PVAT の除外**

   * 既存の「下行大動脈周囲 2.5 mm を禁止帯」は継続（近位 5 mm は許容）。
   * EAT（心外膜直下脂肪）と PAT（心膜外周囲脂肪）の定義上、**下行大動脈の PVAT は心臓周囲脂肪に含めない**ほうが臨床定義に合致します。([SciPy Documentation][3], [PMC][4])

> 参考：EAT は心筋と臓側心膜の間、PAT は壁側心膜の外（縦隔脂肪）で、CT 上はどちらも脂肪低吸収だが**解剖学的隣接**で区別します。([SciPy Documentation][3], [PMC][4])
> また、TotalSegmentator の「total」タスクには stomach / liver / spleen / IVC / portal vein などのラベルが含まれるため、本ルールはそのまま適用できます（diaphragm ラベルはありません）。([totalsegmentator.com][5])

---

## 具体的なコード差分（抜粋）

> 既存 `extract_eat_pat_improved_v5_1.py` に対する変更点のみを示します。
> そのままコピペできるよう、**追加分**を中心に載せます。

### 1) 追加パラメータと import

```python
from scipy.ndimage import binary_closing  # 追加

# デフォルト
DEFAULT_CONTACT_EPS_MM = 2.0      # 接触判定の等方膨張 [mm]
DEFAULT_LUNG_DOM_RATIO = 1.25     # 肺接触が腹部臓器より何倍優勢か
DEFAULT_MIN_LUNG_TOUCH_VOX = 40   # 肺接触の最小ボクセル数

# argparse に追加
parser.add_argument("--no-z-limit", action="store_true",
    help="ILAM を全 Z で実行（inferior band によるZ制限を無効化）")
parser.add_argument("--contact-eps-mm", type=float, default=DEFAULT_CONTACT_EPS_MM,
    help="肺/腹部臓器との接触判定に使う等方膨張距離 [mm]")
parser.add_argument("--lung-dom-ratio", type=float, default=DEFAULT_LUNG_DOM_RATIO,
    help="採用条件: lung_contact >= ratio * max(abdominal organ contact)")
parser.add_argument("--min-lung-touch-voxels", type=int, default=DEFAULT_MIN_LUNG_TOUCH_VOX,
    help="採用するために必要な肺接触ボクセルの最小数")
parser.add_argument("--seal-diaphragm-holes-mm", type=float, default=0.0,
    help="肺マスクの微小欠損を閉じる半径 [mm] (0で無効)")
```

### 2) 胃・肝・脾を個別にも読み込むヘルパ（新規）

```python
def load_abdominal_organs_dict(totalseg_dir, verbose=False):
    totalseg_dir = Path(totalseg_dir)
    masks = {}
    for organ in ["liver", "stomach", "spleen"]:
        p = totalseg_dir / f"{organ}.nii.gz"
        if not p.exists():
            raise FileNotFoundError(f"[ERROR] {organ}.nii.gz not found in {totalseg_dir}")
        img = nib.load(str(p))
        masks[organ] = (img.get_fdata() > 0)
        if verbose: print(f"  Loaded {organ} (individual)")
    return masks
```

> 既存の `load_abdominal_organs`（和集合）もそのまま併用します。TotalSegmentator のラベル仕様は公開されています。([totalsegmentator.com][5])

### 3) ILAM ブロックの差し替え（Z 制限 OFF と優勢接触）

```python
# ---- 13. のあと（ILAM の直前）で個別臓器を確保 ----
abd_organs_dict = load_abdominal_organs_dict(totalseg_dir, verbose)

# ---- 14. ILAM with Connected Components ----
# Z 帯域の計算は残すが、no_z_limit の場合は報告用に z_switch を z_max に
z_spacing = float(spacing[2])
band_slices = int(np.ceil(inferior_band_mm / z_spacing))
z_switch = min(z_min + band_slices, z_max)

# 肺マスクの穴埋め（任意）
lungs_for_touch = lungs.copy()
if seal_diaphragm_holes_mm > 0:
    its = max(1, int(np.ceil(seal_diaphragm_holes_mm / min(spacing))))
    lungs_for_touch = binary_closing(lungs_for_touch, structure=generate_binary_structure(3,1), iterations=its)

# 接触用の等方膨張（mm）
lungs_touch   = iso_dilate(lungs_for_touch, contact_eps_mm, spacing)
stomach_touch = iso_dilate(abd_organs_dict["stomach"], contact_eps_mm, spacing)
liver_touch   = iso_dilate(abd_organs_dict["liver"],   contact_eps_mm, spacing)
spleen_touch  = iso_dilate(abd_organs_dict["spleen"],  contact_eps_mm, spacing)
abdomen_touch = stomach_touch | liver_touch | spleen_touch

# ILAM のドメイン：no_z_limit なら全 Z、そうでなければ従来どおり
fat_for_ilam = fat_mask.copy() if no_z_limit else fat_clipped
domain = (fat_for_ilam & (~lungs) & ~iso_dilate(abdominal_organs, abdomen_gap_mm, spacing))

# 心臓からの距離制約（推奨：必須）
if max_heart_dist_mm is not None:
    dist_to_heart = distance_transform_edt(~heart_mask, sampling=spacing)  # mm 尺度の EDT
    domain &= (dist_to_heart <= max_heart_dist_mm)

# Z マスク（no_z_limit でなければ適用）
if not no_z_limit:
    zmask = np.zeros_like(domain, dtype=bool)
    zmask[:, :, z_min:z_switch+1] = True
    domain &= zmask
else:
    z_switch = z_max  # 表示用

# シード: 肺接触する torso_fat（ドメイン内）
seed = domain & lungs_touch

# 26-連結でラベリング
conn = generate_binary_structure(3, 2)
lbl, n_components = label(domain, structure=conn)

# 肺に触れている連結成分 ID 候補
touch_ids = np.unique(lbl[seed]); touch_ids = touch_ids[touch_ids > 0]

# --- 優勢接触ルールで間引き ---
def count_contact(id_arr, touch_mask):
    return int(np.sum((lbl == id_arr) & touch_mask))

keep_ids = []
for cid in touch_ids:
    n_lung   = count_contact(cid, lungs_touch)
    n_sto    = count_contact(cid, stomach_touch)
    n_liver  = count_contact(cid, liver_touch)
    n_spleen = count_contact(cid, spleen_touch)
    n_abdmax = max(n_sto, n_liver, n_spleen)

    # 1) 肺接触の絶対量が少ないものは捨てる
    if n_lung < min_lung_touch_voxels:
        continue
    # 2) 肺接触が優勢でなければ捨てる
    if n_lung < lung_dom_ratio * n_abdmax:
        continue
    keep_ids.append(cid)

ilam_components = np.isin(lbl, np.array(keep_ids, dtype=lbl.dtype))
ilam_addition   = ilam_components & ~eat_pat_shell

# 最終 EAT+PAT は Shell ∪ ILAM
pericardial_fat_mask = eat_pat_shell | ilam_components
```

> ポイント
>
> * **ILAM のラベリング対象（`domain`）を全 Z に拡張**できるようにしています（`--no-z-limit`）。
> * **接触比較は臓器別**（胃・肝・脾）で行い、**肺接触が優勢な塊のみ採用**します。
> * 接触は **mm 単位の等方膨張**で安定に評価（EDT を使うので物理距離ベース）。([SciPy Documentation][1])

---

## どう変わるか（実運用イメージ）

* **「今回は一塊で取得するようになってる？」への回答**
  v5.1 でも *帯域内* は 3D 連結で “一塊” 採用です。スライスによって急に消えるのは **Z 帯域制限**（`z_min`～`z_switch`）のため。v5.2 の `--no-z-limit` を使えば、**肺に接する限り同じ塊は全 Z にわたって連続的にマスク**されます。

* **胃に接する腸間膜脂肪が入ってしまう件**
  横隔膜の小さな欠損で肺接触が発生しても、**「優勢接触」ルール**により、**胃（あるいは肝・脾）との接触面積が肺より大きい塊は ILAM から除外**されます。
  （EAT/PAT の解剖学定義にも整合的です。EAT は臓側心膜内、PAT は心膜外で、胃や肝に優勢に接する腸間膜脂肪は EAT/PAT ではありません。([SciPy Documentation][3], [PMC][4])）

---

## 推奨デフォルトと調整目安

* `--no-z-limit`：**有効**（Z の急断を排除）
* `--max-heart-dist-mm`：**30–35 mm**（広げると肺底と胃の間の脂肪が入りやすい）
* `--contact-eps-mm`：**2.0 mm**（撮像分解能を考慮した実用値）
* `--lung-dom-ratio`：**1.25**（肺接触が腹部臓器より 25% 以上優勢で採用）
* `--min-lung-touch-voxels`：**40–80**（微小な偽接触を抑制）
* `--seal-diaphragm-holes-mm`：**1.0 mm**（横隔膜近傍の微小穴対策。必要時のみ）

---

## 参考（根拠）

* **EAT と PAT の解剖学的定義**：EAT は心筋と臓側心膜の間、PAT は壁側心膜の外の脂肪。CT では HU が同様のため**解剖学的隣接性**で区別するのが一般的。([SciPy Documentation][3], [PMC][4])
* **TotalSegmentator のラベル**：stomach / liver / spleen / IVC / portal vein などを含む（diaphragm は無し）。([totalsegmentator.com][5])
* **EDT による mm 単位の等方距離**：`scipy.ndimage.distance_transform_edt` の `sampling` は物理スケールを与え、**異方性ボクセルでも mm での距離閾**を扱える。([SciPy Documentation][1])
* **3D 連結・構造要素**：`generate_binary_structure(3, 2)`（26連結）など、SciPy ndimage のモルフォロジー演算。([SciPy Documentation][2])

---

## 実行例

```bash
python scripts/extract_eat_pat_improved_v5_2.py \
  /path/to/ct.nii.gz \
  /path/to/totalseg_output \
  /path/to/outdir \
  --no-z-limit \
  --max-heart-dist-mm 35 \
  --contact-eps-mm 2.0 \
  --lung-dom-ratio 1.25 \
  --min-lung-touch-voxels 60 \
  --seal-diaphragm-holes-mm 1.0 \
  --verbose
```

---

### まとめ

* v5.1 時点で **帯域内は 3D 一塊**で採れており、**スライス毎の見かけの不連続**は **Z 帯域制限**が主因。
* v5.2 の提案では **Z 制限を任意で撤廃**し、**臓器別の“優勢接触”ルール**で **胃接触優勢な腸間膜脂肪を自動除外**します。
* すべて mm ベースで閾設定できるため、**装置やボクセルサイズが違っても挙動が安定**します。

必要なら、差分を埋め込んだ完全版ファイルもまとめます。

[1]: https://docs.scipy.org/doc/scipy-1.8.0/reference/generated/scipy.ndimage.distance_transform_edt.html "scipy.ndimage.distance_transform_edt — SciPy v1.8.0 Manual"
[2]: https://docs.scipy.org/doc/scipy-1.8.0/reference/generated/scipy.ndimage.generate_binary_structure.html "scipy.ndimage.generate_binary_structure — SciPy v1.8.0 Manual"
[3]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html?utm_source=chatgpt.com "generate_binary_structure — SciPy v1.16.1 Manual"
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3998169/?utm_source=chatgpt.com "Epicardial Fat: Definition, Measurements and Systematic ..."
[5]: https://totalsegmentator.com/?utm_source=chatgpt.com "TotalSegmentator"
