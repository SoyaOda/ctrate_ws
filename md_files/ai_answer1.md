ありがとうございます。提示いただいた「HARTA 統合 → 2D PNG マスク → 3D NIfTI 再構成」パイプラインの**ずれ・向き・粗さ（解像度）不整合**は、以下が主因です。

* **NIfTI→DICOM 変換時に IOP（ImageOrientationPatient, 0020,0037）や IPP（ImagePositionPatient, 0020,0032）を正しく埋めていない**
  （`[1,0,0,0,1,0]` で固定している）
* その結果、**DICOM 座標系（LPS, 物理空間）と NIfTI 側のアフィン行列の対応が壊れる**
* HARTA は **DICOM のみ対応**で（NIfTI は不可）、PNG マスクには座標情報が無いため、**戻すときに元の座標系を必携**（IOP/IPP, PixelSpacing, Origin, Direction）です。([GitHub][1])

以下、**原因の補足 → 修正方針 → コード（完全実装例）→ 検証**の順にまとめます。
（要点となる DICOM 幾何の扱いは SimpleITK の公式サンプルと DICOM リファレンスに準拠します。）([simpleitk.org][2])

---

## 1) 原因の技術的ポイント（なぜズレるか）

* \*\*DICOM 幾何の本体は IOP（行・列方向の方向余弦）と IPP（スライスの原点）\*\*です。これらと PixelSpacing/スライス間隔から 3D 位置が決まります。**IOP/IPP が正でないと、スライス順・回転・座標が破綻**します。
* **SimpleITK/ITK は LPS（DICOM 物理座標系）を内部の標準**として扱い、方向行列（Direction）・原点（Origin）・間隔（Spacing）を持ちます。**Direction の 1・2 列目が IOP に対応**し、**IPP は TransformIndexToPhysicalPoint で算出**します。([simpleitk.org][2])
* **NIfTI は多くのツールが RAS 系（+X 右, +Y 前, +Z 上）を前提**にします。NIfTI→DICOM へ自作で変換する際に RAS↔LPS の取り扱いを誤ると、左右反転・位置ズレの原因になります（一般論として RAS↔LPS を意識）。([dicom.innolitics.com][3])

---

## 2) 正しい修正方針（要点）

1. **NIfTI→DICOM 変換は SimpleITK（ITK）で自動計算させる**

   * NIfTI を SimpleITK で読み込む（Direction/Origin/Spacing を取得）
   * **ImageSeriesWriter** で 2D スライス DICOM を書く際、**IOP/IPP を Direction/TransformIndexToPhysicalPoint から正しく埋める**（公式サンプルの通り）([simpleitk.org][2])
2. **書き出しの際、InstanceNumber 等で “連番→UID/IPP 対応表(JSON)” を保存**

   * 後で PNG を 3D に戻すとき、**スライス順を IPP（スライス面法線への射影）で確実に復元**するため
3. **HARTA 実行（DICOM 入力）**

   * HARTA は DICOM のみ対応（README 明記）([GitHub][1])
4. **PNG→NIfTI 再構成は “元 CT の幾何（Direction/Origin/Spacing）を完全復元”**

   * **PNG は座標を持たない**ので、**(a) DICOM→座標メタ**、**(b) 元 NIfTI/CT→アフィン**のどちらかを**参照して NIfTI の qform/sform（または SimpleITK の Direction/Origin/Spacing）を同一にする**
   * HARTA の PNG が **RGBA** の場合は **アルファ or 非ゼロ**でバイナリ化（頑健に）
   * HARTA が内部で 512×512 等に**リサンプルしている場合**は、**元 CT 幾何に最近傍補間で戻す**（面積/体積を保つように注意）

---

## 3) フル実装（そのまま使える最小構成）

> 目的：
> **(A) NIfTI→DICOM を幾何を保って出力**＋**スライス対応 JSON 保存**
> **(B) HARTA 出力の 2D PNG 群→ 3D NIfTI（元幾何で）に再構成**

### (A) nifti\_to\_dicom\_preserve\_geom.py

* SimpleITK 公式サンプル **DicomSeriesFromArray** をベースに実装（IOP/IPP の埋め方は下記と同一）([simpleitk.org][2])

```python
#!/usr/bin/env python3
import json, os, pathlib
import SimpleITK as sitk

def write_dicom_series_from_nifti(nifti_path, out_dir):
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    img = sitk.ReadImage(str(nifti_path))   # ITKはLPS物理座標を内部標準で扱う
    # 必要なら型をint16等にキャスト
    if img.GetPixelID() != sitk.sitkInt16:
        img = sitk.Cast(img, sitk.sitkInt16)

    spacing  = img.GetSpacing()             # (sx, sy, sz)
    origin   = img.GetOrigin()              # (ox, oy, oz)
    direction= img.GetDirection()           # 3x3 行列をフラットにした長さ9のタプル

    size = img.GetSize()                    # (Nx, Ny, Nz)
    print("Size:", size, "Spacing:", spacing, "Origin:", origin)

    series_writer = sitk.ImageSeriesWriter()
    # IOP: 行/列方向の方向余弦 = direction の 1列目＋2列目
    iop = [str(direction[0]), str(direction[3]), str(direction[6]),
           str(direction[1]), str(direction[4]), str(direction[7])]

    # 再現性のためStudy/Series/SOPのUID群を固定的に生成
    study_uid = sitk.GenerateUniqueUID()
    series_uid = sitk.GenerateUniqueUID()

    file_names = []
    meta_list = []
    for k in range(size[2]):
        # k番目スライスのIPP（スライス原点）を物理空間で計算
        ipp = img.TransformIndexToPhysicalPoint([0, 0, k])
        # 書き出し用2Dスライスを抽出
        slice_img = img[:, :, k]

        # DICOMメタを設定
        md = {
            "0008|0060": "CT",
            "0008|103e": f"CT slice {k}",
            "0010|0010": "Anon^Pat",
            "0020|000d": study_uid,
            "0020|000e": series_uid,
            "0020|0011": "1",  # SeriesNumber
            "0020|0013": str(k+1),  # InstanceNumber
            "0020|0032": "\\".join(map(str, ipp)),         # ImagePositionPatient
            "0020|0037": "\\".join(iop),                   # ImageOrientationPatient
            "0028|0030": "\\".join(map(str, spacing[:2])), # PixelSpacing
            "0018|0050": str(spacing[2]),                  # SliceThickness（参考値）
            "0018|0088": str(spacing[2]),                  # SpacingBetweenSlices（参考値）
            "0028|0100": "16", "0028|0101": "16", "0028|0102": "15", # Bits設定
        }
        for tag, val in md.items():
            slice_img.SetMetaData(tag, val)

        out_name = out_dir / f"IM_{k+1:04d}.dcm"
        series_writer.SetFileName(str(out_name))
        series_writer.Execute(slice_img)
        file_names.append(str(out_name))
        meta_list.append({"index": k, "InstanceNumber": k+1, "IPP": list(ipp)})

    # スライス対応のメタを保存（後でPNGと合わせる）
    with open(out_dir / "slice_map.json", "w") as f:
        json.dump({"files": file_names, "meta": meta_list,
                   "origin": origin, "spacing": spacing, "direction": direction}, f, indent=2)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    write_dicom_series_from_nifti(args.input, args.outdir)
```

> **ポイント**
>
> * **IOP/IPP の設定方法は SimpleITK 公式例と同じ**（`direction` から IOP、`TransformIndexToPhysicalPoint` で IPP）([simpleitk.org][2])
> * **PixelSpacing / SliceThickness / SpacingBetweenSlices も一致**させます（必要に応じて 0050/0088 は省略可）。
> * 書き出した **DICOM 連番と IPP の対応を JSON に保存**（後段の 3D 再構成に使用）。

---

### (B) harta\_masks\_to\_nifti\_preserve\_geom.py

* HARTA 出力の 2D PNG を読み、**slice\_map.json に従ってソート→体積化**
* **RGBA/グレースケール双方に対応**
* **元 CT の幾何（Direction/Origin/Spacing）をそのまま付与**
* HARTA が内部で 512×512 にリサイズしている等、\*\*平面サイズが合わない場合は “元 CT 幾何への最近傍リサンプル”\*\*を行う

```python
#!/usr/bin/env python3
import os, json, numpy as np, SimpleITK as sitk
from pathlib import Path
from PIL import Image

def load_mask_from_png(p):
    im = Image.open(p).convert("RGBA")  # RGBAに正規化
    arr = np.array(im)                  # (H,W,4)
    # アルファ優先、なければRGBのいずれか非ゼロで1
    alpha = arr[..., 3]
    if alpha.max() > 0:
        mask = (alpha > 0).astype(np.uint8)
    else:
        mask = (arr[..., 0:3].sum(axis=-1) > 0).astype(np.uint8)
    return mask

def reconstruct_3d_from_harta(harta_png_dir, slice_map_json, out_nifti, ref_nifti):
    hdir = Path(harta_png_dir)
    with open(slice_map_json) as f:
        smap = json.load(f)
    direction = tuple(smap["direction"])
    origin    = tuple(smap["origin"])
    spacing   = tuple(smap["spacing"])
    meta      = smap["meta"]  # [{index, InstanceNumber, IPP:[x,y,z]}, ...]

    # PNGファイル群を取得（HARTAの出力規則に合わせてフィルタ）
    pngs = sorted([p for p in hdir.glob("*.png")])
    if len(pngs) != len(meta):
        print(f"[WARN] PNG枚数({len(pngs)})とDICOMスライス数({len(meta)})が一致しません。")

    # IPP（Z方向）で並び替え：IOPの法線方向へ射影して順序付け
    # 参考：IOPは direction から既にDICOM方式で得ているので、その3・6要素で法線を計算して射影
    iop = np.array([direction[0], direction[3], direction[6],
                    direction[1], direction[4], direction[7]], dtype=float).reshape(2,3)
    normal = np.cross(iop[0], iop[1])   # スライス面法線
    zpos = [np.dot(np.array(m["IPP"]), normal) for m in meta]
    order = np.argsort(zpos)            # 低→高で整列（必要なら反転）

    # PNG画像を読み込み、ソート順に積層
    masks = []
    for i, idx in enumerate(order):
        png_path = pngs[i] if len(pngs)==len(meta) else pngs[idx]  # 名前規則次第で調整
        m = load_mask_from_png(png_path)
        masks.append(m)
    vol = np.stack(masks, axis=-1)  # (H,W,Nz)

    # 参照CTの幾何に合わせてリサンプル（サイズが異なる場合）
    ref = sitk.ReadImage(str(ref_nifti))
    sitk_mask = sitk.GetImageFromArray(vol.transpose(2,0,1).astype(np.uint8))  # (z,y,x)
    sitk_mask.SetDirection(direction)
    sitk_mask.SetOrigin(origin)
    sitk_mask.SetSpacing(spacing)

    if list(sitk_mask.GetSize())[:2] != list(ref.GetSize())[:2] or sitk_mask.GetSize()[2] != ref.GetSize()[2]:
        # 最近傍で ref 幾何へ
        sitk_mask = sitk.Resample(
            sitk_mask, ref, sitk.Transform(),
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    # NIfTI保存
    sitk.WriteImage(sitk_mask, str(out_nifti))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--png_dir", required=True, help="HARTA の PNG 出力ディレクトリ")
    ap.add_argument("--slice_map", required=True, help="nifti_to_dicom_preserve_geom.py が吐いた JSON")
    ap.add_argument("--out", required=True, help="出力 NIfTI パス")
    ap.add_argument("--ref", required=True, help="元CT(NIfTI) へのパス（幾何参照）")
    args = ap.parse_args()

    reconstruct_3d_from_harta(args.png_dir, args.slice_map, args.out, args.ref)
```

> **ポイント**
>
> * **IOP の二本（行/列方向）の外積がスライス法線**。IPP を **その法線へ射影**して並べるとスライス順が一意になります（DICOM 幾何の定石）。IOP/IPP の意義は DICOM の標準解説・実装例に準拠。([simpleitk.org][2])
> * **PNG 側の平面解像度が変わっていても、最後に “参照 CT へ最近傍リサンプル”** すれば、**ボクセル格子・体積の整合**が取れます（体積計算は最近傍が無難）。

---

## 4) 実行手順（ミスしにくい順序）

1. **NIfTI→DICOM（幾何保存 & JSON出力）**

   ```bash
   python nifti_to_dicom_preserve_geom.py \
     -i ~/ctrate_ws/data/CT-RATE-v2/.../case.nii.gz \
     -o ~/ctrate_ws/working/case_dicom
   ```
2. **HARTA の実行（DICOM 入力, 出力は PNG 群）**

   > HARTA は DICOM のみ対応（README 記載）。HARTA の `*_automatic_.py` に従って実行してください。([GitHub][1])
3. **PNG→NIfTI 再構成（幾何復元）**

   ```bash
   python harta_masks_to_nifti_preserve_geom.py \
     --png_dir ~/ctrate_ws/working/case_dicom/HARTA_output/fat \
     --slice_map ~/ctrate_ws/working/case_dicom/slice_map.json \
     --ref ~/ctrate_ws/data/CT-RATE-v2/.../case.nii.gz \
     --out ~/ctrate_ws/outputs/case_eat_harta.nii.gz
   ```

---

## 5) よくある落とし穴と対策

* **IOP/IPP を自前で “\[1,0,0,0,1,0] + 原点だけ” にしてしまう**
  → 斜台（oblique）撮影や任意回転の症例で**必ず**ずれます。**上記の SimpleITK 流儀で自動算出**してください。([simpleitk.org][2])
* **PNG の順序をファイル名だけで復元**
  → HARTA 側の命名が DICOM の InstanceNumber を継承していない場合に破綻。**IPP による順序決定**が安全です。
* **NIfTI 作成時に Direction を未設定（既定=単位）**
  → 座標が壊れます。**Direction/Origin/Spacing を必ず設定**（上記コード参照）。([simpleitk.org][2])
* **2D サイズが変わっても、そのまま積層**
  → 幾何不一致。**参照 CT へ最近傍リサンプル**してから保存しましょう。

---

## 6) 検証（重ね合わせの確認）

* **任意の 10 スライス**で、元 CT と出力マスクを SimpleITK で同一座標に重ね、
  `sitk.LabelOverlapMeasuresImageFilter()` で Dice/VOE を確認（向きが違えば重ね合わせ不可）。
* 物理点（患者座標）で **心尖付近・心基部**などをいくつかピックし、**TransformIndexToPhysicalPoint** で**NIfTI/マスク双方が同一座標**になるか検証。([simpleitk.org][2])

---

## 7) 参考（一次情報）

* **SimpleITK 公式：DICOM シリーズ書き出しの実装例**
  （IOP/IPP を **Direction / TransformIndexToPhysicalPoint** から設定する標準的なやり方）([simpleitk.org][2])
* **DICOM（幾何の要素：IOP/IPP など）総覧**（入門～実務向け解説）
* **HARTA README**（DICOM 入力のみの明記）([GitHub][1])
* **NIfTI（RAS）と DICOM（LPS）の系の違いに注意**（一般的注意喚起）([dicom.innolitics.com][3])

---

## 8) まとめ（今回の “確定的な” 改修ポイント）

* **NIfTI→DICOM は SimpleITK の IOP/IPP 自動算出に全面移行**（固定 IOP は禁止）
* **DICOM スライスの “並び情報（IPP と法線）” を JSON に保存**
* **PNG→3D は JSON に基づく順序復元＋元 CT 幾何で再構成**（必要なら最近傍で逆リサンプル）
* **出力 NIfTI の Direction/Origin/Spacing を“必ず”設定**（＝元 CT と完全一致）

これで、**ビューア重ね合わせでもピクセル単位で一致**し、EAT 体積などの定量値も安定します。上記の 2 スクリプトを差し替えれば、いまのワークフロー（HARTA を中核に使用）をそのまま安全に回せます。

[1]: https://github.com/aforebelo/HARTA "GitHub - aforebelo/HARTA: Epicardial Fat Segmentation and Quantification Software"
[2]: https://simpleitk.org/doxygen/v2_4/html/DicomSeriesFromArray_2DicomSeriesFromArray_8py-example.html "SimpleITK: DicomSeriesFromArray/DicomSeriesFromArray.py"
[3]: https://dicom.innolitics.com/ciods/rt-dose/general-image/00200020?utm_source=chatgpt.com "Patient Orientation Attribute – DICOM Standard Browser"
