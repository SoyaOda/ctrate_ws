了解です。ここでは **HARTA**（aforebelo/HARTA）を、いまお使いの **WSL2 + Docker 環境／CT‑RATE v2** 前提のプロジェクトに統合し、**学習なし・自動で EAT（Epicardial Adipose Tissue）を得る**ための、曖昧性のない“動く”実装プラン（スクリプト付き）を提示します。
HARTA は **Python 3.8.3** で動作し、**入力は DICOM シリーズのみ**、主に **非造影の心臓/胸部CTに最適化**、実装は **クラシカル（非DL）** で、自動モード／半自動モードを備えています（GUI は `harta.py`、バッチは `_automatic_.py` / `_semiautomatic_.py`）。ライセンスは **CC‑BY‑NC‑4.0** です。([GitHub][1])

> **重要な要点（HARTA 仕様）**
>
> * Python 3.8.3 / pip で依存関係を導入して実行（`pip install -r requirements.txt` → `python3 harta.py`）。([GitHub][1])
> * **DICOM シリーズのみ受け付け**（NIfTIは不可）。非造影CTに最適化。([GitHub][1])
> * 自動・半自動スクリプト（`_automatic_.py`, `_semiautomatic_.py`）が同梱。([GitHub][2])
> * 非商用研究前提の CC‑BY‑NC‑4.0。([GitHub][1])

---

## 全体像（統合フロー）

```
CT-RATE v2 (NIfTI, *_fixed/)
   └─(1) NIfTI → DICOM 変換（Plastimatch）
        └─(2) HARTA（自動モード）で EAT セグメンテーション
             └─(3) HARTA 出力（スライス毎の2Dマスク）を3Dに再構成 → NIfTI保存
                  └─(4) 任意で 2D脂肪投影（fatmap）/ DRR とペアリング
```

* **(1) の変換**は Plastimatch の `convert` コマンドで NIfTI→DICOM が可能（`plastimatch convert --input file.nii.gz --output-dicom dicom_out/`）。([MONAI][3])
* **(2)** は HARTA の自動スクリプトを DICOM フォルダに対して実行（EAT 抽出）。
* **(3)** は HARTA の 2D 出力（PNG/TIFF など）を DICOM の幾何情報で 3D に積層して **NIfTI** 化。
* **(4)** はお手元のパイプライン（Plastimatch の DRR、NumPy での投影）に接続。

---

## 0) 推奨ディレクトリ（WSL2 内）

```bash
# 例）$HOME/ctrate_ws を全体の作業ルートに
~/ctrate_ws/
├── data/
│   ├── CT-RATE-v2/                  # 既にダウンロード済みの *_fixed/ NIfTI
│   └── dicom/                       # NIfTI→DICOM 変換の出力先（症例ごと）
├── outputs/
│   ├── harta_raw/                   # HARTA が吐く生の結果（スライス毎のマスクなど）
│   └── nifti/                       # 3D に積層して NIfTI 化した EAT マスク
├── tools/
│   └── HARTA/                       # 本家リポジトリ
└── scripts/                         # ここに本回答のスクリプト群を置く
```

---

## 1) HARTA のセットアップ（Python 3.8.3 環境）

HARTA は Python 3.8.3 を明記しています。conda を使うと衝突なく導入できます。([GitHub][1])

```bash
# Miniconda / Mambaforge がない場合はインストール推奨（省略）
# 環境作成
conda create -y -n harta38 python=3.8.3
conda activate harta38

# HARTA を取得
cd ~/ctrate_ws/tools
git clone https://github.com/aforebelo/HARTA.git
cd HARTA

# 依存関係
pip install -r requirements.txt   # README 記載の手順通り。:contentReference[oaicite:7]{index=7}
```

> 補足：HARTA は **DLフレームワーク不要（非DL）** で、OpenCV・NumPy・pydicom 系の軽量依存が中心です（README の「Implementation is mostly based on basic image operations… No training required.」）。([GitHub][2])

---

## 2) NIfTI → DICOM 変換（症例毎）

CT‑RATE v2 は NIfTI（\*\_fixed/）なので、まず DICOM へ変換します。Plastimatch で一括変換します。([MONAI][3])

**`scripts/nifti_to_dicom.sh`**（コピペで保存）：

```bash
#!/usr/bin/env bash
set -euo pipefail

# 使い方: ./nifti_to_dicom.sh /path/to/case.nii.gz /out/dicom/CASE_ID
NII="$1"
OUT="$2"

mkdir -p "$OUT"
plastimatch convert --input "$NII" --output-dicom "$OUT"
echo "[OK] Converted: $NII -> $OUT"
```

実行例：

```bash
bash ~/ctrate_ws/scripts/nifti_to_dicom.sh \
  ~/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz \
  ~/ctrate_ws/data/dicom/valid_1_a_1
```

> **注意**：HARTA は DICOM 前提（NIfTIは直接不可）。非造影データに最適化と明記されています。([GitHub][1])

---

## 3) HARTA（自動モード）のバッチ実行

HARTA の README は GUI 実行（`python3 harta.py`）に加え、**自動処理／半自動のスクリプト**が同梱されている旨を案内しています（`_automatic_.py`, `_semiautomatic_.py`）。([GitHub][2])
ここでは **自動モード**をバッチで回す想定のラッパーを用意します。

**`scripts/run_harta_auto.py`**：

```python
#!/usr/bin/env python3
import argparse, subprocess, sys, os, glob, shlex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harta_repo", required=True, help="HARTA repo path (~/ctrate_ws/tools/HARTA)")
    ap.add_argument("--dicom_root", required=True, help="Root dir of DICOM cases (each case is a folder)")
    ap.add_argument("--out_root", required=True, help="Output root for raw HARTA results")
    ap.add_argument("--python", default=sys.executable, help="Python executable in harta env")
    args = ap.parse_args()

    auto_py = os.path.join(args.harta_repo, "_automatic_.py")
    if not os.path.exists(auto_py):
        sys.exit(f"ERROR: _automatic_.py not found at {auto_py}")

    cases = sorted([p for p in glob.glob(os.path.join(args.dicom_root, "*")) if os.path.isdir(p)])
    if not cases:
        sys.exit("No DICOM case folders found")

    os.makedirs(args.out_root, exist_ok=True)

    for case_dir in cases:
        case_id = os.path.basename(case_dir.rstrip("/"))
        out_dir = os.path.join(args.out_root, case_id)
        os.makedirs(out_dir, exist_ok=True)

        # HARTA の自動スクリプトを呼ぶ。実際の引数は --help を参照し適宜合わせる。
        # 多くのツールが -d (dicom) -o (output) を採用するため、まずはそれを試し、
        # 失敗する場合は --help を出してユーザにログで可視化。
        cmd_try = f'{shlex.quote(args.python)} {shlex.quote(auto_py)} -d {shlex.quote(case_dir)} -o {shlex.quote(out_dir)}'
        print(f"[RUN] {cmd_try}")
        ret = subprocess.call(cmd_try, shell=True)
        if ret != 0:
            # ヘルプを表示してログに残す（引数シグネチャの違いに備える）
            help_cmd = f'{shlex.quote(args.python)} {shlex.quote(auto_py)} --help'
            print(f"[INFO] _automatic_.py returned {ret}. Printing help:")
            subprocess.call(help_cmd, shell=True)
            sys.exit(f"Adjust arguments according to _automatic_.py --help")

    print("[OK] HARTA auto finished for all cases")

if __name__ == "__main__":
    main()
```

実行例：

```bash
conda activate harta38
python ~/ctrate_ws/scripts/run_harta_auto.py \
  --harta_repo ~/ctrate_ws/tools/HARTA \
  --dicom_root ~/ctrate_ws/data/dicom \
  --out_root  ~/ctrate_ws/outputs/harta_raw
```

> ※ **CLI 引数はリポジトリの実装に依存**します。HARTA 側の `_automatic_.py --help` を必ず確認し、`-d`（DICOMパス）/`-o`（出力）など、正しいオプションに合わせてください（このラッパーは失敗時に `--help` を表示して誘導します）。自動・半自動スクリプトが同梱されていること自体は README に明記。([GitHub][2])

---

## 4) HARTA の出力 → 3D NIfTI へ再構成

HARTA はスライス毎の 2D マスク（PNG/ビットマップ等）を症例ディレクトリ下に生成する設計が一般的です（実出力のファイル名は実行後に `~/ctrate_ws/outputs/harta_raw/<case_id>` をご確認ください）。以下のスクリプトは「**元DICOMの並び順と同じ順でマスク画像を積層**」し、DICOM の PixelSpacing / SliceThickness を用いて正しい **NIfTI（3D）** に書き出します。

**`scripts/harta_masks_to_nifti.py`**：

```python
#!/usr/bin/env python3
import argparse, os, glob, numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image

def read_dicom_series(dicom_dir):
    files = sorted(glob.glob(os.path.join(dicom_dir, "*")), key=lambda x: pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber)
    refs  = [pydicom.dcmread(fp) for fp in files]
    # 画素配列のサイズを確認
    rr, cc = int(refs[0].Rows), int(refs[0].Columns)
    # スペーシング（mm）
    ps = [float(x) for x in refs[0].PixelSpacing]  # [row, col] mm
    try:
        st = float(refs[0].SliceThickness)
    except Exception:
        # Slice間隔から推定
        zs = [float(ds.ImagePositionPatient[2]) for ds in refs]
        zs_sorted = sorted(zs)
        diffs = [abs(zs_sorted[i+1]-zs_sorted[i]) for i in range(len(zs_sorted)-1)]
        st = np.median(diffs) if diffs else 1.0
    spacing = (ps[1], ps[0], st)  # (x, y, z)

    return refs, (rr, cc), spacing, files

def load_harta_slice_masks(mask_dir, target_shape):
    # 例：EAT*.png 等、マスク画像群を読み込む想定。必要に応じてパターン調整。
    # HARTA 出力を確認し、適切なグロブに差し替えてください。
    cand = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not cand:
        cand = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
    if not cand:
        raise RuntimeError(f"No mask images found under {mask_dir}")

    # 画像 → 2D バイナリ配列へ
    masks = []
    for fp in cand:
        im = Image.open(fp).convert("L")
        arr = np.array(im, dtype=np.uint8)
        if arr.shape != target_shape:
            raise RuntimeError(f"Mask size {arr.shape} != DICOM slice size {target_shape} for {fp}")
        masks.append((int(os.path.basename(fp).split('.')[0].split('_')[-1]), arr))  # 末尾の番号で並べる
    masks.sort(key=lambda x: x[0])
    vol = np.stack([m[1] for m in masks], axis=-1)  # HxW x Z
    return vol

def save_as_nifti(mask_vol, dicom_dir, out_nifti, spacing):
    # NIfTI 書き出し（SimpleITK）
    img = sitk.GetImageFromArray(mask_vol.astype(np.uint8).transpose(2,0,1))  # Z,Y,X
    img.SetSpacing((spacing[0], spacing[1], spacing[2]))
    sitk.WriteImage(img, out_nifti)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_dir", required=True)
    ap.add_argument("--harta_mask_dir", required=True, help="HARTAの2Dマスク出力ディレクトリ")
    ap.add_argument("--out_nifti", required=True)
    args = ap.parse_args()

    refs, wh, spacing, _ = read_dicom_series(args.dicom_dir)
    mask_vol = load_harta_slice_masks(args.harta_mask_dir, wh)
    save_as_nifti(mask_vol, args.dicom_dir, args.out_nifti, spacing)
    print(f"[OK] Saved NIfTI: {args.out_nifti}  spacing={spacing}")

if __name__ == "__main__":
    main()
```

実行例：

```bash
conda activate harta38
python ~/ctrate_ws/scripts/harta_masks_to_nifti.py \
  --dicom_dir       ~/ctrate_ws/data/dicom/valid_1_a_1 \
  --harta_mask_dir  ~/ctrate_ws/outputs/harta_raw/valid_1_a_1/EAT_masks \
  --out_nifti       ~/ctrate_ws/outputs/nifti/valid_1_a_1/eat_harta.nii.gz
```

> **補足**：`EAT_masks` のようなサブフォルダ名やファイル名規則は、**実際の HARTA 出力を 1例走らせて確認**し、上記スクリプトの `glob` と並び順ルールを合わせてください（この部分だけ HARTA 版の出力命名に依存します）。HARTA 自体は DICOM 入力・自動スクリプト同梱が公式 README に明記されています。([GitHub][1])

---

## 5) 2D 脂肪投影（fatmap）と DRR の作成（任意）

### fatmap（NumPy の平行投影近似）

```python
# fatmap_from_3d_mask.py
import nibabel as nib, numpy as np, imageio, sys
m = nib.load(sys.argv[1]).get_fdata()>0   # eat_harta.nii.gz
proj = m.sum(axis=1).astype(np.float32)   # Y軸和（PA/AP 近似）
proj = (proj/proj.max()*255).astype(np.uint8) if proj.max()>0 else proj.astype(np.uint8)
imageio.v2.imwrite(sys.argv[2], proj)
```

```bash
python fatmap_from_3d_mask.py ~/ctrate_ws/outputs/nifti/valid_1_a_1/eat_harta.nii.gz \
                              ~/ctrate_ws/outputs/nifti/valid_1_a_1/eat_fatmap.png
```

### DRR（Plastimatch）

既出どおり `plastimatch drr` で幾何（PA, 350×300mm, 700×600px 等）を固定設定にすれば、EAT fatmap とペアで供給できます。Plastimatch の convert/drr 使い方は多くの実例があります。([MONAI][3])

---

## 6) すべて自動で回す一括スクリプト（雛形）

**`scripts/pipeline_harta_all.sh`**：

```bash
#!/usr/bin/env bash
set -euo pipefail

# 事前に: conda activate harta38
NIFTI_CASES=( $(find ~/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed -name "*.nii.gz" | head -n 3) )

for NII in "${NIFTI_CASES[@]}"; do
  CASE_ID=$(basename "$NII" .nii.gz)
  DICOM_DIR=~/ctrate_ws/data/dicom/${CASE_ID}
  RAW_OUT=~/ctrate_ws/outputs/harta_raw/${CASE_ID}
  NIFTI_OUT_DIR=~/ctrate_ws/outputs/nifti/${CASE_ID}
  mkdir -p "$NIFTI_OUT_DIR"

  # 1) NIfTI→DICOM
  bash ~/ctrate_ws/scripts/nifti_to_dicom.sh "$NII" "$DICOM_DIR"

  # 2) HARTA（自動）
  python ~/ctrate_ws/scripts/run_harta_auto.py \
    --harta_repo ~/ctrate_ws/tools/HARTA \
    --dicom_root "$DICOM_DIR/.." \
    --out_root  ~/ctrate_ws/outputs/harta_raw

  # 3) HARTA出力→3D NIfTI
  #    ※ EATマスク格納場所は実行後のRAW_OUTを確認したうえで指定
  python ~/ctrate_ws/scripts/harta_masks_to_nifti.py \
    --dicom_dir "$DICOM_DIR" \
    --harta_mask_dir "$RAW_OUT/EAT_masks" \
    --out_nifti "$NIFTI_OUT_DIR/eat_harta.nii.gz"

  # 4) fatmap
  python ~/ctrate_ws/scripts/fatmap_from_3d_mask.py \
    "$NIFTI_OUT_DIR/eat_harta.nii.gz" \
    "$NIFTI_OUT_DIR/eat_fatmap.png"
done
```

---

## 7) 実務のコツ・注意点

1. **入力は DICOM 限定**
   HARTA 本体が DICOM 前提。NIfTI を使うプロジェクトでも **前段で DICOM に戻して**与えます。([GitHub][1])

2. **非造影での性能**
   README に “optimized for non‑contrast datasets” と明記。造影相では精度が落ちる可能性があるため、CT‑RATE v2 でも **非造影例を優先**。([GitHub][1])

3. **Python 3.8.3 縛り**
   環境は `conda create -n harta38 python=3.8.3` で分離。pip は HARTA 同梱の `requirements.txt` に従う。([GitHub][1])

4. **自動・半自動スクリプト**
   HARTA の `_automatic_.py / _semiautomatic_.py` を活用すると、GUI なしでバッチ運用がしやすい（この統合手順はそれを前提）。([GitHub][2])

5. **NIfTI 書き出し**
   本回答の `harta_masks_to_nifti.py` は **元 DICOM の幾何を尊重**（PixelSpacing/SliceThickness）。出力座標系の一貫性を確保。

6. **ライセンス**
   HARTA は **CC‑BY‑NC‑4.0**。**非商用** 研究限定で利用可能。商用利用想定なら別の手段（BOA や TotalSegmentator + 自前スクリプト等）も検討。([GitHub][1])

---

## 8) 代替・拡張

* **NIfTI→DICOM の別経路**：SimpleITK などで自前実装も可能ですが、**Plastimatch の `convert`** が最短で堅牢。([MONAI][3])
* **HARTA の GUI 確認**：もし出力の構造や名称を確認したい場合は `python3 harta.py` を起動して 1 症例で挙動を把握（README 手順）。([GitHub][1])
* **EAT 以外**：HARTA は EAT を主眼。PAT（心膜外脂肪）が必要なら、EAT NIfTI と TotalSegmentator（heartchambers）などの心臓マスクから**距離ベースで外方へ拡張**する従来手法を組み合わせる（既にお持ちのスクリプトで対応可能）。

---

### まとめ

* **やることは 4 つ**：①NIfTI→DICOM、②HARTA 自動推論、③2D出力→3D NIfTI、④（任意）fatmap/DRR 連携。
* 公式 README に沿って **Python 3.8.3 / DICOM前提** を守れば、**学習なし**で EAT を自動抽出できます。([GitHub][1])
* NIfTI へ戻す処理は本回答のスクリプトで一気通貫にできます。
* 変換や幾何の根拠は Plastimatch／HARTA の公開情報に準拠しています。([MONAI][3], [GitHub][1])

必要なら、上のスクリプトを **1 本の makefile / bash にまとめた“押すだけ”版** も作れます。実行後に、`~/ctrate_ws/outputs/harta_raw/<case_id>` の中身（EATマスクのフォルダ名・拡張子）を一度お知らせいただければ、`harta_masks_to_nifti.py` の `glob` をピンポイントで合わせた版を返します。

[1]: https://github.com/aforebelo/HARTA/tree/main "GitHub - aforebelo/HARTA: Epicardial Fat Segmentation and Quantification Software"
[2]: https://github.com/aforebelo/HARTA "GitHub - aforebelo/HARTA: Epicardial Fat Segmentation and Quantification Software"
[3]: https://docs.monai.io/projects/label/en/latest/quickstart.html?utm_source=chatgpt.com "Quickstart — MONAI Label 0.8.5 Documentation"
