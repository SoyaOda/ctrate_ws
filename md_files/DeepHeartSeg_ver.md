了解。**DeepHeartSeg を “心膜（pericardial sac）セグメンテーションの取得器”として組み込み、EAT を自動抽出する新スクリプト**の設計と実装手順をまとめます。下の手順は、これまでの TotalSegmentator ベースの実装と互換にしてあり、DeepHeartSeg が失敗したときのフォールバック（従来のシェル法）も備えています。

---

## まず把握：DeepHeartSeg で何ができる？

* Harvard AIM の **DeepHeartSeg** は、CT の**心臓領域の自動セグメンテーション**のためのオープンソース実装で、**学習済みモデル（weights）が同梱**されています。README に、**Python 2.7 系で動作**し、`src/` 以下に推論コード、`data/` 以下に学習済みモデルがある旨が記載されています。
* 関連論文（npj Digital Medicine 2021）でも、**ボリュメトリック心臓セグメンテーションを自動化**するツールとして位置づけられています（オープン実装の参照先として GitHub が挙げられています）。

> ※ DeepHeartSeg の出力ラベルは学習設定で多少差があるため、本提案では **「pericardium / pericardial\_sac / heart\_sac / EAT など心膜や心膜内領域を表すファイルが出る場合はそれを採用、出ない場合は heart（心臓全体）を核に“心膜内領域”を再構成**」という二段構えにしています。

---

## 全体像（アルゴリズム）

1. **DeepHeartSeg を tools/ に clone**し、**Docker で隔離実行**（Python 2.7/TensorFlow1 系の依存地獄を避けるため）。
2. DeepHeartSeg で **心膜の閉領域 or 心臓全体を推論**。

   * `pericardium/pericardial_sac` など**心膜境界**が直接出れば、それを**閉曲面に補完**して「**心膜内領域**」を得る。
   * 出ない場合は、\*\*heart（心臓マスク）＋小さな外側帯域（1–2 mm）\*\*を境界とし、**肺/肋骨/皮下/腹部臓器をドメイン外**にした上で、**内外領域を判別**して「心膜内領域」を再構成。
3. TotalSegmentator の **`heart_myocardium`** を除外マスクに使用。
4. **EAT = 心膜内領域 ∧ HU（脂肪） ∧ ¬心筋**

   * HU のデフォルトは**非造影 CT で広く使われる脂肪域を想定（-190〜-30 HU を既定値としてパラメータ化）**。一般に脂肪組織は負の HU（例：-120〜-90 付近）に分布することは周知です。値は撮影条件で変動するため**必ず引数で変更可能**にしています。
5. 品質ゲート

   * **禁制構造（肺・皮下・骨近傍・肝脾胃腎・気道・遠位大血管）**に触れた成分は**丸ごと除外**。
   * **心筋から 2–3 mm 以内に接している**脂肪成分のみ採用（epicardial “接触性”を担保）。
   * 体積・HU の sanity check を JSON ログに保存。

---

## セットアップ

```bash
# 1) DeepHeartSeg を tools/ に clone
cd tools
git clone https://github.com/AIM-Harvard/DeepHeartSeg.git

# 2) （推奨）Docker イメージを作成（Python2.7 + TF1 系）
cd DeepHeartSeg
cat > docker/Dockerfile <<'EOF'
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
RUN apt-get update && apt-get install -y python2.7 python2.7-dev python-pip git && rm -rf /var/lib/apt/lists/*
RUN pip2 install --upgrade pip==20.3.4 && pip2 install numpy scipy SimpleITK nibabel scikit-image h5py==2.10.0 tensorflow-gpu==1.14.0
WORKDIR /opt/DeepHeartSeg
COPY . /opt/DeepHeartSeg
ENV PYTHONIOENCODING=UTF-8
EOF
docker build -t deepheartseg:tf1 -f docker/Dockerfile .

# 3) DeepHeartSeg 推論ラッパー（NIfTI 入出力）スクリプトを用意
cat > run_deepheartseg.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
# Usage: ./run_deepheartseg.sh /path/to/ct.nii.gz /out/dir
CT=$1
OUT=$2
mkdir -p "$OUT"
# ここは DeepHeartSeg の src 側の実行エントリに合わせて調整してください
# 例: python2 src/predict.py --task heart --in $CT --out $OUT
docker run --rm --gpus all -v "$PWD":/work -v "$(dirname "$CT")":/in -v "$OUT":/out \
  deepheartseg:tf1 bash -lc "
    cd /opt/DeepHeartSeg && \
    python2 -c 'print(\"Replace with the actual DeepHeartSeg inference call that outputs NIfTI masks under /out\")'
  "
EOF
chmod +x run_deepheartseg.sh
```

> **備考**：DeepHeartSeg の README には **Python 2.7 必須、`src/` にコード、`data/` に学習済みモデル**と明記されています。実際の CLI 名はバージョンにより異なるため、`run_deepheartseg.sh` の **1 行（`python2 ...`）だけ**は、手元の `src/README.md` に従って置き換えてください。

---

## 新スクリプト：`eat_from_pericardium.py`

* 役割：

  1. DeepHeartSeg の推論を（必要に応じて）実行して**心膜関連マスク**を取得
  2. 「**心膜内領域**」を確実に閉領域で再構成
  3. **EAT** を抽出して NIfTI と JSON を保存

* 入力：

  * `--ct-nifti`: CT（NIfTI）
  * `--totalseg-dir`: TotalSegmentator 出力（`heart_myocardium.nii.gz` を参照）
  * `--deepheartseg-out`: DeepHeartSeg の推論出力ディレクトリ（ここに pericardium/heart 等の NIfTI が出ている前提）
  * `--run-deepheartseg`: 指定時は `tools/DeepHeartSeg/run_deepheartseg.sh` を呼び出して推論も実行
  * HU 閾値、接触距離、禁制構造除外の on/off など

> **どのファイル名を読むか**は自動推測（`pericardium*`, `*pericardial*`, `*heart*`, `*EAT*` を探索）。心膜“境界”しかないときは、**心筋マスクをシードにして region fill**し、閉領域化します。

---

### コード

> 下は **Python 3** で動くオーケストレーター（実際の DL は Docker 内の Python 2.7 側）。そのまま配置して使えます。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EAT segmentation with DeepHeartSeg pericardial segmentation (wrapper).
- Runs DeepHeartSeg (optional) and reads its outputs.
- Reconstructs a watertight 'inside-pericardium' volume.
- Computes EAT = (inside-pericardium) ∧ (fat HU) ∧ (not myocardium),
  with connectivity to myocardium (optional) and strong exclusion gates.
"""

import os, sys, re, json, subprocess
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import (
    binary_dilation, binary_erosion, binary_fill_holes, label,
    distance_transform_edt, generate_binary_structure
)

def load_nii(path):
    img = nib.load(str(path))
    return img, img.get_fdata(), img.header.get_zooms()[:3]

def save_mask_like(ref_img, mask, out_path):
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine), str(out_path))

def find_first_matching(base_dir: Path, patterns):
    for pat in patterns:
        for p in sorted(base_dir.glob(pat)):
            return p
    return None

def mm_to_iters(mm, spacing):
    return max(1, int(np.ceil(mm / min(spacing))))

def run_deepheartseg(ct_path: Path, out_dir: Path, repo_root: Path):
    script = repo_root / "run_deepheartseg.sh"
    if not script.exists():
        raise FileNotFoundError(f"DeepHeartSeg runner not found: {script}")
    cmd = [str(script), str(ct_path), str(out_dir)]
    subprocess.check_call(cmd)

def reconstruct_inside_from_pericardium(peri_mask, heart_seed, spacing):
    """
    peri_mask: 心膜（境界 or 薄い帯）or 心膜内（filled）のどちらでも OK
    heart_seed: 既知の“内側”シード（心筋＋心腔など）
    """
    peri = peri_mask.astype(bool)
    # “線状/薄膜”かどうか推定：体積比と厚みでざっくり判定
    peri_ratio = peri.mean()
    is_thin = peri_ratio < 0.02  # かなり薄いなら境界の可能性が高い

    if is_thin:
        # 薄い境界→ 少し太らせて閉曲面化
        it = mm_to_iters(1.0, spacing)
        peri_thick = binary_dilation(peri, iterations=it)
        # Heart seed から域内を決定：境界外には漏れないように
        inside = binary_fill_holes(peri_thick)  # 万一の穴塞ぎ
        # “心筋＋心腔”シードが必ず inside に入ることを保証
        inside = inside | heart_seed
        # 外の連結成分は除去
        # （境界が完全でない場合の保険：心筋シードと同一 CC のみ残す）
        cc, n = label(inside)
        if n > 1:
            seed_label = np.argmax(np.bincount(cc[heart_seed].ravel().astype(int))[1:]) + 1
            inside = (cc == seed_label)
    else:
        # 既に filled とみなす
        inside = binary_fill_holes(peri)

    return inside

def main():
    import argparse
    ap = argparse.ArgumentParser(description="EAT via DeepHeartSeg pericardial segmentation")
    ap.add_argument("ct_nifti")
    ap.add_argument("totalseg_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--deepheartseg-root", default=str(Path("tools/DeepHeartSeg").absolute()))
    ap.add_argument("--deepheartseg-out", default=None, help="where DeepHeartSeg drops its NIfTI masks")
    ap.add_argument("--run-deepheartseg", action="store_true", help="actually run DeepHeartSeg")
    ap.add_argument("--hu-min", type=float, default=-190.0)
    ap.add_argument("--hu-max", type=float, default=-30.0)
    ap.add_argument("--contact-mm", type=float, default=3.0, help="EAT must contact myocardium within this distance")
    ap.add_argument("--strict-exclusion", action="store_true", help="remove any component touching forbidden structures")
    args = ap.parse_args()

    ct_path = Path(args.ct_nifti).absolute()
    out_dir = Path(args.output_dir).absolute(); out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.deepheartseg_root).absolute()
    dh_out = Path(args.deepheartseg_out).absolute() if args.deepheartseg_out else (out_dir / "deepheartseg")

    if args.run_deepheartseg:
        dh_out.mkdir(parents=True, exist_ok=True)
        run_deepheartseg(ct_path, dh_out, repo_root)

    # --- load CT ---
    ct_img, ct, spacing = load_nii(ct_path)

    # --- load myocardium (TotalSegmentator) ---
    ts = Path(args.totalseg_dir)
    myo_path = ts / "heart_myocardium.nii.gz"
    if not myo_path.exists():
        # フォールバック
        myo_path = ts / "heart.nii.gz"
    if not myo_path.exists():
        raise FileNotFoundError("TotalSegmentator myocardium/heart mask not found")

    _, myo, _ = load_nii(myo_path)
    myo = myo > 0

    # --- DeepHeartSeg outputs: pericardium or heart/EAT ---
    # 優先：pericardium* / *pericardial* / *heart_sac*
    cand_peri = find_first_matching(dh_out, ["*pericardium*.nii*", "*pericardial*.nii*", "*heart_sac*.nii*"])
    cand_eat  = find_first_matching(dh_out, ["*EAT*.nii*","*epicardial*fat*.nii*"])
    cand_heart= find_first_matching(dh_out, ["heart.nii*", "*heart*mask*.nii*"])

    if cand_peri is None and cand_heart is None and cand_eat is None:
        raise FileNotFoundError("No usable DeepHeartSeg outputs found in: %s" % dh_out)

    if cand_peri is not None:
        _, peri_mask, _ = load_nii(cand_peri); peri_mask = peri_mask > 0
        # “心臓コア”シード：心筋＋（あれば）心腔
        heart_seed = myo.copy()
        # 心膜内再構成
        inside_peri = reconstruct_inside_from_pericardium(peri_mask, heart_seed, spacing)
    else:
        # 心膜が無い → 心臓マスクから小帯域を外側に付与し閉領域化
        if cand_heart is None:
            _, eat_dh, _ = load_nii(cand_eat)
            # EAT が直接ある場合は、内包領域として “心筋 + EAT + 心腔（仮に myo 膨張）”を使う
            heart_core = binary_dilation(myo, iterations=mm_to_iters(2.0, spacing))
        else:
            _, heart_core, _ = load_nii(cand_heart)
            heart_core = heart_core > 0
        band = binary_dilation(heart_core, iterations=mm_to_iters(2.0, spacing))
        inside_peri = binary_fill_holes(band)

    # --- fat HU ---
    hu_fat = (ct >= args.hu_min) & (ct <= args.hu_max)

    # --- EAT base ---
    eat_base = inside_peri & hu_fat & (~myo)

    # --- 接触制約：心筋から contact-mm 以内
    if args.contact_mm > 0:
        contact = binary_dilation(myo, iterations=mm_to_iters(args.contact_mm, spacing))
        eat_base &= contact

    # --- 禁制構造の除外（漏れ対策） ---
    # 利用可能であれば TotalSegmentator のマスクで除外（肺・皮下・骨・肝など）
    forbid = np.zeros_like(eat_base, dtype=bool)
    def add_mask(name, dil_mm=0.0):
        p = ts / f"{name}.nii.gz"
        if p.exists():
            _, m, _ = load_nii(p)
            m = m > 0
            if dil_mm > 0:
                m = binary_dilation(m, iterations=mm_to_iters(dil_mm, spacing))
            return m
        return np.zeros_like(eat_base, dtype=bool)

    # 胸腔外・肺・気道・骨近傍・肝・脾・胃・腎を広めに除外
    forbid |= add_mask("lung_upper_lobe_left")
    forbid |= add_mask("lung_lower_lobe_left")
    forbid |= add_mask("lung_upper_lobe_right")
    forbid |= add_mask("lung_middle_lobe_right")
    forbid |= add_mask("lung_lower_lobe_right")
    forbid |= add_mask("trachea", 2.0) | add_mask("main_bronchus_left", 2.0) | add_mask("main_bronchus_right", 2.0)
    # 骨近傍
    forbid |= add_mask("sternum", 3.0)
    for v in ["T%d"%i for i in range(1,13)] + ["L%d"%i for i in range(1,6)]:
        forbid |= add_mask(f"vertebrae_{v}", 2.0)
    # 腹部臓器
    for org in ["liver","spleen","stomach","kidney_left","kidney_right","gallbladder","pancreas"]:
        forbid |= add_mask(org, 2.0)

    eat = eat_base & (~forbid)

    # 触れた成分を丸ごと落とす（厳格モード）
    if args.strict_exclusion:
        cc, n = label(eat)
        if n > 0:
            kept = np.zeros_like(eat, bool)
            for k in range(1, n+1):
                comp = (cc == k)
                if not np.any(comp & forbid):
                    kept |= comp
            eat = kept

    # --- 保存とサマリ ---
    out_mask = out_dir / "eat_mask.nii.gz"
    out_inside = out_dir / "pericardial_inside.nii.gz"
    save_mask_like(ct_img, eat, out_mask)
    save_mask_like(ct_img, inside_peri, out_inside)

    vox_vol = np.prod(spacing)
    eat_vox = int(eat.sum()); eat_ml = eat_vox * vox_vol / 1000.0
    vals = ct[eat]; mu = float(np.mean(vals)) if vals.size else None; sd = float(np.std(vals)) if vals.size else None

    with open(out_dir / "eat_summary.json", "w") as f:
        json.dump({
            "eat_volume_ml": round(eat_ml, 2),
            "eat_voxels": eat_vox,
            "mean_hu": mu,
            "std_hu": sd,
            "hu_range": [args.hu_min, args.hu_max],
            "contact_mm": args.contact_mm,
            "strict_exclusion": bool(args.strict_exclusion)
        }, f, indent=2)

    print(f"[DONE] EAT {eat_ml:.2f} ml  (voxels={eat_vox})  meanHU={mu}±{sd}")

if __name__ == "__main__":
    main()
```

---

## 使い方

```bash
# 1) DeepHeartSeg を事前に実行した場合（出力が deepheartseg_out にある）
python3 scripts/eat_from_pericardium.py \
  /path/to/ct.nii.gz \
  /path/to/totalseg_output \
  /path/to/out_dir \
  --deepheartseg-out /path/to/deepheartseg_out \
  --hu-min -190 --hu-max -30 --contact-mm 3 --strict-exclusion

# 2) スクリプトから DeepHeartSeg も一気に実行（Docker 推奨）
python3 scripts/eat_from_pericardium.py \
  /path/to/ct.nii.gz \
  /path/to/totalseg_output \
  /path/to/out_dir \
  --run-deepheartseg \
  --deepheartseg-root tools/DeepHeartSeg \
  --hu-min -190 --hu-max -30 --contact-mm 3 --strict-exclusion
```

出力：

* `eat_mask.nii.gz`（EAT）
* `pericardial_inside.nii.gz`（再構成した**心膜内領域**）
* `eat_summary.json`（体積/密度のサマリ）

---

## 現場でハマりやすい点と対処

1. **DeepHeartSeg の CLI 名が不明**
   → `tools/DeepHeartSeg/src/README.md` に合わせて、`run_deepheartseg.sh` の `python2 ...` 行を置き換えてください。リポジトリの README には **Python 2.7 / pre-trained weights 同梱 / src にコード**の記載が確認できます。
2. **心膜が“線”として出るケース**
   → スクリプトは自動で\*\*閉曲面化（軽い膨張＋hole fill）\*\*し、**心筋シードと同一連結成分のみ採用**します。
3. **漏れ（前胸壁・肋骨・肝臓など）**
   → **禁制構造に触れた成分を丸ごと削除**する `--strict-exclusion` を ON。さらに必要なら TotalSegmentator マスクの膨張距離（`add_mask(..., dil_mm)`）を広げてください。
4. **HU 閾値**
   → 既定は -190〜-30 HU（非造影 CT でよく使われる範囲）としていますが、撮影条件で脂肪 HU は変動し得ます（脂肪は一般に負の HU に分布し、例として -120〜-90 付近が挙げられます）。現場のプロトコルに合わせて必ずパラメータで調整してください。
5. **EAT の“接触性”**
   → `--contact-mm` を 2–3 mm に設定すると、**心筋に接する脂肪**のみを EAT として残し、心膜腔内の孤立脂肪や液体の混入を抑制できます。

---

## まとめ

* **DeepHeartSeg を tools/ に導入**し、**心膜または心膜内領域を直接使う**ことで、従来の「外側 EDT シェル」より **漏れの少ない EAT** を安定して抽出できます（Docker 推奨）。
* 新スクリプト `eat_from_pericardium.py` は、

  1. DeepHeartSeg の結果を取り込み（必要なら起動）、
  2. **心膜内領域**を閉領域として再構成し、
  3. **HU 閾値＋心筋接触＋禁制構造除外**で EAT を確定します。
* 心膜の出力ラベルや CLI 名は環境で異なることがあるため、**`run_deepheartseg.sh` の 1 行**だけ現場の実行手順に合わせて差し替えてください。

必要なら、このスクリプトをあなたの既存パイプライン（HARTA/TotalSegmentator と同じ出力先・JSON 仕様）に合わせて最終調整します。
