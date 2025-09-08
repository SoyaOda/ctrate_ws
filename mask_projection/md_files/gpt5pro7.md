了解です。ここでは **Plan‑A（単色“実効”モデル：I/I₀→−log→厚み\[mm]）** を、**あいまいさゼロ**で実装できるように、根拠→要件→コード（差し替え可能な完成版）→QC まで**フルパッケージ**でまとめます。
キーポイントは **(1) 同一幾何で撮った flat‑field（I₀）** を使って **I/I₀ を正規化**し、**(2) −log(I/I₀)** を **線減弱の線積分**（“厚み”に比例）として取り出し、**(3) 体積一致で μ\_eff を自己較正**して **画素値＝厚み\[mm]** に落とすことです。これは X 線の **Beer–Lambert 則**（単色なら −log(I/I₀)=μ·L）と、**flat‑field 補正**（I\_norm=(I\_raw−dark)/(flat−dark)≃I/I₀）の標準手順に従っています。([arXiv][1])
DeepDRR は既定で**多色・多材質投影**（ビームハードニングで −log と厚みが厳密線形でなくなる）が使われますが、本 Plan‑A は **“μ\_eff による実効単色化”** で **体積一致**と**画素スケールの mm 化**を担保します。必要なら DeepDRR 論文の「素材＆スペクトルを考慮するレイトレーサ（material‑ and spectrum‑aware）」の説明も参照してください。([arXiv][1])

---

## 0) 仕様（要件整理）

* **幾何**：既存 DRR（CT／Mask）と**完全同一**（LPS 正規化／同じ MobileCArm／γ=90°、PA だけ水平反転は**最後に1回だけ**）。
* **flat‑field（I₀）**：**空気ボリューム（HU=−1000）**を**同一幾何**で投影して取得。I\_mask/I\_air を取り **−log**。
* **厚み画像 T\_mm**：

  $$
    G=-\log\!\left(\frac{I_{\text{mask}}}{I_{\text{air}}}\right),\quad 
    \Omega=\{G>\tau\},\quad
    \mu_{\text{eff}}=\frac{A\cdot\sum_{\Omega} G}{V_{3D}},\quad
    T_{\text{mm}}(x)=\begin{cases}G(x)/\mu_{\text{eff}} & x\in\Omega\\ 0 & \text{otherwise}\end{cases}
  $$

  ここで **A=p^2（ピクセル面積 mm²）**、**V\_{3D}** は 3D マスク体積（mm³）。この定義で常に **$\sum T\cdot A=V_{3D}$**（体積一致）。
* **多色による非線形**：DeepDRR はポリクロマティックなので −log と真の厚みの関係は理論上わずかに非線形（ビームハードニング）。ただし **μ\_eff による“実効単色化”**で**体積一致**かつ\*\*画素スケール（mm）\*\*は保証されます（小厚み領域ではほぼ線形）。([arXiv][1])
* **（任意）物理 μ の固定**：実効単色エネルギ E₀ を仮定し、**NIST XCOM**の μ(E₀) を使えば **T=G/μ(E₀)** で**画素単位で厳密線形**（理想単色）。本書き換えは μ\_eff を用いる実装（運用簡便・堅牢）を主とし、**付録に μ 物理固定の差分**を付けます。

---

## 1) 完成版コード（差し替え用・最小侵襲）

> 既存 `mask_deepdrr_projection.py` に**そのまま貼り替え／上書き**できます。
> 変更点は **flat‑field 取得の厳密化／ピクセルピッチの自動取得／シルエットの頑健化／体積 QC の常設** です。

```python
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Plan-A: 単色“実効”モデルで画素値=厚み[mm]
- I_air (flat-field) を同一幾何で取得
- G = -log( I_mask / I_air )
- μ_eff を体積一致で自己較正
- T_mm = G / μ_eff（Ω=シルエット内、外は0）
- ∑T_mm * p^2 = V_3D を常に満たす

根拠:
- Beer–Lambert (mono): -log(I/I0) = μ * L  → 厚みに線形。 DeepDRR 論文背景等参照
- Flat-field 正規化: I_norm = (I_raw - dark) / (flat - dark) ≈ I/I0
"""
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from nibabel import orientations as nio
from nibabel.affines import voxel_sizes
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector

# ---------- 0) 共通: LPSに厳密変換 ----------
def load_volume_LPS(nifti: Path) -> Volume:
    img = nib.load(str(nifti))
    in_ornt = nio.io_orientation(img.affine)
    out_ornt = nio.axcodes2ornt(("L","P","S"))
    to_lps = nio.ornt_transform(in_ornt, out_ornt)
    data_lps = nio.apply_orientation(img.get_fdata().astype(np.float32), to_lps)
    aff_lps = img.affine @ nio.inv_ornt_aff(to_lps, img.shape)
    spacing = voxel_sizes(aff_lps)[:3]
    shp = np.array(data_lps.shape, dtype=float)
    origin = (-0.5 * (shp - 1.0) * spacing).tolist()
    return Volume.from_hu(
        hu_values=data_lps, origin=origin, spacing=spacing.tolist(),
        anatomical_coordinate_system="LPS",
    )

def load_mask_volume_LPS(mask_nifti: Path) -> Volume:
    img_mask = nib.load(str(mask_nifti))
    in_ornt = nio.io_orientation(img_mask.affine)
    out_ornt = nio.axcodes2ornt(("L","P","S"))
    to_lps = nio.ornt_transform(in_ornt, out_ornt)
    mask_lps = nio.apply_orientation(img_mask.get_fdata().astype(np.float32), to_lps)
    # 疑似HU（背景=-1000, 内部=+3000）→ 可視化性向上。Plan-A自体はここに依存しない
    hu = np.full(mask_lps.shape, -1000.0, dtype=np.float32)
    hu[mask_lps > 0.5] = 3000.0
    aff_mask_lps = img_mask.affine @ nio.inv_ornt_aff(to_lps, img_mask.shape)
    spacing = voxel_sizes(aff_mask_lps)[:3]
    shp = np.array(hu.shape, dtype=float)
    origin = (-0.5 * (shp - 1.0) * spacing).tolist()
    return Volume.from_hu(
        hu_values=hu, origin=origin, spacing=list(spacing),
        anatomical_coordinate_system="LPS",
    )

# ---------- 1) 幾何: C-arm ----------
def make_carm(isocenter, sid=1800.0, oid=80.0, px=0.14, plate_mm=(350.0, 430.0)):
    sad = sid - oid
    W = int(round(plate_mm[0] / px)) & ~1
    H = int(round(plate_mm[1] / px)) & ~1
    return MobileCArm(
        isocenter=isocenter, rotate_camera_left=True,
        source_to_detector_distance=sid,
        source_to_isocenter_vertical_distance=sad,
        pixel_size=px,
        sensor_width=W, sensor_height=H,
        min_alpha=-720, max_alpha=720, min_beta=-720, max_beta=720,
    )

# ---------- 2) 同一幾何のエナジー画像（生） ----------
def render_energy(vol: Volume, carm: MobileCArm, alpha=0, beta=0, gamma=90, flip_pa=True):
    with Projector(vol, carm=carm) as projector:
        carm.move_to(alpha=alpha, beta=beta, gamma=gamma, degrees=True)
        _ = projector()           # warm-up
        img = projector().astype(np.float32)  # [0..1] transmission-like
    if flip_pa:
        img = img[:, ::-1].copy() # radiographic view: 患者左=画像右
    return img

# ---------- 3) flat-field (I_air) 生成 & キャッシュ ----------
def make_air_volume_like(vol: Volume) -> Volume:
    air = np.full(vol.data.shape, -1000.0, dtype=np.float32)
    return Volume.from_hu(
        hu_values=air, origin=vol.origin, spacing=vol.spacing,
        anatomical_coordinate_system=vol.anatomical_coordinate_system,
    )

def flat_cache_key(alpha,beta,gamma,carm):
    W,H = carm.sensor_width, carm.sensor_height
    px   = float(getattr(carm, "pixel_size", 0.14))
    return f"a{alpha}_b{beta}_g{gamma}_W{W}_H{H}_px{px:.5f}"

def get_flat_field(mask_vol: Volume, carm: MobileCArm, view, outdir: Path, flip_pa=True):
    alpha,beta,gamma = view
    cache_path = outdir / f"flat_{flat_cache_key(alpha,beta,gamma,carm)}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    air_vol = make_air_volume_like(mask_vol)
    air_vol.place_center(carm.isocenter_in_world)
    I_air = render_energy(air_vol, carm, alpha, beta, gamma, flip_pa=flip_pa)
    np.save(cache_path, I_air)
    return I_air

# ---------- 4) Beer–Lambert 逆変換 + μ_eff 自己較正 ----------
def thickness_from_energy(I_mask: np.ndarray, I_air: np.ndarray, V3D_mm3: float,
                          px_mm: float, support_tau: float = 1e-4):
    # 形状確認
    assert I_mask.shape == I_air.shape, "I_mask/I_air shape mismatch"
    eps = 1e-6
    ratio = np.clip(I_mask / np.clip(I_air, eps, 1.0), eps, 1.0)
    G = -np.log(ratio)  # Beer–Lambert (mono) の線積分に相当
    # シルエット（Ω）: 固定閾値 + ノイズ頑健化
    tau = max(support_tau, np.percentile(G, 0.05))
    Omega = G > tau
    A = float(px_mm * px_mm)       # ピクセル面積
    sumG = float(G[Omega].sum())
    assert V3D_mm3 > 0 and sumG > 0, "empty projection or zero 3D volume"
    mu_eff = (A * sumG) / V3D_mm3  # 体積一致（∑(G/μ)*A = V3D）
    T = np.zeros_like(G, dtype=np.float32)
    T[Omega] = G[Omega] / mu_eff   # 画素値 = 厚み[mm]
    # QC: 体積一致
    V2D = float(T.sum()) * A
    vol_err = abs(V2D - V3D_mm3) / V3D_mm3 * 100.0
    return T, mu_eff, Omega, vol_err

# ---------- 5) ユーティリティ ----------
def mask_volume_mm3(mask_nifti: Path) -> float:
    img = nib.load(str(mask_nifti))
    data = img.get_fdata()
    dx,dy,dz = img.header.get_zooms()[:3]
    return float((data > 0.5).sum()) * float(dx*dy*dz)

def save_u16(path: Path, x01: np.ndarray):
    x = np.clip(x01, 0, 1)
    imageio.imwrite(path, (x*65535).astype(np.uint16))

# ---------- 6) メイン ----------
def main():
    # 入力
    ct_nifti   = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    mask_nifti = Path("/workspace/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz")
    outdir = Path("/workspace/mask_projection/outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    GAMMA = 90.0
    view  = (0.0, 0.0, GAMMA)  # PA (縦)

    # 1) 読み込み & 幾何
    ct_vol   = load_volume_LPS(ct_nifti)
    mask_vol = load_mask_volume_LPS(mask_nifti)
    carm = make_carm(ct_vol.center_in_world)
    ct_vol.place_center(carm.isocenter_in_world)
    mask_vol.place_center(carm.isocenter_in_world)

    # 2) 画像生成（I_mask, I_air）
    I_mask = render_energy(mask_vol, carm, *view, flip_pa=True)  # radiographic view 準拠
    I_air  = get_flat_field(mask_vol, carm, view, outdir, flip_pa=True)

    # 3) 厚み推定
    px_mm = float(getattr(carm, "pixel_size", 0.14))
    V3D   = mask_volume_mm3(mask_nifti)
    T_mm, mu_eff, Omega, vol_err = thickness_from_energy(I_mask, I_air, V3D, px_mm, support_tau=1e-4)

    # 4) 可視化保存
    if T_mm.max() > 0:
        T_disp = np.clip(T_mm / np.percentile(T_mm[T_mm>0], 99.9), 0, 1)
    else:
        T_disp = T_mm
    save_u16(outdir/"PA_mask_thickness_mm.png", T_disp)
    np.save(outdir/"PA_mask_thickness_mm.npy", T_mm)

    # 5) 参考: マスクDRR（表示用）
    with Projector(mask_vol, carm=carm) as pj:
        carm.move_to(*view, degrees=True)
        _ = pj()
        E = pj().astype(np.float32)
    E = E[:, ::-1].copy()
    # 表示用の正規化（学習/計測には使わない）
    lo,hi = np.percentile(E, (1.0, 99.5))
    E_disp = np.clip((E-lo)/max(hi-lo,1e-6), 0, 1)
    save_u16(outdir/"PA_mask_deepdrr.png", E_disp)

    # 6) QC レポート
    info = {
        "pixel_pitch_mm": px_mm,
        "mu_eff_1_per_mm": float(mu_eff),
        "volume_3D_mm3": float(V3D),
        "volume_2D_mm3": float(T_mm.sum() * px_mm * px_mm),
        "volume_error_percent": float(vol_err),
        "support_pixels": int(Omega.sum()),
        "image_shape": list(T_mm.shape),
    }
    with open(outdir/"thickness_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("[Plan-A] 完了:", json.dumps(info, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

### 重要ポイント（コード内で既に反映）

* **I\_mask と I\_air は**必ず**同じ幾何・同じ PA 反転規則**で取得（`render_energy(..., flip_pa=True)`）。**I/I₀** は **flat‑field 補正**の基本式です。
* **−log(I/I₀)** で **線減弱の線積分**が出ます（単色モデルの基本式）— DeepDRR もこの物理式に基づく前向き投影を内部で扱っています。([arXiv][1])
* **μ\_eff** を $\mu_{\text{eff}}=A\sum G/V_{3D}$ で決めることで **$\sum T\cdot A = V_{3D}$** を保証（体積一致）。
* **Ω（シルエット）** は `max(1e-4, p0.05)` で**薄層ノイズに頑健**（dark/flat が理想の DeepDRR でも数値丸め対策）。
* **学習・推定には `PA_mask_deepdrr.png` は使わない**（表示用にパーセンタイル正規化されており、厚みには無関係）。**厚みは `PA_mask_thickness_mm.npy`** を使う。
* **体積 QC** は JSON に保存（再現性・監視可能）。

---

## 2) QC（必須チェック）

1. **体積一致**（常時ログ化）
   `thickness_info.json` 内の `volume_error_percent` が **≪1–2%** で安定していること（あなたの環境では **0%** を既に達成）。
2. **単調性・非負**
   `T_mm` は **非負**、`G` が大きい画素ほど厚めになる（単調）。
3. **くさび/階段ファントム**（任意）
   既知厚みで −log と厚みの直線性を確認（DeepDRR の多色では微小な曲がり＝ビームハードニングが見える）。必要なら Plan‑B（ポリ補正）へ。([arXiv][1])

---

## 3) よくある落とし穴（回避策はコード済み）

* **flat‑field の幾何不一致**：角度・センサ寸法・ピクセルピッチが**1画素でもズレる**と I/I₀ が破綻 → 本コードでは**幾何キーでキャッシュ**、毎ビュー同一幾何で再生成。
* **PA 反転の重複**：**最後に一回**のみ（I\_mask と I\_air は**同じ側**に揃える）。
* **I\_air のゼロ割**：`eps` 付きの安全クリップ。
* **薄層の穴あき**：Ω を p0.05 で下支え。

---

## 4) 付録：物理 μ（mono）での厳密 mm 化（任意）

装置を **実効単色 E₀（例 60–70 keV）** とみなし、材質を\*\*1種類（脂肪/水等価）\*\*に固定できるなら、**NIST XCOM**の μ(E₀)（\[cm²/g] と密度 ρ\[g/cm³] から μ\[1/mm]）を用い、
**$T_{\text{mm}} = -\log(I/I_0) / \mu(E_0)$** とすれば **画素ごとに厳密線形**になります（散乱無視の理想化）。DeepDRR のスペクトルを単色化できる API がある版ならそれを使い、ない場合でも「実効 E₀」を設定して μ を固定する手はあります（ただし多色現実との差で体積一致が崩れる可能性あり）。

> **実運用の推奨**：まずは上記 **μ\_eff（体積一致）** を採用（堅牢＆簡便）。必要なら別ジョブで **Siddon 厚み**を教師に**多項式補正**（Plan‑B）を重ね、**画素単位の線形性**をさらに詰める、という二段構えが現実的です。DeepDRR が**多色・多材質**であることは論文にも明記されています。([arXiv][1])

---

### 参考（根拠）

* **DeepDRR 論文（arXiv page）**：解析投影は **material- & spectrum‑aware**（多材質・多色）で、Beer–Lambert を基礎にした前向き投影。ビームハードニングの存在を理解するうえで重要。([arXiv][1])
* **DeepDRR GitHub**：X 線スペクトルやエネルギ堆積に言及（多色性の根拠）。
* **Flat‑field 補正**：I\_norm=(I\_raw−dark)/(flat−dark) の標準式（X 線でも同様に用いられる基礎）。
* **NIST XCOM**：単色 μ の物理値（mono で厳密 mm 化する場合に参照）。

---

これで **位置整合を維持したまま**、**各画素を“厚み\[mm]”に直結**させる Plan‑A が完成です。上のコードを差し替えて動かせば、**`PA_mask_thickness_mm.npy` が教師**（総和＝体積）としてそのまま使えます。必要になれば、Plan‑B（多項式補正）も後段で渡します。

[1]: https://arxiv.org/abs/1803.08606 "[1803.08606] DeepDRR -- A Catalyst for Machine Learning in Fluoroscopy-guided Procedures"
