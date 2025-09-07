はい、**DeepDRRだけ**でマスクを2D投影すれば、CTのDRRと**ピクセル単位で**重なるようにできます。
鍵は「**CTと同じ幾何（C‑arm・座標系・向き・反転）で**、**同じ1本の Projector パス**を通す」ことです。DiffDRR を介さず、DeepDRR の `MobileCArm` ＆ `Projector` を **CT・マスクの両方にそのまま適用**します。DeepDRR の基本手順自体は公式の “Usage” にある通りで（`MobileCArm` 構築 → `orient_patient/center` → `Projector` で投影）、**同じ C‑arm と roll（γ=90°）・PA時の最終水平反転**を共有すれば、位置合わせは一致します。([deepdrr.readthedocs.io][1], [GitHub][2])

---

## 進め方（結論）

1. **CTと同じLPS正規化**でマスクNIfTIを読み込み（配列とaffineを同期させる）、
2. **CTと同じ `MobileCArm` インスタンス**を使い、
3. \*\*同じ `Projector`・同じ角度（PA: α=0, β=0, γ=90）\*\*でマスクを投影、
4. **PA画像は最終段で水平反転を1回だけ適用**（CTと同じ規則）、
5. 出力を二値/シルエット化（-log正規化後に閾値）してオーバーレイ。

> 重要: **同一 run の中で同じ `carm` を使う**（整数丸め・主点の微差を排除）。
> `ct.place_center(carm.isocenter_in_world)` と **同じ**ことを **mask Volume にも**行う。([deepdrr.readthedocs.io][1])

---

## サンプル実装（DeepDRRオンリーでマスク投影）

> あなたの CT DRR スクリプト（A案：PAのみ水平反転・`rotate_camera_left=True` 維持）に**追記**できる形です。
> 使う幾何・画素ピッチ・反転は **CTと完全一致**にしています。

```python
#!/usr/bin/env python3
"""
DeepDRRベースのマスク投影（CT DRRと完全整合）
- CTと同じLPS正規化・同じMobileCArm・同じProjector・同じ反転規則を共有
- 出力は -log 正規化後に二値化（シルエット） or 連続値（擬似厚み）
"""

from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel import orientations as nio
from nibabel.affines import voxel_sizes
import imageio.v2 as imageio

# GUI不要でmatplotlib使用（Docker対応）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector

# ---------- 共通: NIfTIをLPSに厳密変換（配列とaffine同期） ----------
def load_volume_LPS(nifti: Path) -> Volume:
    img = nib.load(str(nifti))
    in_ornt = nio.io_orientation(img.affine)
    out_ornt = nio.axcodes2ornt(("L","P","S"))
    to_lps = nio.ornt_transform(in_ornt, out_ornt)

    data_lps = nio.apply_orientation(img.get_fdata().astype(np.float32), to_lps)
    aff_lps  = img.affine @ nio.inv_ornt_aff(to_lps, img.shape)
    spacing  = voxel_sizes(aff_lps)[:3]

    # DeepDRRが扱いやすいように「体積中心=世界原点」
    shp    = np.array(data_lps.shape, dtype=float)
    origin = (-0.5 * (shp - 1.0) * spacing).tolist()

    vol = Volume.from_hu(
        hu_values=data_lps,   # HUベース：マスクは後で変換
        origin=origin,
        spacing=spacing.tolist(),
        anatomical_coordinate_system="LPS",
    )
    return vol

# ---------- CTと同じC-arm ----------
def make_carm(isocenter, sid=1800.0, oid=80.0, px=0.14, plate_mm=(350.0, 430.0)):
    sad = sid - oid
    W = int(round(plate_mm[0] / px)) & ~1
    H = int(round(plate_mm[1] / px)) & ~1
    return MobileCArm(
        isocenter=isocenter,
        rotate_camera_left=True,  # あなたのCT DRRと同じ前提を厳守
        source_to_detector_distance=sid,
        source_to_isocenter_vertical_distance=sad,
        pixel_size=px,
        sensor_width=W, sensor_height=H,
        min_alpha=-720, max_alpha=720, min_beta=-720, max_beta=720,
    )

# ---------- 共通: Project & 反転（PAだけ水平反転） ----------
def project_and_flip(vol: Volume, carm: MobileCArm, alpha=0., beta=0., gamma=90., do_pa_flip=True):
    with Projector(vol, carm=carm) as projector:
        carm.move_to(alpha=alpha, beta=beta, gamma=gamma, degrees=True)
        _ = projector()              # warm-up
        img = projector()            # 本番
    img = img.astype(np.float32)
    if do_pa_flip:
        img = img[:, ::-1].copy()    # radiographic view: 患者左=画像右
    return img  # Energy画像（線形正規化は未適用）

# ---------- マスク固有：HUへの変換 ----------
def mask_to_hu_like(mask_data_float):
    """
    DeepDRRはHU→材質→減弱係数でレンダリングするため、
    マスク（0/1）を疑似HUに変換：
      背景: -1000 (空気)
      マスク内部: +2000（非常に高吸収材に擬態）
    目的は「形状の投影を安定に可視化」すること（位置合わせ用）
    """
    hu = np.full(mask_data_float.shape, -1000.0, dtype=np.float32)
    hu[mask_data_float > 0.5] = 2000.0
    return hu

def main():
    # ===== 入力 =====
    ct_nifti   = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    mask_nifti = Path("/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz")
    outdir     = Path("/workspace/deepdrr/outputs_mask_proj")
    outdir.mkdir(parents=True, exist_ok=True)

    # ===== 1) CTロード（あなたのDRRと同じ） =====
    ct_vol = load_volume_LPS(ct_nifti)

    # ===== 2) マスクを同じLPS正規化でロード → 疑似HUに置換 =====
    # （load_volume_LPS は HU値を受け取る前提なので、maskは一旦取り直し）
    img_mask = nib.load(str(mask_nifti))
    in_ornt  = nio.io_orientation(img_mask.affine)
    out_ornt = nio.axcodes2ornt(("L","P","S"))
    to_lps   = nio.ornt_transform(in_ornt, out_ornt)
    mask_lps = nio.apply_orientation(img_mask.get_fdata().astype(np.float32), to_lps)

    # 疑似HUへ
    hu_mask  = mask_to_hu_like(mask_lps)

    # マスク用 Volume（CTと同じ原点・spacing 算出法で構築）
    aff_mask_lps = img_mask.affine @ nio.inv_ornt_aff(to_lps, img_mask.shape)
    spacing_mask = voxel_sizes(aff_mask_lps)[:3]
    shp_mask     = np.array(hu_mask.shape, dtype=float)
    origin_mask  = (-0.5 * (shp_mask - 1.0) * spacing_mask).tolist()
    mask_vol = Volume.from_hu(
        hu_values=hu_mask,
        origin=origin_mask,
        spacing=list(spacing_mask),
        anatomical_coordinate_system="LPS",
    )

    # ===== 3) 同一の C‑arm を作成し、CT・マスク両方を isocenter に配置 =====
    carm = make_carm(ct_vol.center_in_world)          # 幾何はCT基準で作成
    ct_vol.place_center(carm.isocenter_in_world)      # 公式手順
    mask_vol.place_center(carm.isocenter_in_world)    # ★CTと同じ配置にするのが肝
    # 参考：DeepDRR公式の流れ（place_center → Projector）:contentReference[oaicite:2]{index=2}

    # ===== 4) PA投影（γ=90°で縦向き）・PAのみ水平反転 =====
    GAMMA_ROLL = 90.0
    img_ct_energy   = project_and_flip(ct_vol,   carm, alpha=0, beta=0, gamma=GAMMA_ROLL, do_pa_flip=True)
    img_mask_energy = project_and_flip(mask_vol, carm, alpha=0, beta=0, gamma=GAMMA_ROLL, do_pa_flip=True)

    # ===== 5) 表示用正規化（Energy→-log→0-1） =====
    def to_log01(x):
        x = np.clip(x, 1e-6, None)
        y = -np.log(x)
        y = (y - y.min()) / max(y.ptp(), 1e-6)
        return y

    ct_log01   = to_log01(img_ct_energy)
    mask_log01 = to_log01(img_mask_energy)

    # 位置合わせ確認のため二値化（シルエット）
    # 閾値は適宜（0.05〜0.2 目安）。境界が最も分かる値を選ぶ。
    thr = 0.10
    mask_sil = (mask_log01 >= thr).astype(np.uint8)  # 0/1

    # ===== 6) 保存と可視化 =====
    imageio.imwrite(outdir / "PA_mask_energy.png", (mask_log01*65535).astype(np.uint16))
    imageio.imwrite(outdir / "PA_ct_energy.png",   (ct_log01*65535).astype(np.uint16))
    imageio.imwrite(outdir / "PA_mask_silhouette.png", (mask_sil*65535).astype(np.uint16))

    # 重ね合わせ可視化
    plt.figure(figsize=(14,10))
    plt.subplot(2,2,1); plt.imshow(ct_log01, cmap="gray"); plt.title("CT DRR (-log norm)"); plt.axis("off")
    plt.subplot(2,2,2); plt.imshow(mask_log01, cmap="hot"); plt.title("Mask DRR (-log norm)"); plt.axis("off")
    plt.subplot(2,2,3); 
    plt.imshow(ct_log01, cmap="gray")
    plt.imshow(np.ma.masked_where(mask_sil==0, mask_sil), cmap="Reds", alpha=0.45)
    plt.title("Overlay (CT + Mask silhouette)"); plt.axis("off")

    # 差分（位置検証用）
    # マスク側を[0,1]→[0,1]で合わせ、単純な |I_ct - I_mask| を参考値として表示
    diff = np.abs(ct_log01 - mask_log01)
    plt.subplot(2,2,4); plt.imshow(diff, cmap="plasma"); plt.title(f"Abs diff (mean={diff.mean():.4f})"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "PA_overlay_check.png", dpi=150)
    plt.close()

    print(f"Saved in: {outdir}")

if __name__ == "__main__":
    main()
```

### これで「合う」理由

* **同一の C‑arm 幾何**（SID/SAD、ピクセルピッチ、センサ横縦サイズ）を **同一の1個体**で共有。
* **同一の座標系（LPS）・同一の世界原点**（体積中心=0）で CT とマスクを **同じ isocenter に配置**。
* \*\*同一の投影コードパス（DeepDRR Projector）\*\*を通す。
* **同一のPA水平反転を画像側で1回だけ適用**。
  DeepDRRの典型例でも `MobileCArm`→`place_center()`→`Projector` という流れが基本で、これに忠実です。([deepdrr.readthedocs.io][1])

---

## 実運用での注意（落とし穴と対処）

1. **マスクがCTと同一ヘッダで出ているか**

   * NIfTIの **affineと配列の向き**がCTと一致していないと、LPS正規化後でもズレます。
   * 上のコードは **両者をそれぞれ LPS化**→**体積中心原点**→**同じisocenterにplace**で整合させます。
   * DeepDRRの例でも `place_center(carm.isocenter_in_world)` が推奨です。([deepdrr.readthedocs.io][1])

2. **PAの左右反転**

   * あなたのCT DRRと同じく **画像側で1回だけ水平反転**に統一（LATは反転しない）。
   * こうして radiographic view（患者左=画像右）に揃えます。※二重反転は厳禁。

3. **mask→HUの割り当ては「形状を出す」ための擬態**

   * 物理的な厚み計量が目的でなければ、**どの正値でも可**。
   * もし “厚み(㎜)” を取りたいなら、**単色・既知μでの外挿**が必要（下記）。

---

## （任意）厚み\[mm]をDeepDRR側で近似する方法

* **考え方**：`I = exp(-μ L)` に従うので、**-log(I)/μ ≒ L**。
* そのため、**マスク内の物質を単一材（一定μ）**に擬態させ、**-log**後に **1/μ 倍**すれば厚みになる。
* μ（線減弱係数）はエネルギー依存。DeepDRRはスペクトルを扱うので厳密化は難しいですが、
  **単色近似 & μを実験的に決め打ち**（例えば「厚さT mmの一様スラブ」で投影→ `μ = -log(I_meas)/T`）すれば、
  **厚みマップに換算**できます。
  （DeepDRRの Projector と幾何はそのまま利用できます。）([deepdrr.readthedocs.io][3])

---

## 公式資料（参考）

* DeepDRR “Usage” – `MobileCArm` / `place_center` / `Projector` の基本パイプライン。
  → CTを世界原点・isocenter に合わせてから Projector で投影する流れ。([deepdrr.readthedocs.io][1])
* DeepDRR GitHub – 基本的なセットアップと使用例。([GitHub][2])
* DeepDRR Projector パッケージ（仕様の俯瞰）。([deepdrr.readthedocs.io][3])

---

## うまくいかない場合の最短デバッグ手順

1. **同一 run 内で CT とマスクを連続投影**（同じ `carm`・同じ `Projector`）に必ずする。
2. **LPS化後の shape/spacing/origin をログ**（CTとマスクで一致性を確認）。
3. **PAの水平反転の適用箇所が1回だけか**チェック（CT/Maskとも同じルール）。
4. **オーバーレイ画像の中心クロップ（400×400程度）でエッジ一致**を確認。
5. それでもズレる場合は、**マスク生成段階**（v5.4の処理）で **affine が書き換わっていないか**を疑う。

   * CTヘッダの affine を **強制的に流用**してマスクを保存→再投影で差が消えるかを確認。

この手順・コードで「**DeepDRRのみ**」に統一すれば、DRRとマスク投影は**見た目・位置とも完全一致**になります。

[1]: https://deepdrr.readthedocs.io/README.html "DeepDRR — DeepDRR 1.1.0a3 documentation"
[2]: https://github.com/arcadelab/deepdrr "GitHub - arcadelab/deepdrr: Code for \"DeepDRR: A Catalyst for Machine Learning in Fluoroscopy-guided Procedures\". https://arxiv.org/abs/1803.08606"
[3]: https://deepdrr.readthedocs.io/deepdrr.projector.html "deepdrr.projector package — DeepDRR 1.1.0a3 documentation"
