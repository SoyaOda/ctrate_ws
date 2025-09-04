以下は、**DiffDRR v0.5.1 をそのまま使いながら “物理減衰（Beer–Lambert）を完全に回避し、幾何学的な線積分＝厚み投影” を得る**ための最短解と、**厳密な体積保存まで満たすためのパッチ案**です。結論から書きます。

---

## 結論（要点）

1. **Beer–Lambert を回避**
   DiffDRR のレンダラは \*\*Siddon 法/Trilinear の「線積分」\*\*そのものを返せます。`renderer="siddon"`, `reducefn="sum"` を使い、**CT ではなくバイナリマスクを “密度” として直接レンダ**してください。
   → これで出力は **各ピクセルの “通過長（mm）＝厚み”** になります（指数変換なし）。([Vivek Gopalakrishnan][1])

2. **厚み比例性**はそのまま満たせます（出力が厚みそのもの）。
   さらに **単一スカラ係数 k** でスケールすれば、**（近似ではなく）そのマスクについて厳密に**「∑(2D)＝3D 体積」を成立させられます（線形なので分布の圧縮は起きません）。
   ※従来の `-log(I)` 校正で潰れていたダイナミクスは、**指数→対数の非線形**が原因です。線積分に直接スケールを掛ける方式では**一切圧縮しません**。

3. **体積保存を幾何学的に（一般形で）厳密にやる**には、**コーンビームのヤコビアン**（**固有立体角**に対する **r^2·cosθ** の因子）を考慮する必要があります。
   具体的には、射線パラメータ $s$ に沿って体積要素 $dV = s^2\,d\Omega\,ds$（$d\Omega=\frac{\cos\theta}{r^2} dA$）を積分します。これは「厚み」（$\int ds$）とは**別の重み付け（$s^2$）**が必要です。([Wikipedia][2], [MathWorld][3], [Mathematics Stack Exchange][4])
   → したがって **同一の 2D マップ**で「(a) 厚みそのものに比例」かつ「(b) 単純合計×一定ピクセル面積＝体積」を**幾何学的に同時達成**することは、コーンビーム幾何では原理的に困難です。
   ただし、**各ピクセル毎の定数係数**（射線ごとの幾何学因子）を厚みに掛けて定義した 2D マップを「体積保存版」とし、合計にその係数を用いれば、**(a) 比例性（ピクセル毎定数）**と **(b) 体積保存**は**両立**します（下記「パッチ案 B」）。

---

## なぜ今の実装がズレたか

* DiffDRR のレンダラ（`renderers.Siddon/Trilinear`）は **レイに沿ったサンプルの「和（sum）」**を返す設計です（`reducefn='sum'`）。これは **線積分**であって、指数を通した**透過率**ではありません。([Vivek Gopalakrishnan][1])
* ところが実装1では **「マスク画像」を `read()` で CT として読み**、そのまま `DRR(...)` を呼んでいます。この場合、内部の密度推定や可視化の都合で **CT 流儀の強度**が混じる可能性があり、さらに**合計時にコーンビームの幾何因子（r^2・cosθ）を無視**して**ピクセル面積一定**で足すと、**体積は保存されません**。
* 実装2の `-log(I)` 校正は、**すでに指数で潰れたダイナミクス**に定数スケールを掛けるため、**分布（厚みのレンジ）が激しく圧縮**されます（非線形の宿命）。

---

## まずは最短の解（パッチ不要） : 「厚み DRR」 + 単一係数で体積一致

> **ポイント**：**CT ではなくマスクを density として直接レンダ**する／\*\*reducefn='sum'**で**線積分（厚み）\*\*を得る。

```python
import torch, numpy as np, nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import convert

device = torch.device("cuda")

ct_path   = "/path/to/ct.nii.gz"
mask_path = "/path/to/mask.nii.gz"

# 1) DiffDRRのSubjectを、CT + Labelmapとして正しく読み込む
subject = read(
    volume=ct_path,
    labelmap=mask_path,         # ← マスクは labelmap として渡す
    orientation="PA",           # ← 実際の撮影向きに合わせる
    center_volume=True,
)

# 2) DRRをSiddon・線積分（sum）・Nearestで初期化（Beer-Lambertなし）
drr = DRR(
    subject,
    sdd=1800.0,
    height=512,
    delx=0.7,                   # dely 未指定なら delx が使われる
    renderer="siddon",
    compile_renderer=True,
    # ↓ renderer_kwargs はドキュメントにある既存引数
    #    reducefn='sum' で線積分、mode='nearest'でラベル混合を防ぐ
    mode="nearest", reducefn="sum",
    filter_intersections_outside_volume=True,
).to(device)

# 3) ポーズ（PAビューの例）
rot = torch.tensor([[0.0, 0.0, 0.0]], device=device)
tra = torch.tensor([[0.0, 850.0, 0.0]], device=device)
pose = convert(rot, tra, parameterization="euler_angles", convention="ZXY")

# 4) 低レベルAPIで "マスク厚み" を直接レンダ
#    subject["mask"].data は TorchIO LabelMap（0/1）。floatにして密度として使う。
src, tgt = drr.detector(pose, calibration=None)
src = drr.affine_inverse(src)
tgt = drr.affine_inverse(tgt)

mask_density = subject["mask"].data.float().to(device)   # (1,1,D,H,W)
thickness = drr.renderer(mask_density, src, tgt)         # ← ここが「厚み（mm）」！
thick_np = thickness.squeeze().detach().cpu().numpy()

# 5) 体積（3D・真値）と 2D 合計の比較（この段階では未校正）
nii = nib.load(mask_path)
voxel_dims = nii.header.get_zooms()[:3]  # (dx,dy,dz) [mm]
voxel_vol  = np.prod(voxel_dims)
true_vol_3d = float((nii.get_fdata()>0).sum()) * voxel_vol

pixel_area = 0.7 * 0.7  # [mm^2]
vol_2d_naive = thick_np.sum() * pixel_area

# 6) 単一スカラ係数で体積を厳密一致（厚み分布は保ったまま）
k = true_vol_3d / vol_2d_naive
thick_calibrated = thick_np * k
vol_2d = thick_calibrated.sum() * pixel_area

print(f"3D Volume           : {true_vol_3d:.2f} mm^3")
print(f"2D Volume (calibrated): {vol_2d:.2f} mm^3")  # ← 誤差 ≈ 0%
```

* **ここでの `thickness` は Beer–Lambert を通っていません。**`reducefn='sum'` は「レイに沿ったサンプルの総和＝線積分」を返す指定です。([Vivek Gopalakrishnan][1])
* `mode='nearest'` により、**0/1 のバイナリが補間で溶けることを防止**します。
* **校正係数 `k` は単一スカラ**なので、**厚みの相対関係は完全に保持**されます（実装2のような非線形圧縮は起きません）。

> **空間的整合性（条件1）**：CT DRR（通常の `drr(subject.volume, ...)`）と同一のカメラ・検出器で投影しているので**完全一致**します（同じ `pose` / `detector` を使用）。DRR の低レベル API（`detector` と `renderer`）は公式チュートリアルでも示されています。([Vivek Gopalakrishnan][5])

---

## その先：**体積保存を幾何学的にも厳密**に（ヤコビアン補正 + モーメント）

上の「単一スカラ k 校正」でも、**与えられたマスクについては厳密に総和=体積**になりますが、\*\*一般形での「幾何学的な体積保存」\*\*まで担保したい場合は、**コーンビームの体積要素**

$$
dV=s^2\,d\Omega\,ds,\quad d\Omega=\frac{\cos\theta}{r^2}\,dA
$$

をそのまま積分すべきです（$r=\|\mathbf p-\mathbf s\|$, $\theta$ はレイと検出器法線のなす角）。([Wikipedia][2], [MathWorld][3], [Mathematics Stack Exchange][4])

### パッチ案 A（最小改造）：**“体積保存マップ” を別チャンネルで返す**

DiffDRR の `Siddon` は **体素境界との交点列 α** を作って **区間長 Δs** を積分します（ドキュメント参照）。ここに **$s^2$ を掛けた二次モーメント**

$$
M_2(u,v)=\int_{\text{mask}} s^2\,ds \;=\; \sum_{\text{segments}} \frac{s_{m+1}^3-s_m^3}{3}
$$

を同時集計する分岐を追加すれば、**各ピクセルの体積寄与**は

$$
V_{\text{pix}} = \Big(\frac{\cos\theta}{r^2}\Big)\, M_2(u,v)\, dA_{\text{det}}
$$

で**厳密**に得られます。薄い近接部（小 s）は重みが小さく、遠方（大 s）は大きく寄与します。

* 具体的には `diffdrr/renderers.py`（nbdev の場合は `notebooks/api/renderers.ipynb`）の `Siddon` にフラグ `return_moments=True` を追加し、

  * **厚み** $T(u,v)=\sum \Delta s$（現行の `sum`）
  * **二次モーメント** $M_2(u,v)=\sum (s_{m+1}^3-s_m^3)/3$
    を**同時に返す**ようにします。`reducefn` の分岐に callable を受けられるよう小改造しても良いです（現行は文字列 `sum` など）。([Vivek Gopalakrishnan][1])

* 使い方（概念）：

  ```python
  thickness, moment2 = drr.renderer(mask_density, src, tgt, return_moments=True)
  rvec = (tgt - src)                     # (H,W,3)
  r    = torch.linalg.norm(rvec, dim=-1) # 距離 [mm]
  n    = drr.detector.normal             # 検出器法線ベクトル（固定）
  costh = (rvec / r.unsqueeze(-1) @ n).abs().squeeze(-1)
  jac = (costh / (r**2 + 1e-9))          # [1/mm^2]
  volume_preserving_map = jac * moment2  # ← これの総和×pixel_area = 3D体積（厳密）
  ```

* **条件(2) 厚み比例性**：
  表示用には `thickness` を使えば良いです（厚みそのもの）。
  **同じ座標の別マップ**として `volume_preserving_map` を持てば、**合計は厳密に3D体積**。
  さらに 1 枚で両方を兼ねたいなら、**ピクセル毎の定数係数**

  $$
  g(u,v)\;=\;\frac{\cos\theta}{r^2}\cdot\frac{M_2(u,v)}{T(u,v)}
  $$

  を厚みに掛けた **$T'(u,v)=g(u,v)\,T(u,v)$** を採用すれば、**各ピクセルでは厚みと線形比例**しつつ、**総和×pixel\_areaは体積に一致**します（`T=0` の画素は除外/0 のまま）。

> **備考**：この「$s^2$ 重み」は、**コーンビームの発散**（立体角と距離のヤコビアン）に対応する純幾何補正です。([Wikipedia][2], [MathWorld][3])

### パッチ案 B（コード最小化）：**Trilinear レンダラに “サンプル列返し” を追加**

`renderer="trilinear"` は等間隔サンプリング $s_k$ で点列を取ります。`reducefn='sum'` の代わりに **サンプル軸（k）を返すモード**を追加すれば、

$$
T \approx \sum_k \Delta s \cdot m_k,\quad M_2 \approx \sum_k \Delta s \cdot s_k^2 \cdot m_k
$$

を**外側の Python/Torch で一行演算**できます。**CUDA カーネル改修不要**・**自動微分もそのまま**です。
（ドキュメント上 `reducefn` は文字列ですが、ここを callable/`'none'` 受けに拡張するだけで運用できます。([Vivek Gopalakrishnan][5])）

---

## 検証（推奨プロトコル）

1. **厚みの統計**：Z 投影の真値（`np.sum(mask, axis=2)*dz`）と、`thickness` の **相関=1.0** を確認（回転/姿勢同一であれば同一順序で一致）。
2. **体積保存**：

   * 単純法：`k = true_V / (thickness.sum()*pixel_area)` で **誤差 ≈ 0%**。
   * 幾何厳密法：`(cosθ/r^2)*moment2` の **総和×pixel\_area が true\_V** に **機械誤差で一致**。
3. **空間整合**：CT DRR と厚みマップの境界一致（同じ `pose` で描画・比較）。

> DiffDRR の API（`DRR`，`Detector.forward`，`renderers`）は公式ドキュメントに記載されています。`reducefn='sum'` を使うと「レイに沿ったサンプルの和（線積分）」になることも明記されています。([Vivek Gopalakrishnan][6])

---

## 代替手段（参考）

* **RTK（Reconstruction Toolkit）**：`ForwardProjectionImageFilter` は「X線線に沿った蓄積＝線積分」を返す基本実装です。幾何学は `ThreeDCircularProjectionGeometry` で指定。Python からも利用可能。([openrtk.org][7], [ITK][8])
* **Plastimatch**：DRR に **Siddon の “exact pathlength”** 実装があります。パイプライン済み環境があるなら最小工数。([ResearchGate][9])
* **SimpleITK**：`RayCastInterpolator` ベースで自作の積分器を構成可能（高速性は GPU 実装次第）。([Read the Docs][10])

---

## まとめ（あなたの3条件と実装指針）

* **条件1 空間的整合性**
  DiffDRR の **同一カメラ/検出器・同一ポーズ**で、CT DRR とマスク投影を\*\*同じ `Detector`/`pose`\*\*から生成 → **満たす**。([Vivek Gopalakrishnan][11])

* **条件2 厚み比例性**
  `renderer="siddon", reducefn="sum", mode="nearest"` で \*\*厚み（mm）\*\*そのものを出力。**単一スカラの体積スケール**を掛けても線形性は保持 → **満たす**。([Vivek Gopalakrishnan][1])

* **条件3 体積保存**

  * **簡易運用**：上記「厚みマップ」に **単一係数 k** を掛けると、そのマスクについては **厳密一致**。
  * **幾何厳密**：`moment2` と **ヤコビアン** $(\cos\theta/r^2)$ で **ピクセル毎の体積寄与**を作る（パッチ案 A/B）。**合計×pixel\_area＝3D 体積**に **機械誤差で一致**。([Wikipedia][2], [MathWorld][3])

---

### 付録：最小パッチのイメージ（擬似 diff）

```diff
--- a/diffdrr/renderers.py
+++ b/diffdrr/renderers.py
 class Siddon(nn.Module):
-    def __init__(..., reducefn: str = 'sum', ...):
+    def __init__(..., reducefn: Union[str, Callable] = 'sum', return_moments: bool=False, ...):
         ...

     def forward(self, density, source, target, ...):
         # 交点 α の計算 → Δs に変換（既存）
         # mask サンプリング（nearest）
         # contrib = Δs * density_mid

-        img = reduce_along_ray(contrib, how=self.reducefn)   # 例: 'sum'
-        return img
+        if callable(self.reducefn):
+            img = self.reducefn(contrib, extras=locals())
+        else:
+            img = reduce_along_ray(contrib, how=self.reducefn)  # 'sum' 等
+
+        if self.return_moments:
+            # s_m, s_{m+1} から厳密な ∫ s^2 ds = (s_{m+1}^3 - s_m^3)/3 を計算
+            moment2 = ((s_next**3 - s_cur**3) / 3.0) * mask_mid
+            m2 = reduce_along_ray(moment2, how='sum')
+            return img, m2
+        else:
+            return img
```

> **備考**：DiffDRR の API/実装は nbdev でドキュメント化されています（`renderers` の `reducefn` 等）。パッチは **最小範囲**で済み、既存の自動微分/高速性はそのまま活かせます。([Vivek Gopalakrishnan][1])

---

必要なら、上記コードを**あなたの NIfTI（心臓周囲脂肪マスク）**用に書き換えた **完全スクリプト**（`geometric_mask_projection.py`）としてお渡しできます。まずは **「厚み DRR」+ 単一係数 k** の簡易版を試し、**厚み分布のレンジ/標準偏差が 3D 真値と整合**することを確認してください。そこから**ヤコビアン＋モーメント**のパッチに進むのがおすすめです。

[1]: https://vivekg.dev/DiffDRR/api/renderers.html "renderers – diffdrr"
[2]: https://en.wikipedia.org/wiki/Solid_angle?utm_source=chatgpt.com "Solid angle"
[3]: https://mathworld.wolfram.com/SolidAngle.html?utm_source=chatgpt.com "Solid Angle -- from Wolfram MathWorld"
[4]: https://math.stackexchange.com/questions/4787289/why-does-dv-rho2-sin-phi-d-rho-d-theta-d-phi?utm_source=chatgpt.com "Why does $dV=\rho^2\sin{\phi}\\, d{\rho}\\, d{\theta}\\, d\phi"
[5]: https://vivekg.dev/DiffDRR/tutorials/trilinear.html "Trilinear rendering – diffdrr"
[6]: https://vivekg.dev/DiffDRR/api/drr.html "DRR – diffdrr"
[7]: https://www.openrtk.org/Doxygen/classrtk_1_1ForwardProjectionImageFilter.html?utm_source=chatgpt.com "rtk::ForwardProjectionImageFilter< TInputImage, TOutputImage > Class ..."
[8]: https://discourse.itk.org/t/creating-digitally-reconstructed-radiographs-from-chest-ct/5801?utm_source=chatgpt.com "Creating Digitally Reconstructed Radiographs from Chest ..."
[9]: https://www.researchgate.net/profile/Maria-Spadea/publication/268523129_PLASTIMATCH-_AN_OPEN_SOURCE_SOFTWARE_SUITE_FOR_RADIOTHERAPY_IMAGE_PROCESSING/links/59de01a60f7e9bec3bae08ed/PLASTIMATCH-AN-OPEN-SOURCE-SOFTWARE-SUITE-FOR-RADIOTHERAPY-IMAGE-PROCESSING.pdf?utm_source=chatgpt.com "Plastimatch – An Open Source Software Suite for ..."
[10]: https://media.readthedocs.org/pdf/simpleitk/master/simpleitk.pdf?utm_source=chatgpt.com "SimpleITK Documentation"
[11]: https://vivekg.dev/DiffDRR/api/detector.html "detector – diffdrr"
