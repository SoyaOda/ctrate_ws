すばらしい検証ログです。いただいた現象は **「深DRRの世界座標（LPS・体積中心原点）と、DiffDRRの既定の“中心化+PAフレーム”」の扱い**、および **オイラー角の順序差** と **PAの左右規約** が混ざって起きています。以下に、**DeepDRRとDiffDRRでピクセル単位で一致**させるための“公式チュートリアル準拠”の決定版レシピと、質問 1–4 への回答をまとめます。
（重要な根拠は DiffDRR 公式ドキュメントの「Converting to DeepDRR」「3D geometry」「DRR API」「data.read」各ページと DeepDRR README に基づきます。） ([Vivek Gopalakrishnan][1], [deepdrr.readthedocs.io][2])

---

## 結論（先に）

* **原点合わせは「両者とも体積中心＝原点」に統一**するのが最も安全。
  DiffDRR は `read(..., center_volume=True)` で **体積中心を世界原点**に移すのが既定動作。DeepDRR 側も **体積中心が世界原点になるように origin を置く**（= DeepDRR側のチュートリアルがそうしている）。このとき **DiffDRR の `translation` は \[0,0,0] で良い**（DeepDRR 側で isocenter に体積中心を与える）。 ([Vivek Gopalakrishnan][1])
* **角度は “DeepDRR → DiffDRR”で順序変換が必要**：
  **α′=α, β′=90°−β, γ′=−γ**（単位はラジアン）。これは \*\*DiffDRR の公式「Converting to DeepDRR」\*\*にある対応です。 ([Vivek Gopalakrishnan][1])
* **PAの左右について**：
  DiffDRR の `DRR` は既定で `reverse_x_axis=True`（**放射線科の規約＝心陰影が画像右**）を守る向きに出します。一方、DeepDRR はそのままだと左右が逆なので **最終的に水平反転（`[:, ::-1]`）** するのが公式の対応です（同チュートリアルのコードそのまま）。\*\*両者とも“PAは最終的に radiographic view に合わせる”\*\*という方針を徹底してください。 ([Vivek Gopalakrishnan][3])

---

## 1) 原点一致（center\_volume と isocenter の整合）

### なぜズレたか

* **DiffDRR**：`read(..., orientation="PA", center_volume=True)` は「**体積中心を世界原点に移す**」「**RAS+ へ正規化**」を行います。これにより、**元の物理座標（CT の DICOM/NIfTI の world）から平行移動**されます。
* **DeepDRR**：あなたの最終版は NIfTI を LPS にそろえ、`origin` を体積中心が原点になるように置いています（DeepDRR チュートリアルも **体積中心＝原点** で `Volume` を組み立てています）。**ここを一致させればズレません**。 ([Vivek Gopalakrishnan][1])

### 正しい合わせ方（推奨：両者とも体積中心＝原点）

* **DiffDRR 側**：`read(..., center_volume=True)` のままで OK（= 体積中心が原点）。
* **DeepDRR 側**：体積中心が世界原点になるように `origin` を設定し（深DRRチュートリアルどおり）、**Cアームの isocenter を `patient.center_in_world` に置く**。これで **両者の isocenter/world 原点が一致**します。 ([Vivek Gopalakrishnan][1])
* **DiffDRR の `translation`**：上記なら **`translation = [0,0,0]`** でよい（\*\*R·t が“X線源の世界座標”\*\*になるため、原点を isocenter と一致させる設計では、回転と SDD だけで正しい位置関係が決まります）。 ([Vivek Gopalakrishnan][4])

> 注：DiffDRR のチュートリアルでは、**生の `volume` 配列を直接渡す構成**の例もあり、その場合は `translation=[(shape−1)*spacing/2]` を与えて体積中心を原点に見なす実装になっています（中身はやっていることが同じ）。どちらの流儀でも **「体積中心＝原点」さえ一致**していれば OK です。 ([Vivek Gopalakrishnan][1])

---

## 2) 角度変換の具体実装（DeepDRR → DiffDRR）

* DeepDRR の `MobileCArm.move_to(alpha, beta, gamma)` に対し、**DiffDRR の Euler は “YZX”** 規約で受けるため、**以下の変換**を適用します：

  $$
  \alpha_{\text{DiffDRR}}=\alpha,\quad
  \beta_{\text{DiffDRR}}=\frac{\pi}{2}-\beta,\quad
  \gamma_{\text{DiffDRR}}=-\gamma
  $$

  （単位はラジアン）
* これは公式チュートリアルのコードそのままです（DeepDRR 側では `beta=np.pi/2 - beta`, `gamma=-gamma` で move\_to して、最終的に水平反転しています）。 ([Vivek Gopalakrishnan][1])

**コード例（角度のみ示す）**：

```python
# 例：DeepDRR で使った角度（度）
alpha_deg, beta_deg, gamma_deg = 0.0, 0.0, 90.0   # あなたの最終版の PA

# DiffDRR に渡す角度（ラジアン）
import math, torch
a = math.radians(alpha_deg)
b = math.radians(beta_deg)
g = math.radians(gamma_deg)

angles_diffdrr = torch.tensor([[a, (math.pi/2) - b, -g]], dtype=torch.float32)  # convention="YZX"
```

変換を自分で書きたくない場合は、**`diffdrr.detector.diffdrr_to_deepdrr`** ユーティリティを使う形でも同等に動きます（チュートリアルで使用）。 ([Vivek Gopalakrishnan][1])

---

## 3) 検出器パラメータの完全一致（DeepDRR → DiffDRR）

**DeepDRR から拾う値**（あなたのスクリプトと同じ意味のもの）：

* **SDD（=SID）**：`carm.source_to_detector_distance`
* **ピクセルサイズ**：`carm.pixel_size`（mm/pixel）
* **画像サイズ**：`carm.sensor_width`, `carm.sensor_height`
* **（任意）主点**：中央なら `x0=0, y0=0` のままで良い

**DiffDRR の `DRR` 初期化に渡す**：

```python
drr = DRR(
    subject,                                # read(..., center_volume=True) の戻り値
    sdd=float(carm.source_to_detector_distance),
    height=int(carm.sensor_height),
    width=int(carm.sensor_width),
    delx=float(carm.pixel_size),
    dely=float(carm.pixel_size),
    reverse_x_axis=True,                    # radiographic view を守る（後述）
    renderer="siddon",                      # マスク厚さ用は合計でOK
)
```

* `reverse_x_axis` は **既定で True（“心陰影は画像右”の放射線科規約）**。DeepDRR との一致をとるなら **どちら側で反転するかを必ず統一**してください。 ([Vivek Gopalakrishnan][3])
* DeepDRR 側はチュートリアル通り **最終的に `img = projector(); img = projector()[:, ::-1]`** と **水平反転**しています。従って **DiffDRR 側は `reverse_x_axis=True` のままで整合**がとれます。 ([Vivek Gopalakrishnan][1])

---

## 4) 「PAのみ左右反転」を DiffDRR にも適用する時期と方法

あなたの現行パイプラインは \*\*「DeepDRR は PA のみ最終水平反転」\*\*で落ち着いています。この方針に DiffDRR を合わせるには **2通り**あります：

* **A案（最小変更／現行に追従）**

  * **DiffDRR 側では `reverse_x_axis=False`** にして **反転は行わない**。
  * **PA のみ**最終段で `img_pa = img_pa[:, ::-1]` を適用（DeepDRR と同じタイミング＝**render 後**）。
  * **LAT には反転しない**。
    → 既存 DeepDRR の動作に寄せる最小変更。

* **B案（堅牢化／将来の角度追加にも安心）**

  * **DeepDRR 側の水平反転をやめる**（`[:, ::-1]` を外す or `rotate_camera_left=False` に整備）。
  * **DiffDRR は `reverse_x_axis=True`** のままにして **全ビュー一貫して radiographic view** を守る。
  * これで **PA/LAT とも左右規約が自動で一致**します（おすすめ）。
    → 公式 `DRR` 既定の設計に従うので保守性が高い。 ([Vivek Gopalakrishnan][3])

---

## 追加：厚さ（mask の線積分）を相関係数=1で返すには

* DiffDRR の **Siddon レンダラ**は **線分通過長の総和**（＝**Beer–Lambert ではなく純粋な長さ**）を返せるので、**0/1 マスク**なら **ピクセル値=通過厚（mm）** になります。スカラー正規化をしなければ **相関係数=1** で厚さが出ます（その後の PNG/DICOM 保存時に 16bit 正規化する場合は、**DeepDRR 側マスクも同じ正規化**を適用）。
  実装は **`renderer="siddon"` のまま**でOK、`forward()` 後の配列をそのまま使います（減衰や `-log` は掛けない）。**Beer–Lambert の影響は入りません**。 ([Vivek Gopalakrishnan][3])

---

## 参考コード（要点だけ抜粋）

```python
# --- DiffDRR 側の読み込み（CTとmaskは同じ center_volume=True で） ---
from diffdrr.data import read
subject = read(
    volume=str(ct_path),
    labelmap=str(mask_path),                # 0/1 マスク
    orientation="PA",                       # フレーム変換
    center_volume=True,                     # 体積中心＝原点（DeepDRR と合わせる）
)

# --- DeepDRR 側（体積中心を原点にする。isocenter に center_in_world を渡す） ---
from deepdrr import Volume, MobileCArm, geo
patient = Volume.from_nifti(str(ct_path))  # すでに LPS/中心原点化済みならそのまま
carm = MobileCArm(
    isocenter=patient.center_in_world,     # 体積中心が isocenter
    source_to_detector_distance=SID,
    source_to_isocenter_vertical_distance=SID-OID,
    pixel_size=PIX, sensor_width=W, sensor_height=H,
)

# DeepDRR 投影（最終で水平反転; 公式チュートリアル通り）
from deepdrr.projector import Projector
with Projector(patient, carm=carm) as projector:
    carm.move_to(alpha=alpha_deg, beta=beta_deg, gamma=gamma_deg, degrees=True)
    _ = projector()
    img_deepdrr = projector()[:, ::-1]     # radiographic view 準拠の水平反転
# --- DiffDRR 側の DRR 初期化（DeepDRR と同じ幾何） ---
from diffdrr.drr import DRR
drr = DRR(
    subject,
    sdd=float(carm.source_to_detector_distance),
    height=int(carm.sensor_height),
    width=int(carm.sensor_width),
    delx=float(carm.pixel_size),
    dely=float(carm.pixel_size),
    reverse_x_axis=True,                   # radiographic view 準拠（B案では必須）
    renderer="siddon",
)

# 角度変換（DeepDRR → DiffDRR）
import math, torch
a = math.radians(alpha_deg)
b = math.radians(beta_deg)
g = math.radians(gamma_deg)
angles = torch.tensor([[a, (math.pi/2) - b, -g]], dtype=torch.float32)

# 原点一致なら translation はゼロでよい
xyz = torch.zeros(1,3, dtype=torch.float32)

# 投影
img_diffdrr = drr(angles, xyz, parameterization="euler_angles", convention="YZX")
```

> 上の角度写像・最終水平反転・幾何一致のやり方は、\*\*DiffDRR の公式「Converting to DeepDRR」\*\*が示す方針にそのまま沿っています。 ([Vivek Gopalakrishnan][1])

---

## まとめ（質問ごとの回答）

1. **isocenter と中心化の一致**
   　\*\*両者とも「体積中心＝世界原点」\*\*にそろえるのが最良。
   　DiffDRR は `center_volume=True`、DeepDRR は `origin` を体積中心に置き `isocenter=patient.center_in_world`。**DiffDRR の `translation` は \[0,0,0]** で構いません。([Vivek Gopalakrishnan][1])

2. **角度変換の実装**
   　**α′=α, β′=90°−β, γ′=−γ（rad）**、`convention="YZX"` を指定して DiffDRR に渡してください。チュートリアル通りです。([Vivek Gopalakrishnan][1])

3. **検出器パラメータの一致**
   　SDD/SID, pixel\_size, 幅・高さを **DeepDRR の `MobileCArm` からそのまま** DiffDRR の `DRR` に入れます。`reverse_x_axis=True` が **radiographic view 準拠**です。([Vivek Gopalakrishnan][3])

4. **PA の左右反転の適用タイミング**
   　**レンダ直後の最終段**で水平反転するのが基本。
   　**A案（現行維持）**：DiffDRR は `reverse_x_axis=False` にして **PAのみ** `[:,::-1]`。
   　**B案（推奨）**：DeepDRR で反転せず、DiffDRR を `reverse_x_axis=True` にして **全ビュー radiographic view を自動で統一**。([Vivek Gopalakrishnan][3])

---

必要なら、あなたの最終スクリプト（DeepDRR 側）に合わせた **DiffDRR 側の最小修正パッチ**（A案/B案のどちらか）も書き起こします。

[1]: https://vivekg.dev/DiffDRR/tutorials/metamorphasis.html "Converting to DeepDRR – diffdrr"
[2]: https://deepdrr.readthedocs.io/README.html?utm_source=chatgpt.com "DeepDRR 1.1.0a3 documentation - Read the Docs"
[3]: https://vivekg.dev/DiffDRR/api/drr.html "DRR – diffdrr"
[4]: https://vivekg.dev/DiffDRR/api/pose.html "pose – diffdrr"
