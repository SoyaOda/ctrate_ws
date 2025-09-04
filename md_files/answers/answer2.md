ご提供の3枚（`deepdrr_PA.png` / `deepdrr_LAT.png` / 比較図）を見ると、

* **画面の右上に一部だけ写る／大きく拡大され過ぎ** → **幾何設定（SID/ピクセルピッチ/検出器サイズ）が胸部撮影向けでない**
* **白黒が臨床のX線と逆**（骨が黒っぽい）→ **DeepDRRの生出力は「検出器への平均エネルギー堆積」**なので、そのままだと**透過の大きい肺が明るく**なります（臨床は骨=白, 肺=黒）。
* **全体にコントラストが低い** → 動的範囲が広く、0–1正規化だけでは潰れがち

が主因です。下面の「**そのまま差し替えるだけ**」のコードで、**PA/LATの胸部レントゲン様DRR**に直ります。

---

## まず押さえるポイント（根拠）

* DeepDRRの最小例は `Volume.from_nifti → orient_patient → place_center → MobileCArm.move_to → projector()` です。生の\*\*検出器読みは“平均エネルギー堆積”\*\*で、必要なら独自の検出器を実装して良いと明記されています。([DeepDRR][1])
* `MobileCArm` は**ソース–検出器距離**や**ピクセルサイズ**、**センサー縦横画素**などを**明示的に指定**できます（例：`source_to_detector_distance`、`pixel_size`、`sensor_width/height` など）。**幾何を更新した直後の最初の `projector()` が反映されないため 2 回呼ぶ**という実例もあります。([Vivek Gopalakrishnan][2])
* **胸部PA/LATは SID=180 cm が標準**（多くのガイドや臨床記事）。([elearning.isrrt.org][3], [AuntMinnie][4], [PMC][5], [Radiopaedia][6])
* **FPDの画素ピッチは概ね 0.1–0.3 mm**（胸部一般）。([PMC][7])

---

## すぐ直る差分コード（PA/LATの見え・階調・FOVを修正）

> そのまま置き換え可能。**SID=1800 mm, ピクセルサイズ=0.30 mm, 2048×2048** で胸部全体をカバー。
> 幾何更新直後の **`projector()` を“2回”呼ぶ**点が重要です（既知の挙動）。([Vivek Gopalakrishnan][2])

```python
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector  # PyCUDA初期化を先に
import matplotlib.pyplot as plt

# 1) 見た目をX線らしく： -log + パーセンタイル窓で16bit化
def to_uint16_xray(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    x = np.clip(img.astype(np.float32), 1e-6, None)   # 透過0回避
    x = -np.log(x)                                    # Beer-Lambert 表示（骨=白, 肺=黒）
    lo, hi = np.percentile(x, [1.0, 99.5])            # 外れ値に頑健
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma != 1.0:
        x = np.power(x, 1.0/gamma)                    # 必要なら階調補正
    return (x * 65535).astype(np.uint16)

# 2) 胸部撮影向けのCアームを作る：SID=180 cm, 0.30 mm/pix, 2048^2
def make_cxr_carm(isocenter):
    return MobileCArm(
        isocenter=isocenter,
        rotate_camera_left=False,              # 左右が反なら True に
        source_to_detector_distance=1800.0,    # mm（PA/LATの標準SID）  :contentReference[oaicite:5]{index=5}
        source_to_isocenter_vertical_distance=900.0,  # ソース→アイソセンタ=SID/2
        pixel_size=0.30,                       # mm（FPDの代表値域）      :contentReference[oaicite:6]{index=6}
        sensor_height=2048,                    # 検出器画素
        sensor_width=2048,
        min_alpha=-720, max_alpha=720,         # 可動域は広め
        min_beta=-720,  max_beta=720,
    )

# 3) 1ビュー投影（幾何を動かした直後は projector() を2回）
def render_one(ct: Volume, carm: MobileCArm, alpha, beta, gamma=0.0):
    carm.move_to(alpha=alpha, beta=beta, gamma=gamma, degrees=True)
    with Projector(ct, carm=carm) as projector:
        _ = projector()          # 1回目は幾何更新の“慣らし”  :contentReference[oaicite:7]{index=7}
        img = projector()        # 2回目を採用
    return img

# 4) 実行
nifti_path = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
out_dir = Path("/workspace/deepdrr/outputs"); out_dir.mkdir(parents=True, exist_ok=True)

ct = Volume.from_nifti(str(nifti_path))
ct.orient_patient(head_first=True, supine=True)        # 体位を規格化（READMEの流れ）  :contentReference[oaicite:8]{index=8}
carm = make_cxr_carm(isocenter=ct.center_in_world)     # 体幹中心をアイソセンタへ
ct.place_center(carm.isocenter_in_world)

# **PA（正面：後前方向）** と **LAT（側面）** を生成
img_pa  = render_one(ct, carm, alpha=0.0,   beta=0.0,  gamma=0.0)   # PA
img_lat = render_one(ct, carm, alpha=90.0,  beta=0.0,  gamma=0.0)   # 右側面（必要なら -90）

# 5) 保存（16bit PNG）
pa_path  = out_dir / "deepdrr_PA_fixed.png"
lat_path = out_dir / "deepdrr_LAT_fixed.png"
imageio.imwrite(pa_path,  to_uint16_xray(img_pa,  gamma=1.0))
imageio.imwrite(lat_path, to_uint16_xray(img_lat, gamma=1.0))

# 6) 目視用の比較図
plt.figure(figsize=(12,6))
plt.subplot(121); plt.imshow(imageio.imread(pa_path),  cmap='gray'); plt.title("PA (fixed)");  plt.axis('off')
plt.subplot(122); plt.imshow(imageio.imread(lat_path), cmap='gray'); plt.title("LAT (fixed)"); plt.axis('off')
plt.suptitle('DeepDRR (Chest-like DRR, SID=180cm, 0.30mm/px)')
viz = out_dir / "deepdrr_fixed_comparison.png"
plt.tight_layout(); plt.savefig(viz, dpi=150, bbox_inches='tight')
print(f"Saved:\n  {pa_path}\n  {lat_path}\n  {viz}")
```

### これで直る理由

* **FOV（画角）**：2048 px × 0.30 mm ≒ **615 mm** の有効視野 → 成人胸郭（\~300–360 mm）を余裕で包含。
* **拡大/トリミング**：**SID 1800 mm** & **SAD 900 mm** で幾何を安定化。**体積中心をアイソセンタへ**置き、**角度だけ回す**運用に。([elearning.isrrt.org][3])
* **白黒反転**：**-log 変換**で「**骨=白、肺=黒**」の臨床のルックに。DeepDRR生出力が**エネルギー堆積**である点に整合。([DeepDRR][1])
* **コントラスト**：1–99.5% **パーセンタイルの窓**でダイナミックレンジを圧縮し見やすく。

> ※ 左右が逆に見えるときは `rotate_camera_left=True` か、`gamma=180` を与えて反転してください。
> ※ LAT は装置・軸の定義差で ±90° が入れ替わることがあります。像の向きを確認して ± を選んでください。([Vivek Gopalakrishnan][2])

---

## 追加のデバッグ・安全運用チェックリスト

1. **値域ログを見る**
   `print(img.min(), img.max())` でDeepDRR出力の範囲を確認。0付近が多いと `-log` で飽和するので `1e-6` の**下限クリップ**は必須。([DeepDRR][1])

2. **幾何の再設定直後は2回投影**
   `move_to()` の直後に `projector()` を**2回**。1回目は捨てる（上の実装どおり）。([Vivek Gopalakrishnan][2])

3. **FOVを数式で把握**
   `FOV_mm = pixel_size * np.array([sensor_width, sensor_height])`。胸部は**30–36 cm**が目安。
   例：`0.30 mm × 2048 ≒ 615 mm` なので充分。

4. **PA/LAT以外**

   * **AP**なら `alpha=180` 目安（心陰影が大きく写るのがAP）。([radiologymasterclass.co.uk][8])
   * β/γは**肩倒しや横向きの補正**で使用（0のままでOKなことが多い）。

5. **画素ピッチと解像度の妥協点**
   3090(24GB)なら 2048² でも通常問題ありませんが、重いときは

   * `sensor_width/height=1536` または
   * `pixel_size=0.35–0.40 mm`
     にしてFOVを維持しつつ画素数を落とす。**FPDの典型画素ピッチは0.1–0.3 mm**。([PMC][7])

---

## 根拠リンク（主要な箇所）

* **DeepDRR 公式**：最小使用例／検出器読み（平均エネルギー堆積）／PyTorch併用時の注意。([DeepDRR][1])
* **MobileCArm の幾何パラメータ例**（`source_to_detector_distance`、`pixel_size`、`sensor_width/height` など）と**幾何更新直後は `projector()` を2回**の実例。([Vivek Gopalakrishnan][2])
* **胸部PA/LATのSID（180 cm）**。([elearning.isrrt.org][3], [AuntMinnie][4], [PMC][5], [Radiopaedia][6])
* **FPDの画素ピッチ**（0.1–0.3 mmが一般的）。([PMC][7])

---

### もしそれでも端に寄る／欠ける場合

* `sensor_width/height` をさらに拡大（例：**2304–2560**）。
* `pixel_size` を **0.25–0.30 mm**に微調整（FOV 拡大）。
* `isocenter` を `ct.center_in_world` ではなく **肺野の重心寄り**に（`ct.place_center()` 前に `isocenter` を変更）。
* 左右が逆なら `rotate_camera_left=True`、上下が逆なら `beta=180` か `gamma=180` を試す。([Vivek Gopalakrishnan][2])

この設定で、**拡大・見切れ・白黒逆転・低コントラスト**の4点は一度に解消できるはずです。必要ならこの幾何を前のバッチスクリプト（`render_views()`）に移植し、PA/LATの角度だけ差し替えてください。

[1]: https://deepdrr.readthedocs.io/README.html "DeepDRR — DeepDRR 1.1.0a3 documentation"
[2]: https://vivekg.dev/DiffDRR/tutorials/metamorphasis.html "Converting to DeepDRR – diffdrr"
[3]: https://www.elearning.isrrt.org/mod/book/view.php?id=287&utm_source=chatgpt.com "Projectional X-ray of the chest: Radiographic technique"
[4]: https://www.auntminnie.com/clinical-news/digital-x-ray/article/15558899/good-positioning-is-key-to-pa-chest-xray-exams?utm_source=chatgpt.com "Good positioning is key to PA chest x-ray exams"
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8424719/?utm_source=chatgpt.com "Local diagnostic reference levels for routine chest X-ray ..."
[6]: https://radiopaedia.org/articles/chest-lateral-view-2?lang=us&utm_source=chatgpt.com "Chest (lateral view) | Radiology Reference Article"
[7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11225395/?utm_source=chatgpt.com "Next-generation digital chest tomosynthesis - PMC"
[8]: https://www.radiologymasterclass.co.uk/tutorials/chest/chest_quality/chest_xray_quality_projection?utm_source=chatgpt.com "Chest X-ray Quality - Projection"
