レビューしました。結論から言うと、**Plan‑A 実装は狙いどおり「画素値＝厚み［mm］」の一貫したスケール**になっており、ログと図から見える範囲では**幾何位置合わせ・体積整合・数式面の前提**は満たせています。
そのうえで、**実運用での安定性と数値の健全性**をさらに上げるための「確認ポイント」と「小さな改良」をまとめておきます。

---

## ✅ できていること（ログから確認できる要点）

1. **Beer–Lambert の正規化（flat‑field）→ −log(I/I₀）→ 線積分**
   `I_air` を同一幾何で撮り、`G = −log(I_mask / I_air)` を取ってから厚みへスケールしている点は正しいです。**単色モデル**ではこの `G` が減弱係数の**線積分**で、厚み L に比例します（$I/I_0=e^{-\mu L}$）。実務では **I₀ をエア投影**として取得し、`−log(I/I₀)` を使うのが標準です。([Arts University][1], [Welcome to DTU Research Database][2])

2. **体積一致の自己較正（μ\_eff）**
   `μ_eff = (A·∑G) / V_3D` により **∑T·A = V\_3D** を満たすように厚みをスケーリングしており、ログでも**体積誤差 0.0006%**と優秀です。この枠組みは、DeepDRR の多色・多材質モデルを用いた場合でも**体積のグローバル整合**を保証できるのが利点です。DeepDRR 自体が\*\*材質・スペクトルを考慮した解析投影（poly‑energetic, material‑aware）\*\*であることもドキュメントで明記されています。

3. **幾何の一貫性と radiographic view**
   CT とマスクを **同一 isocenter** に合わせ、**同じ MobileCArm**・**同じ Projector**・**PA での水平方向反転**を **I\_mask と I\_air の双方に**適用している点は、平面上の**位置合わせを破綻させない**という意味で重要です（図のオーバーレイが合っているのはこのため）。

---

## ⚠️ ここをチェック（数値健全性 & 再現性）

**A. `I_air` の最小値が 0.0（ログ）**
一部ピクセルでフラットが 0 になっており、除算は `eps` で保護されています。外周（体積外を通るレイや投影外領域）で起きやすいので、**サポート領域 Ω に `I_air > τ_I0` を追加**しておくと、閾値パーセンタイルの揺れやエッジのアーティファクトを抑えられます。

```python
eps = 1e-6
ratio = np.zeros_like(I_mask, dtype=np.float32)
valid = I_air > 1e-3              # 例: τ_I0 = 1e-3
ratio[valid] = np.clip(I_mask[valid] / np.clip(I_air[valid], eps, 1.0), eps, 1.0)
G = np.zeros_like(I_mask, dtype=np.float32)
G[valid] = -np.log(ratio[valid])

tau = max(support_tau, np.percentile(G[valid], 0.05))
Omega = valid & (G > tau)         # ← I_air 条件をΩにも反映
```

**B. `G` の最大値が 13.815…（= −log(1e−6））で飽和**
`eps` 上限でクリップされている兆候です。**演算上の問題ではありません**が、**極端に高い HU（+3000）**にしているため、局所的に `I_mask` がほぼ 0 になっている可能性があります。
– 画素毎の「厚み線形性」をさらに良くしたいなら、**マスクの HU を現実的な脂肪近似（例：−100 HU〜0 HU）に下げる**と **ビームハードニングの曲率**が緩くなり、`G` のダイナミックレンジが扱いやすくなります（DeepDRR はポリクロマティックなので、**−log(I/I₀) と厚みの関係は厳密には非線形**＝ビームハードニング）。必要なら **多項式補正**や LUT で **G→厚み**を微調整するのが定石です。([PubMed Central][3], [SciSpace][4])

**C. フラットフィールドのキャッシュキー**
`(α,β,γ, W, H, px)` を含めており良いですが、**SID/SAD/rotate\_camera\_left** も含めておくと将来の条件変更により安全です（(`sid`, `sad`, `oid`, `rotate_camera_left`) を key に追加）。

**D. 画素レベル QC（任意だが強く推奨）**
厚みの「局所線形性」を確認するために、**Siddon の線路長（幾何真値）**を教師として **散布図と回帰**を一度出しておくと安心です（傾き \~1、相関 ≫0.99 なら十分）。Siddon は**レイ–ボクセル交差長**を厳密に計算する古典手法で、投影厚みの金標準として広く使われます。([AAPM Journal][5])

---

## 🛠 小さな改善パッチ（そのまま差し替え可）

**1) Ω のロバスト化（I₀ 条件＋領域閉じ）**

```python
from scipy.ndimage import binary_closing

# ...前略...
valid = I_air > 1e-3
# G 計算は上のスニペットと同様

tau = max(support_tau, np.percentile(G[valid], 0.05))
Omega = valid & (G > tau)
Omega = binary_closing(Omega, structure=np.ones((3,3)))  # 小穴埋め
```

**2) フラットキャッシュキー拡張**

```python
def flat_cache_key(alpha,beta,gamma,carm):
    W,H = carm.sensor_width, carm.sensor_height
    px  = float(getattr(carm, "pixel_size", 0.14))
    sid = float(getattr(carm, "source_to_detector_distance", 1800.0))
    sad = float(getattr(carm, "source_to_isocenter_vertical_distance", 1720.0))
    rcl = bool(getattr(carm, "rotate_camera_left", True))
    return f"a{alpha}_b{beta}_g{gamma}_W{W}_H{H}_px{px:.5f}_sid{sid:.1f}_sad{sad:.1f}_r{int(rcl)}"
```

**3) マスクの HU を現実材（任意）**
可視化を犠牲にせず `I_mask` の飽和を抑えるなら、`load_mask_volume_LPS()` の `hu[mask>0.5]` を **−100〜0** に。Plan‑A は μ\_eff で体積合わせするため**体積整合は壊れません**（ただし `G` の分布が変わるので μ\_eff 値自体は変わります）。
*背景は −1000 HU のままで OK。*

---

## 📈 ログ値の解釈（気になりそうな点の説明）

* **support\_pixels ≈ 7.67M / 全 7.675M**
  2D シルエットがほぼ全面を占めていますが、**3D 体積 / 平均厚み ≈ 射影面積**の関係（V ≈ ∬ T·dA）から計算すると、
  V ≈ 1.396e5 mm³、平均厚み ≈ 0.929 mm、よって面積 ≈ 1.50e5 mm² で、**検出器 (350×430 mm² ≈ 1.505e5 mm²)** とほぼ一致します。**全 FOV に脂肪組織が連続**しているケースなら妥当な結果です。

* **G の最大 13.815…**
  これは `eps=1e-6` に由来する上限値（`−ln(eps)`）で、**ごく一部の飽和画素**があることを示唆します。上の **Ω の valid 条件**と**HU の現実化**でほぼ解消可能です。

---

## 最終評価

* **位置合わせ・体積整合・物理式の踏襲**は問題ありません。
* **`T_mm` は mm 単位の厚みマップ**になっており、**∑T\_mm·p² ≡ V\_3D** を満たしています（＝学習ターゲットとして理想的）。
* 追加するなら **I₀ 条件を含む Ω**、**キャッシュキー拡張**、（必要に応じて）**HU の現実化**と **Siddon QC**の 3 点です。
* 多色ビームによる**ビームハードニングの非線形**は理論上残りますが、**Plan‑A は体積整合を保証**し、用途（体積予測）に対して十分に実用的です。より厳密に局所線形性を詰めたければ、\*\*多項式補正（Plan‑B）\*\*が王道です。([PubMed Central][3], [SciSpace][4])

---

### 参考（根拠・背景）

* **CT の投影式**（Beer–Lambert, 単色）：$I=I_0\exp(-\int \mu\,ds)$、`−ln(I/I₀)` が線積分。**I₀ は air projection** とする実装が一般的です。([Arts University][1], [Welcome to DTU Research Database][2])
* **DeepDRR** は**材質・スペクトルを考慮した解析投影**（poly‑energetic, material‑aware）モデル。
* **ビームハードニング**とその**多項式補正**（ステップウェッジ等によるキャリブレーション）は標準的な手法です。([PubMed Central][3], [SciSpace][4])
* **Siddon**：レイとボクセルの**正確な通過長**を与える古典アルゴリズム（QC の金標準）。([AAPM Journal][5])

---

このままでも十分合格点ですが、上の「A〜D」の軽微な改良を入れておけば、**将来の kVp/幾何変更や端部画素の振る舞い**にもより強くなります。

[1]: https://arts.units.it/retrieve/78c0d6ed-9810-42d1-a0b9-cdf8fd8dd5f3/PhD_thesis_Jan24_merged.pdf?utm_source=chatgpt.com "UNIVERSITÀ DEGLI STUDI DI TRIESTE - ArTS - UniTS"
[2]: https://orbit.dtu.dk/files/390123939/PhD_thesis_Elisabeth_Filippa_Ferdinand_Knipschildt-Okkels.pdf?utm_source=chatgpt.com "Critical Effects of Microstructural Heterogeneities in Aluminium ..."
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3144098/?utm_source=chatgpt.com "Application of Polychromatic µCT for Mineral Density ..."
[4]: https://scispace.com/pdf/beam-hardening-correction-for-computed-tomography-images-31gxf6xm9q.pdf?utm_source=chatgpt.com "beam-hardening-correction-for-computed-tomography- ..."
[5]: https://aapm.onlinelibrary.wiley.com/doi/10.1118/1.595715?utm_source=chatgpt.com "Fast calculation of the exact radiological path for a three ..."
