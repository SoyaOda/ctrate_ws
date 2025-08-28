結論（要点）

* **横隔膜より尾側に入っていく領域では、心臓シェル（heart を等方膨張した殻）だけに頼るより、**
  **(a)「横隔膜の上限」を解剖学的に推定してZでクリップ**し、**(b) その範囲内を “縦隔ROI＝(\~lungs)” で制限**する方が安定です。横隔膜は**胸腔と腹腔を隔てる**膜で、ここを境に“心膜周囲脂肪”の評価対象が終わるためです。([Radiopaedia][1])
* **TotalSegmentator（TS）には現状、横隔膜の専用ラベルはありません**（公式のクラス一覧やREADMEに列挙されない）。一方で **lungs / lung lobes, heart, aorta, pulmonary\_artery, pulmonary\_vein, IVC, liver / stomach / spleen, body などは提供**されます。これらを組み合わせれば**胸腔と腹腔の実用的な分離は可能**です。([GitHub][2], [PMC][3], [3D Slicer Community][4])

---

## なぜ“(\~lungs) の縦隔クランプ + 下端Zクリップ”が必要か

* **横隔膜が胸腔と腹腔を分ける**ため、下端を越えると心膜外脂肪（PAT）の定義外（腹腔脂肪）に入ります。横隔膜自体は薄く、CTでは部分容積の影響を受けやすいので、**幾何的なheartシェルだけだと腹腔側へ漏れやすい**のが実務的な課題です。([Radiopaedia][1], [Radiology Key][5])
* 一方、**縦隔は左右の胸膜（=肺）に囲まれた胸腔中央の区画**です。したがって「**(\~lungs)** で縦隔側に限定」するのは胸部画像処理の定番の絞り込みで、下部胸郭でも**心臓周囲の脂肪（EAT+PAT）へ寄与しない腹腔脂肪の混入を抑える**効果があります。([Kenhub][6], [NCBI][7], [Cleveland Clinic][8])

---

## ただし：(\~lungs) だけでは不十分—Z下限を先に決める

肺が消える直前の**横隔膜ドーム直上**でも、(\~lungs) は**胸壁・腹壁の軟部を広く含む**ため、**Z下限でのクリップが先**です。
横隔膜ラベルが無い代わりに、**「心膜が被覆する近位大血管」を手掛かりに下限Zを決める**のが安定します。

**実装指針（TSラベルのみ）**

1. **下端Z（z\_min）**：`heart` の距離画像（EDT）から **≤5 mm 近接帯**を作り、`inferior_vena_cava` と `pulmonary_vein` の**近接部分**が最後に存在するスライスを z\_min とする（±1スライスマージン）。
   　根拠：**心膜は心臓と“近位の大血管の根部”を被覆**し、IVC・肺静脈は\*\*横隔膜付着/心膜反転（斜洞）\*\*の近傍で終わります。([Radiopaedia][9], [PMC][10])
2. **上端Z（z\_max）**：上行大動脈・肺動脈の**心膜近接部**が最後に見えるスライス（±1）。**上縦隔へ入りすぎない**ための安全策。([Radiopaedia][11])
3. **Zクリップ**：`shell` と `torso_fat` を **\[z\_min, z\_max]** のみ残す。
4. **縦隔クランプ**：上記範囲で **`cand = (shell ∩ torso_fat) ∩ (~lungs)`**。
5. **腹部臓器クランプ（任意・効果大）**：`liver | spleen | stomach` を **2–3 mm 膨張**した禁止帯で `cand` を削る（横隔膜直下の偽陽性を抑制）。([PMC][12])
6. **HUフォールバック（-190〜-30 HU）と“心膜ブリッジ”(heart を1–2 mm膨張) による連結制約**で**心膜直外の薄いPATの偽陰性を救済**。
7. **遠位大血管/傍椎体の除外（任意）**：`aorta` は心臓近接（≤5 mm）のみ許容、残り（下行Ao沿いのPVAT）は**2–3 mm膨張して除外**。椎体（`vertebrae_T*`）は**3–5 mm膨張禁止**。TSの total タスクでラベル提供。([GitHub][13])

> これらはすべて \*\*TSの標準/ハイレゾ心腔タスク・tissue\_types（torso\_fat）\*\*のラベルで実装可能です。ラベル一覧は GitHub の README／クラス表に明記されています。([GitHub][2])

---

## あなたの質問への直接回答

**Q1. 横隔膜（または肝上面）より尾側では、heartシェルよりも肺マスクを使うべき？**
→ **はい（ただしZ下限でクリップした上で）**。

* 下端に近づくほど heartシェルは**腹腔側へ幾何学的に“回り込み”やすい**ため、**(\~lungs)** による縦隔限定の方が**定義（胸腔内の心膜内外）に整合**しやすい。
* その際、**z\_min を IVC/肺静脈の“心膜近接”で決めて**から (\~lungs) を適用するのが安全。そうしないと (\~lungs) だけでは**腹腔軟部**が混入します。([Radiopaedia][1], [NCBI][7])

**Q2. TSで“胸腔”と“腹腔”を直接ラベルで区別できる？**
→ **“専用の cavity ラベル”はありませんが、実用上は分けられます**。

* **胸腔（縦隔）**は**(\~lungs)** と **胸壁（sternum/ribs/vertebrae）内側**で近似可能。
* **腹腔側の混入**は **liver/stomach/spleen** などを**薄く膨張した禁止帯**で抑制。
* **横隔膜の専用ラベルはTSに無い**ため、**近位大血管ベースのZ下限**＋上記クランプの組合せが現実解です。([GitHub][2])

---

## すぐ入れられるパッチ（擬似コード）

```python
# 1) 下端・上端Z
dist = edt(~heart, sampling=spacing)
prox = dist <= 5.0
ivc = load("inferior_vena_cava"); pv = load("pulmonary_vein")
aorta = load("aorta"); pa = load("pulmonary_artery")

z_min = infer_zmin_from((ivc|pv)&prox, heart, margin=1)  # 横隔膜側
z_max = infer_zmax_from((aorta|pa)&prox, heart, margin=1)  # 上縦隔側

# 2) Zクリップ
shell[:, :, :z_min] = False; shell[:, :, z_max+1:] = False
torso_fat[:, :, :z_min] = False; torso_fat[:, :, z_max+1:] = False

# 3) 縦隔ROI（胸腔限定）
lungs = load("lung_left") | load("lung_right")
cand = shell & torso_fat & (~lungs)

# 4) 腹部臓器クランプ（2–3 mm）
abd = dilate(load("liver")|load("stomach")|load("spleen"), mm=3)
cand &= ~abd

# 5) HUフォールバック + 心膜ブリッジ
hu = (ct>=-190)&(ct<=-30)
bridge = dilate(heart, mm=2) & ~heart
eat_pat = binary_propagation((cand&hu)|bridge, mask=(shell|bridge)) & shell & hu
```

---

## 参考（定義確認の根拠）

* **横隔膜は胸腔と腹腔の境界**、下胸郭口を閉じる。([Radiopaedia][1])
* **縦隔は左右の胸膜（=肺）に囲まれた胸腔中央の区画**で、上方は胸郭入口、下方は横隔膜で区切られる。([Kenhub][6], [NCBI][7])
* **心膜は心臓と“大血管の根部”を包む**（上行大動脈根部の**上行大動脈上方陥凹**などの反転/recessが典型）。下方は**横隔膜中央腱**へ付着。これを外れると心膜周囲脂肪の定義外。([Radiopaedia][9], [PMC][10])
* **TSのラベル**：tissue\_types に **torso\_fat**、total/heartchambers\_highres に **aorta/pulmonary\_artery/pulmonary\_vein/IVC**、**lungs** や腹部臓器（liver/stomach/spleen）などが含まれる。([GitHub][2])

---

### まとめ

* **尾側（横隔膜付近）の処理は**「**近位大血管ベースの下端Z**→**(\~lungs) の縦隔クランプ**→**腹部臓器クランプ**→**HU＋心膜ブリッジ**」の順が最も再現性が高いです。
* **TSだけで胸腔/腹腔の区別は“十分に実用的に可能”**（専用の横隔膜ラベルは無いが、既存ラベルの組合せで代替可能）。
* あなたの現行実装にこのロジックを差し込めば、添付の下部スライスの\*\*腹腔混入（偽陽性）と胸腔側の抜け（偽陰性）\*\*の多くを抑えられます。

[1]: https://radiopaedia.org/articles/diaphragm?lang=us&utm_source=chatgpt.com "Diaphragm | Radiology Reference Article"
[2]: https://github.com/wasserth/TotalSegmentator?utm_source=chatgpt.com "wasserth/TotalSegmentator: Tool for robust segmentation ..."
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10546353/?utm_source=chatgpt.com "Robust Segmentation of 104 Anatomic Structures in CT ..."
[4]: https://discourse.slicer.org/t/segment-the-body-surface-using-totalsegmentator/27666?utm_source=chatgpt.com "Segment the body surface using TotalSegmentator - Support"
[5]: https://radiologykey.com/10-diaphragm/?utm_source=chatgpt.com "10 Diaphragm"
[6]: https://www.kenhub.com/en/library/anatomy/mediastinum?utm_source=chatgpt.com "Mediastinum: Definition, anatomy, borders and contents"
[7]: https://www.ncbi.nlm.nih.gov/books/NBK519048/?utm_source=chatgpt.com "Anatomy, Thorax, Lung Pleura And Mediastinum - StatPearls"
[8]: https://my.clevelandclinic.org/health/body/24113-mediastinum?utm_source=chatgpt.com "What is the Mediastinum?"
[9]: https://radiopaedia.org/articles/pericardium?lang=us&utm_source=chatgpt.com "Pericardium | Radiology Reference Article"
[10]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9431844/?utm_source=chatgpt.com "Pericardial Recess: Computed Tomography Findings ..."
[11]: https://radiopaedia.org/articles/superior-aortic-recess?lang=us&utm_source=chatgpt.com "Superior aortic recess | Radiology Reference Article"
[12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12092311/?utm_source=chatgpt.com "Intra-Individual Reproducibility of Automated Abdominal ..."
[13]: https://github.com/StanfordMIMI/TotalSegmentatorV2?utm_source=chatgpt.com "StanfordMIMI/TotalSegmentatorV2"
