以下、**結論→背景（文献の定義）→いまの実装の評価→実装改善の具体案→検証のポイント**の順でまとめます。
（要点だけ先に）**結論**：文献で広く用いられている体積評価は、**上限＝右肺動脈（RPA）レベル、下限＝横隔膜**というZ軸の範囲を採るものが主流です。一方で、**上限＝肺動脈分岐、下限＝左室心尖**までに留めるプロトコルも少なくありません。現行コードの「IVC/肺静脈ベースで決めるz\_min」は、典型的な「横隔膜まで」をやや\*\*頭側寄り（= 広く使われる下限より浅い）\*\*に設定する傾向があり、**心横隔膜角（cardiophrenic angles）付近のPAT（パラカード脂肪）を一部取り逃す**リスクがあります。**再現性（他研究との互換性）を重視するなら、“RPA→横隔膜”モードを用意**して使い分けるのが無難です。([PMC][1], [BioMed Central][2], [Global Heart][3], [e-kcj.org][4])

---

## 背景：研究で使われているZ軸（尾側）定義

* **主流（大規模コホートや多くの近年研究）**

  * **上限：右肺動脈（mid‑level of RPA）／下限：横隔膜**。ROMICAT II などでは「**RPAを起点に横隔膜まで**」で\*\*EAT（心膜内）**と**PAT（心膜外）\*\*を測定しています。Framingham 系や近年の放射線・呼吸器領域の研究、総説・原著でもこのレンジが繰り返し採用されています。([PMC][1], [BioMed Central][2], [Global Heart][3], [Revista Portuguesa de Cardiologia][5])

* **別系統（しばしば撮影視野が心臓中心）**

  * **上限：肺動脈分岐／下限：左室心尖（apex）**。この設定は**横隔膜直上の心横隔膜角に溜まりやすい脂肪を部分的に除外**しがちで、**体積は小さめ**に出ます（方法論として明記している研究が複数あり）。([PMC][6], [Oxford Academic][7])

* **命名・区分の注意**

  * **EAT**=心膜**内**脂肪、**PAT**=心膜**外**（縦隔）脂肪、**Pericardial fat**という語は**EATと同義に使う場合と、EAT+PATの合算を指す場合の両方がある**—という混乱が古くから指摘されています。比較研究では**用語とZ範囲の定義**を必ず明記・整合させる必要があります。([PMC][8])

---

## 現行アルゴリズム（提示コード）における尾側Z設定の評価

* **z\_min（尾側）を「IVC/肺静脈/門脈」の“心臓近接”で決める**実装は、**心臓近傍に限定する**という意味では理にかなっていますが、文献の主流である\*\*「横隔膜まで」**より**浅めに切り上げやすい**です。とくに**cardiophrenic angles の脂肪（典型的PAT）**は**RPA→横隔膜**レンジではROIに含まれますが、現行のz\_minだと**除外されやすい\*\*構造です。([PMC][1], [Global Heart][3])

* さらに、コードでは**ZクリップをShellとfatに先に適用**し、その後ILAMで**下部帯（inferior\_band\_mm=50 mm）付近を重視**して救済しています。ただし**種（seed）Z範囲は z\_min〜z\_max±margin**に広がるため、**実質的に“横隔膜基準の網羅性”を完全に担保はしない**設計です。
  → \*\*再現性（方法の互換性）\*\*を最優先するなら、**明示的に“横隔膜まで”**を下限としてクリップする**モード**を持たせるのが安全です。([PMC][1])

---

## 改善案（研究定義と互換にするための実装オプション）

**A. “RPA→横隔膜”モードを追加（推奨）**

1. **尾側（下限）= 横隔膜推定**

   * TotalSegmentatorに**横隔膜セグメントは標準で出ない**ため、**肺マスクから横隔膜ドームを近似**します：

     * 既に実装済みの**binary\_closing（seal\_diaphragm\_holes\_mm）**で**肺下縁の小孔を閉鎖**。
     * **心臓のXY外接矩形±数cm**の範囲に限定して、\*\*肺が存在する最尾側スライス（z\_lung\_min）**を求め、**その1–2スライス下**を**横隔膜近似（z\_dia）\*\*とします（呼吸相や体位のバラツキを考え小さめマージンを推奨）。
     * **z\_min = z\_dia**として**Shell/fatのZクリップ**に用います。
   * これは多くの研究が採る\*\*「横隔膜まで」\*\*に最も近い機械的近似です。([PMC][1], [BioMed Central][2])

2. **頭側（上限）= RPA近傍**

   * 既存の**compute\_intrapericardial\_zmax**は**大血管近接**（大動脈・肺動脈）で決めていますが、**RPAスライスを明示**したい場合は、**肺動脈分岐レベル**を**肺動脈マスクと主気管支分岐の相対位置**等で近似して固定できるようにしておくと、**研究文献の“RPA起点”に揃えやすい**です。([PMC][1], [e-kcj.org][4])

> 参考実装イメージ（擬似コード）
>
> ```python
> # 1) 横隔膜近似（尾側）
> lungs_closed = binary_closing(lungs, structure=struct, iterations=iter_for_mm(seal_mm))
> x0,x1,y0,y1 = bbox_of(heart_mask, pad_mm=30, spacing=spacing)  # 心臓周囲にXYを限定
> z_has_lung = np.any(lungs_closed[x0:x1, y0:y1, :], axis=(0,1))
> z_lung_min = np.where(z_has_lung)[0][0]              # 肺が最初に出現する=最も尾側
> z_diaphragm = max(0, z_lung_min-1)                   # 1スライス下に余裕
> z_min = z_diaphragm
>
> # 2) RPA近似（頭側）は、既存z_maxを流用しつつ、
> #    optional: 肺動脈分岐の高さを検出できればそれを優先
> ```
>
> *（概念の提示。実コード化の際は貴環境の軸方向定義に合わせて極性確認が必要です）*

**B. “肺動脈分岐→心尖”モードも併設**

* **左室心尖でクリップ**するモードも残しておくと、**apex下限を採る研究**（心臓CTで撮影視野が狭い場合など）との比較が容易です。**左室心尖の推定**は、**心臓マスクの最尾側スライス**（心尖側）＋小マージンで代替できます。([PMC][6])

**C. モード選択の公開パラメータ化（例）**

* `--z-range-mode {rpa_to_diaphragm, rpa_to_apex, current_prox_vessel}`（既定は `rpa_to_diaphragm` を推奨）
* レポートJSONに**選択モードとz範囲**を必ず記録（論文比較の再現性向上）。

---

## 影響と検証ポイント

* **体積の系統差**

  * **横隔膜まで含める**と、**apex止まり**より**EAT/PAT体積は大きく出る**傾向があります（心横隔膜角・心基部下の脂肪を取り込めるため）。研究間比較では**同一レンジを必ず担保**してください。([PMC][6])
* **用語と領域の整合**

  * **EAT（心膜内）**と**PAT（心膜外）**を合算して**Pericardial fat**と呼ぶ論文／EATと同義で呼ぶ論文が混在します。\*\*用語・Z範囲・HU閾値（多くは-190〜-30HU）\*\*の3点セットで併記すると誤解を避けられます。([PMC][8])
* **既存コードとの整合**

  * いまの**ILAM**は**下部帯（inferior\_band）で肺優勢接触を使って救済**しますが、**Zクリップ自体が浅い**と**救済の“母集団”が減る**ため、まず**Z下限を適切に**してからILAM閾値（`lung_dom_ratio`, `min_lung_touch_voxels`など）を微調整するのが筋です。

---

## まとめ

* **広く使われる定義**は\*\*「RPA→横隔膜」**（EAT/PATをこの範囲で評価）で、**“apex止まり”**の流儀も**少数派ながら実在\*\*します。([PMC][1], [BioMed Central][2])
* 現行の**z\_min（IVC/PV/門脈近接）**は**横隔膜基準より浅く**なりやすく、**PATの一部を過小評価**するおそれがあります。**研究比較の再現性を重視**するなら、**“RPA→横隔膜”モード**の導入を推奨します。([PMC][1])

---

### 参考（代表的な記述の原文要旨）

* **「RPAから横隔膜まで」**で**心膜輪郭をトレース**してEAT/PAT体積を算出（ROMICAT II など、多数）｡ ([PMC][1])
* **Framingham**などでも**右肺動脈〜横隔膜**を胸腔内脂肪の**解剖学的レンジ**として用いる方式が示される。
* \*\*「肺動脈分岐〜心尖」\*\*も採用例あり（apex基準）。([PMC][6])
* **EAT＝心膜内／PAT＝心膜外**だが**Pericardial fat**の語は**EAT**または**EAT+PAT**で使われ方が揺れるので注意。([PMC][8])

必要なら、上記Aの\*\*“横隔膜近似”コード片\*\*を実際の軸極性（inferior/superiorのz方向）に合わせて具体化してお渡しできます。

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4983201/ "
            Epicardial and Paracardial Adipose Tissue Volume and Attenuation – Association with High-Risk Coronary Plaque on Computed Tomographic Angiography in the ROMICAT II Trial - PMC
        "
[2]: https://respiratory-research.biomedcentral.com/articles/10.1186/s12931-024-02977-x?utm_source=chatgpt.com "Radiomics parameters of epicardial adipose tissue predict ..."
[3]: https://globalheartjournal.com/articles/10.5334/gh.1078?utm_source=chatgpt.com "Association of Periaortic Fat and Abdominal Visceral Fat ..."
[4]: https://e-kcj.org/pdf/10.4070/kcj.2010.40.7.328?utm_source=chatgpt.com "Assessment of Epicardial Fat Volume With Threshold- ..."
[5]: https://www.revportcardiol.org/pt-influence-epicardial-adipose-tissue-in-articulo-S0870255120304157?utm_source=chatgpt.com "Influence of EPICardial adipose tissue in HEART diseases ..."
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9981370/?utm_source=chatgpt.com "Epicardial fat density, coronary artery disease and ..."
[7]: https://academic.oup.com/eurheartj/article-pdf/doi/10.1093/eurheartj/ehae666.1464/60042750/ehae666.1464.pdf?utm_source=chatgpt.com "Impact of systolic blood pressure on epicardial adipose ..."
[8]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3998169/ "
            Epicardial Fat: Definition, Measurements and Systematic Review of Main Outcomes - PMC
        "
