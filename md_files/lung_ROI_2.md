とても良い着眼点です。**結論**としては――

* **下部（横隔膜付近）では heart シェルだけに依存せず，縦隔内（\~lungs）に限定したうえで「**肺に接する torso\_fat**」を EAT+PAT 候補に加える**のは妥当です。横隔膜が胸腔と腹腔の境界であり（縦隔の下界も横隔膜）、心膜は心臓と“大血管の根部”を包んで横隔膜中央腱に付着するため，この帯域では胸腔側にある脂肪を拾うことが定義と整合的だからです。([Radiopaedia][1])
* ただし\*\*“肺に接する”を唯一の条件にすると\*\*，上部では胸腺前脂肪や食道周囲脂肪など**心膜と無関係な縦隔脂肪**まで拾う危険があります。よって**下部の限定的な帯域だけで適用**し，**心臓からの距離や腹部臓器への接触**で制御する**ハイブリッド運用**が安全です。([Radiopaedia][1])
* 定義面：EAT は**心外膜（臓側心膜）と心筋の間**，PAT は**壁側心膜の外側**で，**心膜周囲脂肪**はこの両者の総称です。**心横隔角の“pericardial fat pad”はEATとPATの混合**として記載され，今回の下部帯域での拾い上げはこの概念に合います。([PMC][2], [Radiopaedia][3])

---

## 提案：下部限定「肺接触モード（Inferior Lung‑Adjacency Mode; ILAM）」を追加

あなたの v3 アルゴリズムに**切替可能な追加モード**として組み込みます。**切り替えは Z で**行い，それより下では “heart シェルに頼らず” **肺接触ベース**で EAT+PAT を抽出します。

### 1) 解剖学に基づく Z 範囲の決定（既存ロジックの再利用）

* **下端 z\_min**：`heart` の EDT で ≤5 mm 近接帯をつくり，**IVC** と **pulmonary\_vein** の**近位部が最初に出るスライス**を z\_min（±1）に設定（心膜が大血管根部までを被覆することに整合）。([Radiopaedia][4], [PMC][5])
* **上端 z\_max**：**上行大動脈・肺動脈の近位部**が最後に見えるスライス（±1）で打ち切り（上縦隔への逸脱防止）。([Radiopaedia][6])
* これで得た \[z\_min, z\_max] が**心膜内外の中核範囲**。さらに**z\_switch = z\_min + band\_mm**（例 20–30 mm）を設け，\*\*\[z\_min, z\_switch] を“下部帯域”\*\*として ILAM を適用。上部は従来どおり `shell ∩ torso_fat` を基本とします。**横隔膜が縦隔の下界**という一般解剖と矛盾しません。([Radiopaedia][1])

### 2) 下部帯域での「肺接触による EAT+PAT 候補」C₂

* **縦隔限定**：`(~lungs)` を掛けて胸腔中央に制限。([Radiopaedia][1])
* **肺への“接触”**：

  * 形態学的に **`touch(B, L, δ)` = `(B ∩ dilate(L, δ))`** を採用。
  * δ（mm）を**パラメータ化**（例 0=厳密接触，1–2 mm=横隔膜の薄さを許容）。
* **腹部臓器の除外**：**`liver | spleen | stomach` を γ mm 膨張**して禁止帯（γ=2–3 mm）。横隔膜直下の**腹腔脂肪の混入を抑制**します。([Radiopaedia][1])
* **心臓との関連づけ（過剰除外を防ぎつつ逸脱を抑制）**：**`dist_heart ≤ 25–30 mm`** を課す（パラメータ ρ）。これで**食道/椎体近傍の縦隔脂肪**や**遠位下行大動脈PVAT**の混入を抑制できます。必要に応じて**下行大動脈（aorta）遠位部を2–3 mm膨張で禁止**。([Radiopaedia][4])

**定義式（下部帯域のみ）：**

* `C2 = torso_fat ∩ (~lungs) ∩ touch(torso_fat, lungs, δ) ∩ (dist_heart ≤ ρ) ∩ ~(dilate(liver|spleen|stomach, γ))`
* **最終 EAT+PAT**：`(shell ∩ torso_fat)` **∪** `C2`（「置換」ではなく**加算**を推奨）。

> **なぜ加算（union）か**：cardiophrenic fat pad の一部は**肺に必ずしも密接せず**（スライス位相で肺が離れることがある），**heart シェルに基づく候補 C₁ が拾えている場合もある**ためです。EAT/PAT の解剖（心膜反転や recess）を踏まえると妥当です。([PMC][5], [AJR Online][7])

### 3) 推奨デフォルト（調整可能）

* `band_mm = 25`（z\_min から 25 mm を ILAM 帯域）
* `δ = 1.0–2.0 mm`（肺接触の許容。横隔膜の厚みを越えない）
* `γ = 3.0 mm`（腹部臓器クランプ）
* `ρ = 30 mm`（心臓からの最大距離）
* **遠位大動脈禁止**：`aorta` 近位（dist\_heart ≤5 mm）以外を 2–3 mm 膨張で除外（下行Ao PVATを外す）。([Radiopaedia][4])

---

## 実装パッチ（v3 への差し込みの要所）

```python
# === parameters (CLIから可変に) ===
parser.add_argument("--inferior-band-mm", type=float, default=25.0)
parser.add_argument("--lung-contact-mm", type=float, default=1.5)   # δ
parser.add_argument("--abdomen-gap-mm",  type=float, default=3.0)   # γ
parser.add_argument("--max-heart-dist-mm", type=float, default=30.0) # ρ
parser.add_argument("--replace-by-contact", action="store_true")     # 置換モード（実験用）

# === after z_min/z_max computed ===
band_slices = int(np.ceil(args.inferior_band_mm / min(spacing)))
z_switch = min(z_min + band_slices, z_max)

# utility: mm->iters
def iters(mm): return max(1, int(np.ceil(mm / min(spacing))))

# masks prepared earlier:
# heart_mask, shell_mask, fat_clipped (z-clipped torso_fat), lungs, abdominal_organs

# 下部帯域のスライス選択
sl = slice(z_min, z_switch+1)
lungs_d = binary_dilation(lungs, structure=generate_binary_structure(3,1),
                          iterations=iters(args.lung_contact_mm))
abd_forbid = binary_dilation(abdominal_organs, structure=generate_binary_structure(3,1),
                             iterations=iters(args.abdomen_gap_mm))

# 肺接触候補 C2
dist_to_heart = distance_transform_edt(~heart_mask, sampling=spacing)
near_heart = dist_to_heart <= args.max-heart-dist-mm   # ρ

cand_contact = (fat_clipped & (~lungs) & lungs_d & near_heart) & (~abd_forbid)
cand_contact[:, :, :z_min] = False
cand_contact[:, :, z_switch+1:] = False

# 既存の EAT+PAT（シェル基準）
eat_pat_shell = shell_mask & fat_clipped

# 合成
if args.replace_by_contact:
    eat_pat_final = (eat_pat_shell & (np.arange(shell_mask.shape[2])[None,None,:] > z_switch)) | cand_contact
else:
    eat_pat_final = eat_pat_shell | cand_contact
```

> **可視化/QA**
>
> * `volume(cand_contact)` と `volume(eat_pat_shell ∩ [z_min:z_switch])` の比率をレポート。ILAM適用で**急増 (>×2)** ならフラグ。
> * `lung-contact-mm` を 0, 1, 2 mm でスイープし，**安定範囲**を統計で提示。
> * `eat_pat_final ∩ dilate(aorta_distal,2mm)` の体積比（期待 ≪ 5%）を表示。

---

## この定義の妥当性と限界

* **妥当性**：

  * **縦隔の下界は横隔膜**であり（縦隔＝左右肺の間の胸腔中央区画），**心膜は心臓と大血管根部を包み横隔膜中央腱へ付着**します。下部で**肺に隣接する縦隔脂肪（cardiophrenic fat pad 周辺）は心膜周囲脂肪の一部**と解されます。([Radiopaedia][1])
* **限界／注意**：

  * **呼吸相・体位**で肺底と心横隔角の関係が変わり，**“接触”判定が過敏**になることがあります。δ を 0–2 mmで調整してください。
  * 上部で ILAM を使うと**前縦隔脂肪や食道周囲脂肪**を拾う恐れがあるため，**下部帯域だけ**に限定してください。([Radiopaedia][1])
  * 文献によって EAT/PAT の測り方は多様で（HU域，範囲，アキシャル/3D）**完全な標準は未統一**です。今回の方法は\*\*「心膜＋近位大血管の範囲」**と**「縦隔＝(\~lungs)」\*\*という解剖定義に寄せており，レビュー論文の定義と矛盾しません。([PMC][2])

---

## 参考にした一次情報

* **EAT/PAT の定義**（心外膜内／壁側心膜外，cardiophrenic fat pad は両者の混合）と**測定に関するレビュー**。([PMC][2], [Radiopaedia][8])
* **縦隔の境界（下界＝横隔膜）**，**心膜が“大血管根部”を包む**こと，**上行大動脈周囲の pericardial recess**。([Radiopaedia][1], [PMC][5])
* **TotalSegmentator のクラス（tissue\_types の torso\_fat，total の肺・腹部臓器，heartchambers/vessels の大血管）**。([GitHub][9])

---

### まとめ

* 肝臓出現レベルなど**下部では ILAM（肺接触モード）を追加**し，**heart シェル＋torso\_fat**に\*\*“縦隔内で肺に接する torso\_fat（腹部臓器には非接触，心臓からは一定距離内）”**を**加算\*\*するのが実務的に最も安定です。
* すべて **TotalSegmentator の標準出力だけで実装可能**です（横隔膜ラベルは不要）。
* パラメータ（δ, γ, ρ, band\_mm）を露出し，症例ごとの**偽陽性/偽陰性トレードオフ**をチューニングできる仕様にしておくのがよいです。

[1]: https://radiopaedia.org/articles/mediastinum-1?lang=us&utm_source=chatgpt.com "Mediastinum | Radiology Reference Article"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3998169/?utm_source=chatgpt.com "Epicardial Fat: Definition, Measurements and Systematic ..."
[3]: https://radiopaedia.org/articles/pericardial-fat-pads-2?lang=us&utm_source=chatgpt.com "Pericardial fat pads | Radiology Reference Article"
[4]: https://radiopaedia.org/articles/pericardium?lang=us&utm_source=chatgpt.com "Pericardium | Radiology Reference Article"
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9431844/?utm_source=chatgpt.com "Pericardial Recess: Computed Tomography Findings ..."
[6]: https://radiopaedia.org/articles/superior-aortic-recess?lang=us&utm_source=chatgpt.com "Superior aortic recess | Radiology Reference Article"
[7]: https://ajronline.org/doi/10.2214/ajr.181.4.1811109?utm_source=chatgpt.com "Anatomy of Pericardial Recesses on Multidetector CT"
[8]: https://radiopaedia.org/cases/pericardial-fat-pad-1?lang=us&utm_source=chatgpt.com "Pericardial fat pad | Radiology Case"
[9]: https://github.com/wasserth/TotalSegmentator?utm_source=chatgpt.com "wasserth/TotalSegmentator: Tool for robust segmentation ..."
