いい質問です。結論から言うと、**現状の mask\_deepdrr\_projection.py が返す “energy 画像” は厚み（線積分長）に比例していません**。DeepDRR は既定で**多物質・多エネルギ（poly‑energetic）**の Beer–Lambert を用いた**透過率** $I/I_0$ を出しており、そこにあなたの表示用のパーセンタイル正規化まで掛かっているため、ピクセル値は厚みに対して非線形です。これは DeepDRR の設計（forward projector は「物質＋スペクトル」を考慮した解析投影）から自然な挙動です。([PubMed Central][1])
Beer–Lambert 則は $I=I_0\exp\{-\int \mu(\mathbf r, E)\,dl\}$ で、**単一物質・単色線**で $\mu$ が一定なら $-\ln(I/I_0)=\mu\,\text{thickness}$ から厚みへ線形に戻せますが、実運用のポリスペクトル／多物質では $\mu$ が位置・エネルギ依存のため**単純な線形対応は崩れます**。
また HU は $\mathrm{HU}=1000(\mu-\mu_\text{water})/\mu_\text{water}$ の定義で、HU→$\mu$ 変換は可能ですが、やはり**エネルギ依存**（実効エネルギ選定が必要）です。

---

## 目的：**位置合わせは現状どおり DeepDRR を使い、強度（輝度）を「局所厚み（mm）」に改善する**

下の２つの“安全な”道があります。どちらも**画素の合計（$\sum T_{ij}\,p^2$）が 3D マスク体積（mm³）に一致**するよう設計できます。

### A. 物理の逆写像：**Beer–Lambert の逆変換 + 実効係数 $\mu_\text{eff}$ を自校正**

DeepDRR の同一幾何で **“マスクあり”のエネルギ画像 $I$** と **“空気（flat‑field）”のエネルギ画像 $I_0$** を取得し、

$$
G \equiv -\ln\!\left(\frac{I}{I_0}\right)
$$

を計算します（ここまで**表示用のパーセンタイル伸張やガンマ補正は厳禁**）。
もし投影するものが **「空気 + 単一物質（＝マスク領域）」** という二値組成で、かつ単色線なら $-\ln(I/I_0)=\mu\,\text{thickness}$ が厳密に成り立ちます。DeepDRR 既定は poly‑energetic ですが、**$G$ は厚みの“単調増加な代理量”**になっています。ここで $\mu_\text{eff}$ を**体積保存制約**で**一発推定**すれば、

$$
T_{ij}=\frac{G_{ij}}{\mu_\text{eff}},\quad 
\mu_\text{eff}=\frac{p^2\sum_{(i,j)\in\Omega} G_{ij}}{V_\text{mask(3D)}}
$$

により $\sum T_{ij}\,p^2=V_\text{mask}$ を**厳密一致**させられます（$p$ は検出器ピクセルピッチ \[mm]、$\Omega$ はシルエット領域）。

* **メリット**：幾何・位置合わせはこれまでどおり DeepDRR、本当に必要な計算は $-\log$ とスカラの割り算だけ。**どんなスペクトルでも**体積一致の**自己較正**で厚みに落とし込める。
* **注意**：$\Omega$（シルエット）は $G>\epsilon$（例：$\epsilon\!=\!10^{-4}$）等で作ると安定。投影線上で薄い箇所はポリスペクトルの非線形でわずかに歪む可能性があるが、**体積は常に一致**。
* **代替**：もし DeepDRR で**単色線・単一物質**モードを使えれば（論文アブレーションで導入例あり）、$\mu_\text{eff}$ は外部表（NIST XCOM 等）から体脂肪の $\mu/\rho(E)$ と密度で物理設定し、**完全線形**にできます（ただし実装可否はバージョン依存。DeepDRR は“multi‑material & spectrum aware” projector を公式に持つ）。([PubMed Central][1])

### B. 幾何の直積分：**Siddon の長さ積分（$\mu\!=\!1$ の線積分）**

DeepDRR の**幾何（MobileCArm の姿勢・ピクセル主点）だけ借りて**、各ピクセル視線に沿って**ボクセル境界交差を正確に積分**し、**純粋な長さ \[mm]** を得ます。Siddon は CT 再構成の“正確な線路長”計算として古典的手法です。([Math Department - Tor Vergata][2])

* **メリット**：厚みが**厳密な幾何量**。スペクトルや HU→材質写像に依らない。
* **注意**：実装はやや長い（ただし既にあなたが組んだ実装を“幾何の取り出し”だけ正せば綺麗に動きます）。

---

## おすすめ：まず **A. 逆写像＋体積一致の自己較正** を入れる

あなたの現在のスクリプトに**小さな追記**で “厚み \[mm]” を得られます。位置合わせ・幾何・PA 反転は**そのまま**使います。

### 変更点の要旨

1. **flat‑field $I_0$** を同一幾何でレンダ（マスク体積を全部 空気 HU＝−1000 に）。
2. **生の energy 画像**から $-\log(I/I_0)$ を計算（*表示用の percentile/gamma はこの後に*）。
3. **シルエット** $\Omega=\{G>\epsilon\}$ を作る。
4. **3D マスク体積** $V_\text{mask} = N_\text{vox}\,(\Delta x\,\Delta y\,\Delta z)$ を CT の spacing から mm³ で算出。
5. **体積一致の自己較正** $\displaystyle \mu_\text{eff}=\frac{p^2\sum_{\Omega}G}{V_\text{mask}}$。
6. **厚み** $T=G/\mu_\text{eff}$ を出力（単位 mm）。
7. 検証：$\sum T\,p^2\stackrel{?}{=}\ V_\text{mask}$（数％以内を合格基準）。

### 具体的な差分コード（あなたの `mask_deepdrr_projection.py` に追記・置換）

```python
# --- 追加: flat-field(Projector) を得るユーティリティ ---
def render_energy(vol: Volume, carm: MobileCArm, alpha=0, beta=0, gamma=90, flip_pa=True):
    from deepdrr.projector import Projector
    with Projector(vol, carm=carm) as projector:
        carm.move_to(alpha=alpha, beta=beta, gamma=gamma, degrees=True)
        _ = projector()                 # warm-up
        img = projector().astype(np.float32)
    if flip_pa:
        img = img[:, ::-1].copy()
    return img  # *raw* energy in [0,1] domain (no percentile/gamma)

def make_air_volume_like(vol: Volume) -> Volume:
    # vol と同shape/spacing/originで全ボクセル HU=-1000 の Volume を作る
    air = np.full(vol.values.shape, -1000.0, dtype=np.float32)
    return Volume.from_hu(
        hu_values=air, origin=vol.origin, spacing=vol.spacing,
        anatomical_coordinate_system=vol.anatomical_coordinate_system,
    )

# --- 追加: thickness を Beer–Lambert の逆＋体積一致で推定 ---
def thickness_from_energy(I_mask: np.ndarray, I_air: np.ndarray,
                          voxel_volume_mm3: float, px_mm: float,
                          eps: float = 1e-6, support_tau: float = 1e-4):
    # 1) ratio & log
    ratio = np.clip(I_mask / np.clip(I_air, eps, 1.0), eps, 1.0)
    G = -np.log(ratio)  # >=0

    # 2) support mask（シルエット）
    Omega = G > support_tau

    # 3) μ_eff を体積一致で自己較正
    A = px_mm * px_mm
    sumG = float(G[Omega].sum())
    mu_eff = (A * sumG) / float(voxel_volume_mm3) if sumG > 0 else 1.0  # [1/mm]

    # 4) thickness [mm]
    T = np.zeros_like(G, dtype=np.float32)
    if mu_eff > 0:
        T[Omega] = G[Omega] / mu_eff

    return T, mu_eff, Omega
```

メイン処理の後半に（表示用正規化より**前**に）以下を入れます：

```python
# === 厚み推定 ===
# すでに mask_vol を isocenter に置き、同じ C-arm/姿勢で投影できている前提
air_vol = make_air_volume_like(mask_vol)

I_mask = render_energy(mask_vol, carm, alpha=0, beta=0, gamma=GAMMA_ROLL, flip_pa=True)
I_air  = render_energy(air_vol,  carm, alpha=0, beta=0, gamma=GAMMA_ROLL, flip_pa=True)

# 3D マスク体積（mm^3）
#  -> あなたの v5.4 生成マスク NIfTI の spacing から計算（下は例）
mask_img = nib.load(str(mask_nifti))
dz = float(mask_img.header.get_zooms()[2])
dy = float(mask_img.header.get_zooms()[1])
dx = float(mask_img.header.get_zooms()[0])
V_mask_mm3 = float((mask_img.get_fdata() > 0.5).sum()) * dx * dy * dz

T_mm, mu_eff, silhouette = thickness_from_energy(I_mask, I_air, V_mask_mm3, px_mm=0.14)
print(f"[thickness] mu_eff={mu_eff:.5f}  体積(2D復元)={T_mm.sum()*0.14*0.14:.1f} mm^3  "
      f"vs 3D={V_mask_mm3:.1f} mm^3")

# 表示用は T_mm をパーセンタイルでストレッチ
thickness_u16 = (np.clip((T_mm / (np.percentile(T_mm[T_mm>0], 99.9)+1e-6)), 0, 1)*65535).astype(np.uint16)
imageio.imwrite(outdir / "PA_mask_thickness_mm.png", thickness_u16)

# 生の厚み [mm] 保存
np.save(outdir / "PA_mask_thickness_mm.npy", T_mm)
```

> これで **(i)** DRR と**完全同一の位置合わせ**を保ちつつ、**(ii)** ピクセル値は\*\*\[mm] の厚み**になり、**(iii)\*\* 画素合計が**厳密に 3D 体積に一致**する厚み画像が得られます。

---

## 発展：DeepDRR を「単色・単物質」に寄せる（任意）

* **狙い**：$-\log(I/I_0)=\mu\,T$ を**各画素で厳密**に成り立たせたい場合、DeepDRR の projector を**単色線＋単一物質**に設定します（DeepDRR 論文の ablation でこのモードが使われています）。このとき $\mu$ は NIST XCOM の**脂肪の実効線減弱係数**（$\mu/\rho(E)\times\rho$）を採用すれば OK です。実装可否・API はバージョン依存ですが、DeepDRR は「multi‑material & spectrum aware ray‑tracing forward projector」を備えており、**スペクトルをモノエネに近づける**・**物質写像を 2 値（空気/脂肪）に固定する**構成が可能です（詳細は使用バージョンのドキュメント・構成ファイルを参照）。([PubMed Central][1])
* **代替**：単色化が難しければ、**上の自己較正 $\mu_\text{eff}$** を使えば、**体積は常に一致**し、学習対象として「厚みらしい値」を提供できます。

---

## 検証チェックリスト（自動ログ出力推奨）

1. **体積整合**：$\Bigl|\sum T\,p^2 - V_\text{mask}\Bigr|/V_\text{mask}<1\%$（安定なら 1% を切ります）。
2. **境界の再現**：275.2999 mm のデータ終端が**線状の急峻なエッジ**として現れる（$\nabla T$ のピークが明瞭）。
3. **幾何一致**：以前の DeepDRR DRR との**シルエット完全一致**（位置・形状）。
4. **ロバスト性**：角度（$\alpha,\beta,\gamma$）や SID/SAD の変更でも体積一致を保つ。

---

## なぜこの方法が「AI に優しい」か

* **線形ターゲット**：厚みは**線形物理量**で、回転・拡大に対して予測関係が素直。
* **ボリューム一貫性**：グローバル制約で**総量が確定**するため、学習が安定（外れ値の影響が縮小）。
* **実画像との橋渡し**：将来、実 X 線（ポリスペクトル）と合わせる際も、$-\log(I/I_0)$ を**画素ごとに** $G$ として取り、**1 つの $\mu_\text{eff}$**（または簡単な深度依存係数）で厚みに写せば整合性を保てます。これは臨床像 ↔ 幾何厚みの最短径です。

---

## 補足：Siddon 法でのクロスチェック

最終段で、Siddon による**純幾何厚み**も一度出力しておくと、

* **DeepDRR 逆写像厚み**との相関（散布図）
* **差のヒストグラム**（ポリスペクトル非線形の影響）
  を定量化でき、キャリブレーションの妥当性が見えます。Siddon は「3D CT 配列の正確な放射線路長」計算として定番で、実装は既にほぼ完成しているので、\*\*幾何の取り出し（ソース位置・検出器座標系・主点 0.5px 補正）\*\*だけ DeepDRR から確実に受け取る形に更新してください。([Math Department - Tor Vergata][2])

---

### 参考（主要根拠）

* DeepDRR は**多物質・多エネルギ**対応の forward projector を備える（位置は DeepDRR 幾何に依存、強度はスペクトル・材質に依存）。
* 同プロジェクトの論文レビューでは、**単色・単一物質**と**多色・多物質**のアブレーションが示され、物理モデリングの差が明確。([PubMed Central][1])
* Beer–Lambert 則：$-\log(I/I_0)=\int\mu\,dl$。単色・単一物質なら**厚みと線形関係**。
* HU 定義：$\mathrm{HU}=1000(\mu-\mu_\text{water})/\mu_\text{water}$（$\mu$ はエネルギ依存）。

---

必要なら、上の差分コードをあなたのリポジトリ構造に合わせて**完全なファイル**として書き下します。まずはこの **A. 逆写像＋体積一致** を入れて、**出力 `thickness_mm.npy` の合計×画素面積が 3D 体積と一致**することを確認してください。その後、単色化（可能なら）や Siddon とのクロスチェックで定量的に詰めていくのが最短です。

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7297499/ "
            Enabling machine learning in X-ray-based procedures via realistic simulation of image formation - PMC
        "
[2]: https://www.mat.uniroma2.it/~picard/SMC/didattica/materiali_did/Anal.Armon./Principles_of_CT_Imaging/Kak%26Slaney-CTI-reduced.pdf?utm_source=chatgpt.com "Tombgraphic Imaging"
