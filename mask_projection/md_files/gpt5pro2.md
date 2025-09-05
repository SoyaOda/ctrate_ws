結論から先に——**DiffDRRはデフォルトで Beer–Lambert の“指数（exp）”をかけません**。
DiffDRR の DRR は **線積分（line integral）** を直接返します（サンプルをレイに沿って **sum** するだけ）。公式ドキュメントでも、レンダリング方程式は

$$
E(R)=\|p-s\|\int_0^1 V(s+\alpha(p-s))\,d\alpha
$$

と明記されており、指数は登場しません（既に $\int \mu\,ds$ そのもの）。デフォルトの実装パラメータも `reducefn='sum'` で「足し算」になっています。([Vivek Gopalakrishnan][1])

なので「Beer–Lambert の影響を削除できない」という現象は、次のどれかが原因であることがほとんどです：

---

## よくある原因と対処

### 1) 可視化や後処理で `exp()` / `-log()` をかけている

* 研究メモや可視化ユーティリティのどこかで `torch.exp(-img)` や `-torch.log(img)` を適用していると、**指数形の見え方**になってしまいます。
* **対処**：DiffDRR の `DRR(...)` が返す `img` をそのまま使ってください（必要なら 0–1 正規化だけ）。**指数変換は不要**です。レンダラ自体が線積分を返しているためです。([Vivek Gopalakrishnan][2])

### 2) `renderer` / `reducefn` の選択ミス

* `renderer='siddon'`（既定）＋ `reducefn='sum'` が**純粋な線積分**です。`reducefn='mean'` や独自 reduce で正規化が入ると「厚みと完全比例」からズレます。
* **対処**：`DRR(..., renderer='siddon', reducefn='sum')` を明示し、`n_points` などのパラメータも既定のままに。([Vivek Gopalakrishnan][1])

### 3) 体積値が「マスクの1/0」ではなく「CT密度（HU→擬似減弱係数）」になっている

* CT そのものを入力にすると、線積分は **厚み**ではなく**減弱係数の積算**になります（当然、骨が強調）。
* **対処（厚みを相関1で得たい場合）**：

  * 3D マスク（0/1）だけを**体積値 V**としてレンダリングする（V=1 内部、0 外部）。このとき DiffDRR の線積分は**ボクセル長に比例**するので、「画素値＝そのレイに沿った**マスク厚み**」になります。
  * 既存の CT と併用して「マスクの寄与を別チャネルで」欲しいときは `mask_to_channels=True` を使う（ただしこれは**構造ごとの寄与**であって **厚み=1/0 の厳密積算**にしたいなら“V=1のマスク体積”で一枚レンダ**リング**するのが確実）。([Vivek Gopalakrishnan][2])

### 4) Radiographic view（左右規約）の反転で「指数っぽい濃淡の違い」に見える

* DiffDRR は既定で `reverse_x_axis=True`（**心陰影が画像右**＝放射線科の標準表示）を守るため、DeepDRR と見た目が左右入れ替わることがあります。**物理量は同じ**ですが、ヒストグラムの見え方が変わって誤解されがちです。
* **対処**：左右を合わせたいなら `reverse_x_axis` の設定とポストフリップを統一（DeepDRR 互換は公式チュートリアルに記述あり）。([Vivek Gopalakrishnan][3])

---

## 「マスク厚み（相関=1）」を返す最小コード（DiffDRR）

```python
import torch
from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import convert  # 任意。EulerなどをSE(3)に変換

# --- 1) 0/1 マスクを体積として読む（厚み専用の“密度=1”体積）
#      volume=mask_nifti だけを read() に渡すのが手っ取り早い
subject = read(
    volume="mask_01.nii.gz",    # 0/1 の三次元マスク（1=目的部位）
    orientation="PA",           # radiographic view 規約を使うなら "PA" or "AP"
    center_volume=True          # ワールド原点に移す（既定）
)

# --- 2) DRR: 線積分（siddon + sum）。pixel ピッチや SDD は DeepDRR と一致させる
drr = DRR(
    subject,
    sdd=1800.0,        # mm
    height=3072,       # 例: 43cm / 0.14mm
    delx=0.14,         # mm/px (dely省略=>delxに揃う)
    width=2500,
    renderer="siddon",
    reverse_x_axis=True,  # 放射線科の左右規約（心陰影が右）
    reducefn="sum"
).to("cuda")

# --- 3) ポーズ（DeepDRRのPA/LATに揃える）
#     ここでは PA 正面 (yaw=0, pitch=0, roll=0) 例
pose = convert(
    torch.tensor([[0., 0., 0.]], device="cuda"),        # 回転（オイラー等）
    torch.tensor([[0., 900., 0.]], device="cuda"),      # 並進 (by = SAD)
    parameterization="euler_angles",
    convention="ZXY"
)

# --- 4) 厚み画像（mmに比例）を得る：Beer–Lambert の指数なし
thickness = drr(pose)            # shape: [B=1, C=1, H, W], 線積分 = 厚み
thickness = thickness.squeeze().cpu().numpy()
```

* ここで得られる `thickness` は **各ピクセルのレイに沿った 1/0 マスクの長さの総和**です。
* 0/1 マスクなので **画素値 \~ 厚み（mm）に厳密比例**（相関係数=1）。
* `mask_to_channels=True` は「CT密度の寄与をラベル別チャンネルに分ける」機能なので、**“厚み”そのものが欲しい**ときは上記のように**マスク単体を体積としてレンダリング**するのがベストです。([Vivek Gopalakrishnan][2])

> DeepDRR と完全に画素対応させるときは、**SDD/ピクセルサイズ/画像サイズ/姿勢の変換**を両者で一致させます。公式「DeepDRR への変換」チュートリアルに、\*\*角度の変換と左右反転（`[:, ::-1]`）\*\*の対応例があります。([Vivek Gopalakrishnan][4])

---

## まとめ

* **DiffDRR は Beer–Lambert の指数を掛けない**（線積分を返す）。→「指数の影響を消す処理」は不要。([Vivek Gopalakrishnan][1])
* **マスク厚みを相関=1で得る**には、0/1 マスクをそのまま体積として `renderer='siddon'`＋`reducefn='sum'` でレンダリング。
* **DeepDRR とピクセル一致**させるには、幾何（SID/SAD/ピクセルサイズ/解像度）と座標系（角度の順序・左右規約）をチュートリアルの通りに合わせる。([Vivek Gopalakrishnan][4])

もし以前の実験で「指数っぽい見え方」だった場合は、**後処理で `exp()` / `-log()` を適用していないか**、あるいは **CTボリュームをそのまま投影**していないかを確認してください。DiffDRR 本体は**指数を掛けない**仕様です。([Vivek Gopalakrishnan][2])

[1]: https://vivekg.dev/DiffDRR/api/renderers.html "renderers – diffdrr"
[2]: https://vivekg.dev/DiffDRR/tutorials/introduction.html "How to use DiffDRR – diffdrr"
[3]: https://vivekg.dev/DiffDRR/api/drr.html "DRR – diffdrr"
[4]: https://vivekg.dev/DiffDRR/tutorials/metamorphasis.html "Converting to DeepDRR – diffdrr"
