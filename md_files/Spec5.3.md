何が起きているか（原因の整理）

腹部臓器の“禁止帯”が接触判定より広い
v5.2 では ILAM のドメインを作る段階で

abdomen_forbidden = iso_dilate(abdominal_organs, abdomen_gap_mm, spacing)  # 既定 3 mm
domain = (fat_for_ilam & (~lungs) & ~abdomen_forbidden)


としています。一方、臓器接触量は

lungs_touch    = iso_dilate(lungs_for_touch,   contact_eps_mm, spacing)  # 既定 2 mm
stomach_touch  = iso_dilate(stomach_mask,      contact_eps_mm, spacing)
...


で 2 mm の膨張帯を使って数えています。
この順序と設定だと、胃（や肝・脾）から 3 mm 以内の脂肪は最初から domain から消えてしまうため、後段の「接触数を数える」処理に到達するころには n_stomach = 0 になります。
→ 胃には強く隣接しているのに 3–5 mm 離れている脂肪塊は “腹部臓器接触 0” と判定され、肺接触があれば Lung-dominant で採用 されます。画像に見える薄い帯は abdomen_gap_mm（3 mm）で削ったリングそのものです。

--no-z-limit により下方（横隔膜下）でもシードが発生
v5.2 では Z 制限を外したため、肺の最下端（横隔膜の小孔や segmentation のスリット）近傍で 肺膨張帯 lungs_touch が腹側へ漏れ、腸間膜脂肪が seed を持つようになりました。
横隔膜穴塞ぎ（seal_diaphragm_holes_mm = 1.0）は binary closing で 1–2 voxel 相当しか埋めないので（binary closing は小孔を埋めるが、閉じるサイズは構造要素と反復回数に依存します）、大きめの欠損は塞げません。

“肺接触優勢”判定が腹部臓器禁止帯により偏る
dominant contact では
n_lung >= lung_dom_ratio * max(n_stomach, n_liver, n_spleen) を採用条件にしていますが、(1)の理由で n_stomach 等が 0 になりやすく、実質的に n_lung の下限チェックだけになってしまっています。ログの
Accepted (lung 29241, stomach 0, liver 0, spleen 0) はこの帰結です。

参考：EDT の sampling で mm 単位の等方膨張を使っている点は正しい実装です（画素間隔を渡すことで物理距離ベースになる）。
連結成分抽出に scipy.ndimage.label を用いて塊ごとに採用している方針も妥当です。
（背景）EAT は心外膜（臓側心膜）内、PAT は壁側心膜外で縦隔脂肪に相当するという定義。

修正方針（v5.3 の提案）
A. 接触判定と禁止帯の順序を入れ替える（最重要）

ラベル付けと接触カウントは “禁止帯なし” の Domain で行う。

domain_cc = (fat_for_ilam & (~lungs))           # 心臓距離制約は可
lbl, n = label(domain_cc, structure=conn)

# 接触帯（± contact_eps_mm）
lungs_touch   = iso_dilate(lungs_for_touch, contact_eps_mm, spacing)
stomach_touch = iso_dilate(stomach_mask,    contact_eps_mm, spacing)
liver_touch   = iso_dilate(liver_mask,      contact_eps_mm, spacing)
spleen_touch  = iso_dilate(spleen_mask,     contact_eps_mm, spacing)

# 各成分の n_lung, n_stomach, n_liver, n_spleen を計算


採用判定（dominant contact） はいまの条件のままで OK。
ただし min_lung_touch_voxels と lung_dom_ratio は場面に応じて調整（推奨値は下に記載）。

採用後の整形 として、はじめて 薄い“安全帯”を差し引き ます（post-trim）。

post_trim_mm = min(abdomen_gap_mm, contact_eps_mm)  # 推奨: 1.0–1.5 mm
accepted &= ~iso_dilate(abdominal_organs, post_trim_mm, spacing)


こうすると「接触の判定には使うが、最終マスクは臓器から 1–1.5 mm 離す」挙動になります。
※ いまの v5.2 のように「禁止帯 > 接触帯」にしてしまうと、臓器接触カウントが常に 0 になって判定が片寄るのが問題の本質です。

ガード：if abdomen_gap_mm >= contact_eps_mm: warn(...) のように設定矛盾を検出してログに警告を出すのがおすすめ。

B. シードだけ Z 範囲で制限し、伝播は 3D 全域を許可（“ソフトな Z 制御”）

--no-z-limit を使うと下腹部側にも seed が立ってしまいます。
seed 用の Z バンド（seed_z_margin_mm）を導入し、

seed_z_mask[:, :, z_min - m : z_max + m] = True   # m=10 mm など
seed = (domain_cc & lungs_touch & seed_z_mask)


として シードの発生位置だけ 心臓の上下 ±10 mm 程度に制限します。
これなら “下で突然切れる” 現象は避けつつ、シードに連なる塊は Z をまたいで丸ごと採用されます。

C. 心膜近傍との連結性を採用条件に追加（任意だが効果大）

採用予定の成分 comp が 心膜周囲（Shell）に触れているか を条件にします：

shell_near = iso_dilate(shell_mask, 2.0, spacing)  # Shell 2 mm 膨張
touches_shell = np.any( (lbl == cid) & shell_near )


touches_shell が False の成分（横隔膜下で肺とだけ隣接している腸間膜塊など）は除外。
“肺に触れていて、かつ心臓殻にも触れている脂肪だけ” を ILAM に残せます。
これで横隔膜の小孔越しの誤結合も抑制できます。

D. 横隔膜の小孔をもう少し確実に塞ぐ

seal_diaphragm_holes_mm の 既定値を 2–3 mm に。
SciPy の binary closing は“小さな穴を埋める” 作用なので、構造要素と反復回数が不足すると抜けます。

影響を胸郭下部に限定したい場合は Z バンド（z_min−5 mm 〜 z_min+10 mm）だけに closing を適用すると副作用が減ります。

最小変更で効くコード差分（概略）
# 1) ラベリング用 domain は腹部禁止帯を外す
domain_cc = (fat_for_ilam & (~lungs))
if max_heart_dist_mm is not None:
    domain_cc &= (dist_to_heart <= max_heart_dist_mm)

# 2) seed を心臓付近 Z のみに
seed_z_margin_mm = 10.0
m = int(np.ceil(seed_z_margin_mm / spacing[2]))
z_seed = np.zeros_like(domain_cc, dtype=bool)
z_seed[:, :, max(0, z_min - m) : min(z_max + m + 1, domain_cc.shape[2])] = True
seed = domain_cc & lungs_touch & z_seed

# 3) 連結成分・接触数（禁止帯なしで計測）
lbl, n = label(domain_cc, structure=generate_binary_structure(3, 2))
def contact_count(cid, touch_mask):
    return int(np.sum((lbl == cid) & touch_mask))

keep_ids = []
for cid in np.unique(lbl[seed])[1:]:
    n_lung    = contact_count(cid, lungs_touch)
    n_stom    = contact_count(cid, stomach_touch)
    n_liver   = contact_count(cid, liver_touch)
    n_spleen  = contact_count(cid, spleen_touch)
    abdmax    = max(n_stom, n_liver, n_spleen)
    if n_lung < min_lung_touch_voxels:          continue
    if n_lung < lung_dom_ratio * abdmax:        continue
    if not touches_shell(lbl, cid, shell_mask): continue  # 任意（推奨）
    keep_ids.append(cid)

ilam_components = np.isin(lbl, keep_ids)

# 4) 最終整形としてのみ薄い安全帯を差し引く（post-trim）
post_trim_mm = min(abdomen_gap_mm, contact_eps_mm)  # 例: 1.0–1.5 mm
accepted = ilam_components & ~iso_dilate(abdominal_organs, post_trim_mm, spacing)
pericardial_fat_mask = eat_pat_shell | accepted


パラメータの推奨例（今回ケース）
contact_eps_mm=3.0, post_trim_mm=1.0–1.5, lung_dom_ratio=1.5,
min_lung_touch_voxels=100–150, seal_diaphragm_holes_mm=2.0,
max_heart_dist_mm=25–30, seed_z_margin_mm=10.

なぜこれで直るか

判定の公平性：腹部禁止帯の適用を後ろに回すため、胃に優勢に接している塊は abdmax が大になり、Lung-dominant を満たせず不採用になります。
その上で薄い post-trim を差し引くので、最終マスクは臓器から 1–1.5 mm 離れた滑らかな境界になります。

3D 連続性：ラベリングは全 Z で行う一方、seed を心臓近傍 Z に限定するため、心臓近傍で肺に触れた塊は塊ごと採用され、逆に 遠位で偶然肺に触れた腸間膜塊はシードが無いので採用されません。

解剖学的整合：Shell 近傍との接触要件を入れることで、心膜周囲脂肪（EAT+PAT）の概念に近い幾何拘束が加わります（EAT は心外膜内、PAT は心膜外の縦隔脂肪）。

参考（用いている処理の根拠）

EDT による mm 等方膨張：distance_transform_edt(..., sampling=spacing) を使えば体軸の画素間隔を考慮した物理距離ベースの膨張ができます。

連結成分ラベリングで 3D の“塊”を一括扱い：scipy.ndimage.label（26 近傍）で医用画像の体積塊を抽出する標準手法。

binary closing は小孔（横隔膜マスクのピンホール等）を埋める処理。閉じる大きさは構造要素・反復回数依存。

追加の実装チェックリスト

 ログに abdomen_gap_mm と contact_eps_mm の関係を出力し、abdomen_gap_mm >= contact_eps_mm のときは警告。

 Dominant contact の結果で “Stomach dominant / Liver dominant で棄却” 件数を必ず表示（現状 0 が続くなら要見直し）。

 --require-shell-touch（既定 ON）と --seed-z-margin-mm を CLI に追加。

 --seal-diaphragm-holes-mm は mm→ボクセル変換の反復数をログに出す（closing はボクセル反復であるため）。

 評価用に accepted-before-posttrim と accepted-after-posttrim の 2 つの診断 NIfTI を保存し、差分を確認。

必要であれば、上の差分をあなたの v5.2 にそのまま入れ込んだ v5.3 スケルトンも提示できます。今回のポイントは「禁止帯は“最終整形”で使う、判定は未加工のドメインで行う」の一点に尽きます。これで、胃に接する腸間膜脂肪の混入は抑えつつ、肺接触により裏打ちされた 胸腔側（心臓近傍）の脂肪塊を 3D で丸ごと拾えるようになります。