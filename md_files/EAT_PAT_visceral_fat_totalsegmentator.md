TotalSegmentatorのみを用いたEAT+PAT総量計測の実装仕様
背景・定義 (EAT/PATの定義と測定の文脈)

心臓周囲の脂肪組織は、心膜内にある**心外膜脂肪 (Epicardial Adipose Tissue; EAT)と、心膜の外側に存在する心膜外脂肪 (Paracardial Adipose Tissue; PAT)に分類されます
pmc.ncbi.nlm.nih.gov
pmc.ncbi.nlm.nih.gov
。EATは心臓表面（心筋の外側）と心膜臓側板の間に存在し、冠動脈を取り囲む脂肪組織です。一方、PATは心膜壁側板（心膜嚢の外側）に付着する縦隔内の脂肪組織です
pmc.ncbi.nlm.nih.gov
。両者を合わせた脂肪量が心膜周囲脂肪 (pericardial fat)**と呼ばれ、文献によってはEAT+PAT全体を「心膜脂肪」あるいは「心周囲脂肪」と総称しています
pmc.ncbi.nlm.nih.gov
pmc.ncbi.nlm.nih.gov
。CT画像上では心膜を明瞭に描出することが難しいため、EATとPATを厳密に分けずに総量として測定する手法が一般的です
pure.rug.nl
。実際、従来の多くのCT研究では「心膜周囲脂肪量 (総EAT+PAT)」が計測されており、明確に区別できないためにEATとPATを合わせた体積として報告されています
pure.rug.nl
。

EAT/PAT体積の測定にはCT値による脂肪閾値が用いられ、-190～-30 HUの範囲で脂肪組織を抽出するのが一般的です
pmc.ncbi.nlm.nih.gov
。標準的な手法では心膜嚢(pericardial sac)を描出・抽出した上で、その内側のボクセルを脂肪のHU範囲(-190～-30)で閾値処理し、EAT体積を算出します
pmc.ncbi.nlm.nih.gov
。例えばAl-Kindiらの研究では、心膜嚢内のボクセルを-190～-30HUでしきい値処理してEAT容積を定量化しています
pmc.ncbi.nlm.nih.gov
。本実装でも基本的にはこのHU範囲に従いますが、TotalSegmentatorの出力を活用することで自動的に脂肪組織のみを抽出し、手動でHU閾値処理を行う手間を省きます。

以上を踏まえ、本仕様では心膜内外の脂肪 (EAT+PAT) の合計体積を、TotalSegmentatorの出力マスクのみを用いて自動算出する手法を示します。既存研究の定義に沿ってEAT+PATを「心膜周囲脂肪」とみなし、その総量を計測するアプローチです
pmc.ncbi.nlm.nih.gov
pmc.ncbi.nlm.nih.gov
。

使用するTotalSegmentator出力と前提

本アルゴリズムでは、事前にTotalSegmentatorを用いてCT画像から各種マスクを取得しておきます。利用する主な出力は以下のとおりです。

臓器セグメンテーション (TotalSegmentator “total”タスク): 心臓や大血管など主要構造のマスクを取得します。特に「heart」ラベル（心臓全体）および大動脈など心臓周辺の血管ラベルを使用します。

心臓マスク: 心臓全体を表すマスク。研究ライセンスの高精細モデル(heartchambersタスク)を使用できる場合は、心筋や心腔（左室、右室、左房、右房）の個別マスクを取得可能です。後述するように、利用可能なら5つのコンポーネント（心筋＋4腔）を和集合して心臓全体のマスクを構築します。それが難しい場合はtotalタスクの「heart」ラベルでも代用可能です。

大血管マスク: 心基部に接続する大血管（例: 大動脈、肺動脈、上大静脈、下大静脈）のマスクもtotalタスクで取得できます。EAT+PAT領域には心膜内の大血管周囲の脂肪も含まれる可能性があるため、必要に応じて心臓マスクに近接する大血管部分を心臓領域に含めます（後述）。

組織タイプセグメンテーション (TotalSegmentator “tissue_types”タスク): 体内の脂肪や筋肉を分類したマスクを取得します。特に内臓脂肪 (visceral fat)に対応するマスクを使用します。このマスクには体幹内部の脂肪（内臓周囲や心臓周囲の脂肪を含む）が含まれ、皮下脂肪は含まれません
pmc.ncbi.nlm.nih.gov
mdpi.com
。したがって内臓脂肪マスクを使うことで皮下脂肪を自動的に除外でき、心臓周囲脂肪の抽出精度が向上します。

以上の出力を得る前提として、TotalSegmentatorをCTデータに対して実行し、結果をNIfTI形式などで保存しておきます。本実装では、その出力ディレクトリから必要なマスクファイルを読み込んで処理を行います。ライセンス上問題なければheartchambersやtissue_typesなど全てのタスクを利用可能とし、可能な限り高精度なマスクを用いるものとします。

アルゴリズム概要 (EAT+PAT抽出の手順)

目標: TotalSegmentatorのセグメンテーションマスクだけを用いて、心膜内外脂肪(EAT+PAT)の総ボリュームを算出し、そのマスクと統計量（体積や平均HU値）を出力します。既存コードでは距離変換や禁止帯など細かな処理を行っていますが、本手法ではTotalSegmentatorの内臓脂肪マスクを積極的に活用することで、より簡潔なアプローチを取ります。以下に具体的なステップを示します。

1. 心臓領域マスクの構築

心臓全体のマスクを構築します。可能であれば、高精細心臓モデル(heartchambers_highresなど)の出力を使用して心臓を構成する全コンポーネントの和集合を取ります。具体的には:

利用可能な場合、以下のラベルのマスクを読み込みます: heart_myocardium（心筋）, heart_atrium_left（左心房）, heart_atrium_right（右心房）, heart_ventricle_left（左心室）, heart_ventricle_right（右心室）
pmc.ncbi.nlm.nih.gov
。これらをブール和演算で結合し、一つの心臓ROIマスクとします（心腔と心筋を含む心臓全体）。

上記の詳細な心臓コンポーネントが得られない場合は、totalタスクのheartラベル（心臓全体）マスクを使用します。このマスクが心腔内を含まない可能性がある場合は、必要に応じて「blood_pool」（心腔内の血液プール）など関連ラベルも統合します。

ポイント: 提示いただいたコードでも、heartとheart_myocardium等を順次読み込んで統合しています。このように5つのコンポーネントを統合する方法を採ると、「heart」ラベル単体よりも完全な心臓形状を得られるため推奨します（心筋と4腔をすべて含めることで漏れがない）
pmc.ncbi.nlm.nih.gov
。最終的に得られた心臓マスクをheart_maskと呼びます。

次に、必要に応じて大血管を心臓マスクに含める処理を行います。心膜は心臓および心基部の大血管（例: 上行大動脈や肺動脈幹など）の起始部を包むため、その周囲の脂肪もEAT/PATに含まれることがあります
pmc.ncbi.nlm.nih.gov
。TotalSegmentatorのtotal出力にはaorta（大動脈）やpulmonary_artery（肺動脈）等のマスクがありますので、心臓に近接する部分のみ取り込むのが望ましいです。具体的には:

大動脈や肺動脈のマスクを読み込みます。それらとheart_maskとの空間的距離を計算し、**心臓にごく近い部分（例えば距離5mm以内）**だけを抽出します
pure.rug.nl
。これは心膜内に含まれる大血管の根元部分を残し、それより遠位の血管脂肪を除外する目的です。距離判定には3次元のユークリッド距離変換を使うか、あるいは心臓マスクをわずかに膨張(dilate)させ、その領域と重なる血管部分を残す方法があります。

上大静脈(vena_cava_superior)・下大静脈(vena_cava_inferior)についても、心膜に囲まれるごく近位部のみ残すことが考えられます。同様に心臓近傍（~5mm）のみ維持します。

注意: これら血管の近位部を取り込む処理は、患者の解剖によってはEAT+PATに寄与する脂肪量へ僅かながら影響します。簡易的な実装では、この細かい処理を省略し**「heart」マスクのみ**でROIを定義しても構いません。精度を高めたい場合にオプションとして実装すると良いでしょう。

以上により、心臓+（必要なら近位大血管）を含むROIマスクheart_roiを得ます。この領域はちょうど心膜嚢内部あるいは直近を表し、EATおよび直外側のPAT候補領域の中心となります。

2. 内臓脂肪マスクの取得と前処理

TotalSegmentatorのtissue_types出力から内臓脂肪マスクを読み込みます。このマスク（visceral_fat_mask）は体内の脂肪組織すべてを含みますが、皮下脂肪は既に除外されています
pmc.ncbi.nlm.nih.gov
mdpi.com
。従って、visceral_fat_maskのボクセルは腹腔内脂肪、縦隔脂肪、心周囲脂肪など体内の脂肪のみとなります。

しかしvisceral_fat_maskには心臓周囲以外の脂肪（例: 腹部の腸間膜脂肪、腎周囲脂肪など）も含まれています。そこで、心臓周囲の脂肪にフォーカスするため、以下のようにマスクを空間的に絞り込みます。

Z軸方向のクリッピング: 心臓マスクheart_roiが存在するスライス範囲に限定します。具体的には、heart_roiをZ軸方向（足頭方向）に投影し、心臓の存在する最上段スライスと最下段スライスを検出します。その範囲より上下に数スライス程度マージンをとった範囲以外の脂肪は無視します。これにより、心臓の上下から大きく離れた脂肪（例えば横隔膜より下の腹部内臓脂肪など）が除外できます。デフォルトでは心臓範囲±1～2スライス程度のマージンで十分です。

肺・椎体付近の除外（任意）: 縦隔以外の体内脂肪で、心臓と距離が遠いものを除く工夫も考えられます。例えば肺のマスクが得られるなら、左右肺に囲まれた縦隔領域内の脂肪に限定する方法があります。具体的には体軸断で、左右の肺マスクの内側（間）の領域に存在する脂肪だけを残す、といった処理です。ただし、TotalSegmentatorの内臓脂肪マスク自体が主に体幹内部（肺門周辺や腹腔内）に分布するため、この追加フィルタは必須ではありません。実装を簡潔にするため、肺によるフィルタリングは省略可能です。

以上の前処理により、visceral_fat_maskから心臓の高さ近辺に存在する脂肪だけを残したマスクfat_near_heartを得ます。

3. 心膜周囲脂肪 (EAT+PAT) 領域の抽出

次に、心臓周囲の脂肪組織のみを抽出します。これはheart_roiマスクに近接した脂肪を取り出す操作となります。具体的には以下のいずれかの方法で実現できます。

方法A: 心臓マスクの拡張 (dilation)
heart_roiを一定半径だけ膨張させ、その膨張領域内に存在する脂肪を取得します。膨張半径は心膜からの最大距離に相当し、EATとPATを含めるためにはおおむね10～20mm程度が目安となります（患者の体格によりますが、心膜外の脂肪が前胸部に厚く蓄積する場合でも数cm程度です）。例えば半径約15mmで膨張させれば、心臓表面から15mm以内の領域をカバーできます。この膨張領域とfat_near_heartマスクをAND演算し、一致するボクセルを抽出します。これにより心臓の周囲半径15mm以内に存在する脂肪のみが残ります。

方法B: 距離変換によるフィルタ
心臓マスクheart_roiについてユークリッド距離変換(EDT: Euclidean Distance Transform)を計算し、各ボクセルが最も近い心臓ボクセルから何mm離れているかのマップを得ます
pure.rug.nl
。そして、fat_near_heart内の脂肪ボクセルで距離が所定閾値X(mm)以下のものを選択します。閾値Xは方法Aの膨張半径と同様、20mm前後が想定されます。距離判定により心臓から遠く離れた縦隔脂肪（例: 脊柱近くの脂肪）は除外され、心臓付近の脂肪のみが抽出できます。

どちらの方法でも結果は概ね似た領域になります。本仕様ではシンプルな方法Aで説明を続けます。膨張処理には3次元の構造要素（6近傍や26近傍の立方体など）を用いてN回の繰り返しdilateを実行します。膨張ボクセル数Nを決めるにはCTのボクセル解像度(spacing)を考慮し、例えばボクセルサイズが1mmの場合15回、ボクセルサイズが0.5mmの場合30回、といった具合に設定します。なお、Python実装ではscipy.ndimage.binary_dilation関数を用いると便利です。

膨張した心臓領域マスクをheart_dilatedとすると、EAT+PAT候補マスクは:

pericardial_fat_mask = heart_dilated AND fat_near_heart


となります。ここでpericardial_fat_maskには、心臓近傍に存在する脂肪がすべて含まれます。すなわち、心膜内の脂肪(EAT)と心膜直外の脂肪(PAT)の両方を包含するマスクです
pmc.ncbi.nlm.nih.gov
pmc.ncbi.nlm.nih.gov
。

注記: 心膜が非常に薄い膜であるため、実際のCTではEATとPATの境界は明瞭でないことが多いです
pmc.ncbi.nlm.nih.gov
。本手順では心臓マスクをわずかに膨張させているため、本来心膜で隔てられた内外の脂肪が一続きに選択されます。しかし先述のように、多くの研究でEATとPATは総和で評価され「心膜周囲脂肪」として扱われているため問題ありません
pure.rug.nl
。CT画像から両者を厳密に分離することは困難であり、本アプローチでもEAT+PATの総量として扱います。

4. HU閾値による脂肪ボクセル確認（オプション）

TotalSegmentatorの内臓脂肪マスクはディープラーニングにより脂肪領域を予測していますが、基本的には-190～-30HU程度の範囲を学習していると考えられます
pmc.ncbi.nlm.nih.gov
。念のため、抽出したpericardial_fat_mask領域のCT値を確認し、脂肪のHU範囲から外れるボクセルがあれば除外する処理を入れることもできます。ただし通常は不要です。オプションとして、CT画像データ（HU値配列）を参照し以下のフィルタを適用可能です:

ct = (CT画像のHU配列)
fat_hu_mask = ((ct >= -190) & (ct <= -30))
pericardial_fat_mask &= fat_hu_mask


こうすると、万一セグメンテーションにノイズがあり脂肪以外が混入していても排除できます。以降はpericardial_fat_maskを最終的な心膜周囲脂肪マスクとして扱います。

5. ボリュームおよび密度統計の計算

最終マスクpericardial_fat_maskから体積および**平均CT値(HU)**を計算します。

体積(ml): マスク中のボクセル数に各ボクセルの体積(mm³)を掛け、1000で割ることでmLに変換します。ボクセルの体積はCTのピクセルスペース寸法 (spacing) の積で求めます。例えば、voxel_volume_mm3 = spacing_x * spacing_y * spacing_z、マスク中のTrueボクセル数をN_voxとすると体積(ml)はvolume_ml = voxel_volume_mm3 * N_vox / 1000です。

平均HU値および標準偏差: 元のCT画素値配列から、pericardial_fat_maskがTrueとなっているボクセルのHU値を抽出し、その平均値と標準偏差を計算します。これにより脂肪組織の平均CT値（脂肪密度の指標）を得られます
pmc.ncbi.nlm.nih.gov
。典型的には-100HU前後になりますが、組成により多少変動します。

6. マスクと結果の出力

最後に、結果を所定の形式で保存します。以下の出力を想定します。

心膜周囲脂肪マスク (NIfTI形式): pericardial_fat_maskをCTと同じ空間情報でNIfTI画像として保存します（例えばeat_pat_mask.nii.gzというファイル名）。

統計値JSON: 体積(ml)、ボクセル数、平均HU、標準偏差HU、使用したHU範囲(-190～-30HU)などをJSON形式で保存します（例えばpericardial_fat_summary.json）。JSONには以下の情報を含めます。

volume_ml: 抽出脂肪の体積 [ml]

voxels: 抽出脂肪に含まれるボクセル数

mean_hu: 抽出領域の平均CT値 (HU)

std_hu: 抽出領域のCT値標準偏差 (HU)

hu_range: 閾値としたHU範囲 (ここでは[-190, -30])

その他任意で、処理に関するパラメータ（例えば心臓マスク膨張半径mm、使用モデル種類など）

以上で、EAT+PAT総量の算出処理は完了です。出力された体積値に関して、一般的な健常者では50～150 mL程度と報告されることが多いです
pure.rug.nl
。極端に外れた値が出ていないか簡単なバリデーションを行うと良いでしょう。

実装例 (Pythonスクリプト)

最後に、上述の手順を実現するPythonスクリプトの例を示します。NiBabelやSciPyなどのライブラリを用いてNIfTIマスクの読み書きや画像処理を行っています。

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, generate_binary_structure
import json

# パラメータ設定
HU_MIN = -190
HU_MAX = -30
DILATION_RADIUS_MM = 15.0  # 心臓マスク膨張半径 [mm]

# 入力ファイルパスの指定
ct_nifti_path = "input_ct.nii.gz"
totalseg_dir = "totalseg_output/"  # TotalSegmentator出力ディレクトリ

# 1. 心臓マスクの読み込み（heartchambersがあれば統合）
heart_mask = None
heart_components = ["heart_myocardium", "heart_atrium_left", "heart_atrium_right",
                    "heart_ventricle_left", "heart_ventricle_right"]
for comp in heart_components:
    path = f"{totalseg_dir}/{comp}.nii.gz"
    try:
        img = nib.load(path)
        data = img.get_fdata() > 0
        heart_mask = data if heart_mask is None else (heart_mask | data)
    except FileNotFoundError:
        continue

# heart_componentsが存在しない場合はtotalタスクのheartを使用
if heart_mask is None:
    heart_img = nib.load(f"{totalseg_dir}/heart.nii.gz")
    heart_mask = heart_img.get_fdata() > 0

# （オプション）大血管近位部の統合：ここでは大動脈のみ例示
try:
    aorta_img = nib.load(f"{totalseg_dir}/aorta.nii.gz")
    aorta_mask = aorta_img.get_fdata() > 0
    # 心臓マスクを少し膨張させた領域と重なるaortaだけ残す
    struct = generate_binary_structure(3, 1)  # 6近傍構造要素
    heart_dilate_small = binary_dilation(heart_mask, structure=struct, iterations=2)  # 2ボクセル膨張
    aorta_proximal = aorta_mask & heart_dilate_small
    heart_mask |= aorta_proximal
except FileNotFoundError:
    pass

# 2. 内臓脂肪マスク読み込み
visceral_fat_img = nib.load(f"{totalseg_dir}/internal_organs.nii.gz")  # internal organ fat mask
visceral_fat_mask = visceral_fat_img.get_fdata() > 0

# 3. 心臓高さ近辺の脂肪に限定（心臓マスクのZ範囲抽出）
z_indices = np.where(np.any(heart_mask, axis=(0,1)))[0]
if len(z_indices) > 0:
    z_min, z_max = z_indices[0], z_indices[-1]
else:
    z_min, z_max = 0, heart_mask.shape[2] - 1
# 上下1スライスずつ拡張（必要に応じて）
z_min = max(0, z_min - 1)
z_max = min(heart_mask.shape[2] - 1, z_max + 1)

fat_near_heart = np.zeros_like(visceral_fat_mask, dtype=bool)
fat_near_heart[:,:,z_min:z_max+1] = visceral_fat_mask[:,:,z_min:z_max+1]

# 4. 心膜周囲脂肪マスクの抽出（心臓マスク膨張とAND）
# 膨張ボクセル数の計算 (voxel spacing を考慮)
ct_img = nib.load(ct_nifti_path)
spacing = ct_img.header.get_zooms()[:3]  # (dx, dy, dz) in mm
# 最小ピクセルサイズに基づき膨張回数を計算
voxel_size_min = min(spacing)
iterations = int(np.ceil(DILATION_RADIUS_MM / voxel_size_min))

struct = generate_binary_structure(3, 2)  # 26近傍
heart_dilated = binary_dilation(heart_mask, structure=struct, iterations=iterations)
pericardial_fat_mask = heart_dilated & fat_near_heart

# HU範囲によるチェック（オプション）
ct_data = ct_img.get_fdata()
hu_mask = (ct_data >= HU_MIN) & (ct_data <= HU_MAX)
pericardial_fat_mask &= hu_mask

# 5. 統計量計算
voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
voxels = int(np.sum(pericardial_fat_mask))
volume_ml = voxel_vol_mm3 * voxels / 1000.0
fat_hus = ct_data[pericardial_fat_mask]
mean_hu = float(np.mean(fat_hus)) if voxels > 0 else None
std_hu  = float(np.std(fat_hus)) if voxels > 0 else None

# 6. 出力保存
# マスクをNIfTI画像として保存
out_mask_img = nib.Nifti1Image(pericardial_fat_mask.astype(np.uint8), ct_img.affine)
nib.save(out_mask_img, "eat_pat_mask.nii.gz")

# 統計をJSONで保存
stats = {
    "eat_pat_volume_ml": round(volume_ml, 2),
    "eat_pat_voxels": voxels,
    "eat_pat_mean_hu": round(mean_hu, 1) if mean_hu is not None else None,
    "eat_pat_std_hu": round(std_hu, 1) if std_hu is not None else None,
    "hu_range": [HU_MIN, HU_MAX],
    "dilation_radius_mm": DILATION_RADIUS_MM
}
with open("pericardial_fat_summary.json", "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"Pericardial fat volume: {volume_ml:.2f} mL")
print(f"Mean HU: {mean_hu:.1f} (±{std_hu:.1f})")


上記スクリプトでは、まず心臓および必要なら大血管近位部のマスクを作成し(heart_mask)、TotalSegmentatorの内臓脂肪マスクから心臓の高さに存在する脂肪fat_near_heartを抽出しています。続いて、heart_maskを物理距離で約15mm膨張させ、その領域内にある脂肪をAND演算で取り出すことでpericardial_fat_maskを得ています。最後に体積とHU値を計算し、NIfTIマスクとJSONレポートを保存しています。

この実装により、追加の手動操作なしで**心膜内外脂肪の総量(EAT+PAT)**を自動抽出できます。心膜を明示的にセグメンテーションしなくても、TotalSegmentatorのマスクと距離条件を用いることで文献定義に沿った心周囲脂肪量を算出できる点が特徴です
pure.rug.nl
。得られた値は論文報告されている範囲内に収まることが期待されますが
pure.rug.nl
、不自然に大きな値（例えば300 mLを超えるなど）の場合はパラメータ（膨張距離やHU範囲）の見直しやマスク結果の検証が推奨されます。

参考文献: 本アルゴリズムの定義や手法は、文献におけるEAT/PATの定義
pmc.ncbi.nlm.nih.gov
pmc.ncbi.nlm.nih.gov
、総脂肪量測定の実情
pure.rug.nl
、およびHU閾値の基準
pmc.ncbi.nlm.nih.gov
に基づいています。以上を踏まえ、TotalSegmentator出力のみでEAT+PATを包括的に評価するスクリプトを構築しました。