はい、**ここまでの結果は良好**です。

* PowerShell 側の `docker run --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi` で GPU を認識できています（Docker Desktop の WSL2 + GPU パスが機能）。([Docker Documentation][1])
* WSL2(Ubuntu) 内でも `nvidia-smi` が通り、アイドル時の使用量（584MiB）が表示されています。これは X サーバ等の常駐プロセスが少量の VRAM を使っているだけなので問題ありません。WSL2 における CUDA パスの要件も満たしています。([Microsoft Learn][2])

以下、「初心者にもわかる」ことを最優先に**次の手順**を丁寧に書きます。
（※コマンドは **WSL2 の Ubuntu ターミナル**で実行する想定です。`$` はプロンプトです。）

---

## ステップ 0：WSL2 と Docker Desktop の連携確認（1 回だけ）

1. Docker Desktop の設定を確認

* Windows 側で Docker Desktop を開く → **Settings ▸ Resources ▸ WSL Integration** → **Ubuntu にチェック** → Apply & Restart。([Docker Documentation][3], [Microsoft Learn][4])

2. Ubuntu で Docker CLI が使えるか確認

```bash
$ docker --version
$ docker run --rm hello-world
# GPU パスの最終確認（WSL 内からも）
$ docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

これが通れば、**WSL 内から Docker＋GPU が使える状態**です。([Docker][5])

---

## ステップ 1：作業フォルダを作る（WSL 側）

> Windows ドライブ直下（/mnt/c/...）より、**WSL のホーム配下**の方が I/O が速くトラブルが少ない慣行です。([Microsoft Learn][4])

```bash
$ mkdir -p ~/ctrate_ws/{data,outputs,tools}
$ cd ~/ctrate_ws
```

---

## ステップ 2：Hugging Face にログイン & ダウンローダの導入

CT‑RATE v2（\*\_fixed/）を正しく取得するため、主催側が案内している**公式ヘルパースクリプト**を使うのが安全です。([vlm3dchallenge.com][6])

1. Python ツール導入＆ログイン

```bash
$ python3 -m venv .venv && source .venv/bin/activate
$ pip install -U pip huggingface_hub datasets
$ huggingface-cli login         # ブラウザで作成したHFトークンを貼り付け
```

2. 公式ヘルパースクリプトの取得と実行
   （リポジトリ：`sezginerr/example_download_script`）([vlm3dchallenge.com][6])

```bash
$ cd tools
$ git clone https://github.com/sezginerr/example_download_script.git
$ cd example_download_script
$ pip install -r requirements.txt   # 必要なら

# v2 の "valid_fixed" を 1 件だけテスト取得
$ python download_dataset.py \
    --dataset CT-RATE \
    --output ~/ctrate_ws/data/CT-RATE \
    --split valid_fixed \
    --max_files 1
```

> CT‑RATE は Hugging Face の **`ibrahimhamamci/CT-RATE`** にホストされ、v2 は `*_fixed/` ディレクトリ（例：`valid_fixed/`）にあります。**v2 を推奨**という公式の注意書きもあります。([Hugging Face][7], [vlm3dchallenge.com][6])

3. 取得確認（HU 範囲やヘッダの妥当性）

```bash
$ python - << 'PY'
import nibabel as nib, numpy as np, glob
p = sorted(glob.glob('~/ctrate_ws/data/CT-RATE/valid_fixed/**/*.nii.gz', recursive=True))[0]
img = nib.load(p); data = img.get_fdata()
print("file:", p)
print("shape:", data.shape)
print("range[%.0f, %.0f]"%(data.min(), data.max()))
print("zooms:", img.header.get_zooms()[:3])
print("affine:\n", img.affine)
print("HU-like:", (data.min()<=-800) and (data.max()>=1000))
PY
```

**`HU-like: True`** になれば v2 仕様に合致しており、TotalSegmentator 等がそのまま動きます（v1 は 0–4095 で HU 変換が必要。v2 は補正済みでその手間が省けます）。([vlm3dchallenge.com][6])

---

## ステップ 3（推奨の小テスト）：TotalSegmentator の公式 Docker で心臓を一例セグメント

BOA を本格導入する前に、**まず 1 例で心臓セグメントが正常生成されるか**を最短経路で確認します。TotalSegmentator の Docker 画像には CLI がまとまっており、**心房・心室・心筋などの heartchambers 系**が使えます（高精細タスクはライセンス要）。([GitHub][8], [3D Slicer Community][9])

1. ライセンスをコンテナに登録して永続化
   （配布されたキーを `YOUR_KEY_HERE` に置換。ホームの `.totalsegmentator` をボリュームにマウントし永続化します）

```bash
$ mkdir -p ~/.totalsegmentator
$ docker pull totalsegmentator/totalsegmentator
$ docker run --rm -it --gpus all \
    -v ~/.totalsegmentator:/root/.totalsegmentator \
    totalsegmentator/totalsegmentator \
    totalseg_set_license -l YOUR_KEY_HERE
```

2. 1 件を heartchambers\_highres で推論（GPU 使用）

```bash
# 例: 取得した v2 の NIfTI を1つ指す
$ export CT=~/ctrate_ws/data/CT-RATE/valid_fixed/valid_*/valid_*_*/valid_*_*_*.nii.gz
$ export OUT=~/ctrate_ws/outputs/ts_test

$ docker run --rm -it --gpus all \
  -v "$HOME:$HOME" \
  totalsegmentator/totalsegmentator \
  TotalSegmentator -i "$CT" -o "$OUT" \
    --task heartchambers_highres --device gpu --robust_crop --body_seg
```

3. 生成物の確認

```bash
$ ls ~/ctrate_ws/outputs/ts_test | grep heart_
# heart_myocardium.nii.gz, heart_atrium_left.nii.gz, heart_ventricle_left.nii.gz ... が並べばOK
```

> heartchambers\_highres はライセンス対象タスクです。上のように `totalseg_set_license` を通しておけば CLI が動きます。([3D Slicer Community][9])

---

## ステップ 4（ベースライン）：BOA を Docker で動かす

BOA（Body‑and‑Organ‑Analysis）は **TotalSegmentator + 体組成解析**等を一括運用できるツールです。Docker での利用が前提で、README には \*\*「環境変数を設定し、PACS 連携またはコマンドラインツールを使う」\*\*とあります。Windows 向け `docker-compose-win.yml` も同梱されています。([GitHub][10])

1. 取得

```bash
$ cd ~/ctrate_ws/tools
$ git clone https://github.com/UMEssen/Body-and-Organ-Analysis.git BOA
$ cd BOA
# 参考: リポジトリ直下に README.md / docker-compose.yml / docker-compose-win.yml / .env_sample などが入っています
```

2. .env を作成（最小設定）

```bash
$ cp .env_sample .env
# エディタで必要項目を設定（例）
#  - 入出力のマウントパス
#  - 利用GPU数
#  - （必要時）TotalSegmentator のライセンスを格納するホスト側パス
```

3. コンテナ起動
   Windows の Docker Desktop を使うため、**WSL 内からでも `docker compose` が使えます**。

```bash
# Windows 環境向けの compose ファイルがある場合はこちらを優先
$ docker compose -f docker-compose-win.yml up -d
# ない場合は通常の compose
# $ docker compose up -d
```

4\)（コマンドライン実行の例・雛形）
BOA は \*\*「コマンドラインツール」\*\*経由でも使える旨が README に記載されています。実行サンプルは `example_scripts/` 配下に用意されています。まずは **1 ケースを指定して出力先にセグメンテーションを生成**します。
※実際のサブコマンド名やオプションは BOA の README／`example_scripts` のスクリプトに合わせてください（リポジトリの更新に追随）。([GitHub][10])

```bash
# 例：BOA コンテナに入って CLI を実行する（サービス名は compose 内の定義に合わせる）
$ docker compose exec boa bash

# （コンテナ内で）入力と出力を指定して実行する想定
root@container:/app# boa_cli run \
    --input /data/CT-RATE/valid_fixed/.../case.nii.gz \
    --output /out/boa_case1 \
    --tasks total heartchambers body_composition
# ↑ 実際のコマンドは example_scripts/ のサンプルに準拠してください
```

> BOA は **TotalSegmentator を内部で用いる**設計です。TotalSegmentator のライセンスが必要なタスク（heartchambers\_highres 等）を BOA から実行する場合、コンテナにライセンスファイルをマウントし、必要なら BOA コンテナ内で `totalseg_set_license` を一度実行してください（`~/.totalsegmentator` を永続ボリュームにするのが簡単）。([GitHub][10])

---

## ステップ 5：EAT/PAT の抽出（簡易スクリプト）

BOA/TotalSegmentator の出力（心筋・心房・心室マスク）から、**HU 閾値（-190〜-30HU）＋心筋の膨張距離**で EAT/PAT を計算できます（ご自身の既存スクリプトでも可）。以下は最小動作の例です。

```python
# ~/ctrate_ws/tools/eat_pat_min.py
import nibabel as nib, numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

ct_p   = "<CT v2 NIfTI>"
myo_p  = "<heart_myocardium.nii.gz>"
out_e  = "epicardial_5mm.nii.gz"
out_p  = "pericardial_10mm.nii.gz"

ct  = nib.load(ct_p);  ct_data = ct.get_fdata()
myo = nib.load(myo_p).get_fdata() > 0
sp  = ct.header.get_zooms()[:3]; vx = np.prod(sp)/1000.0

fat = (ct_data>=-190) & (ct_data<=-30)
st  = generate_binary_structure(3,2)
d5  = int(5.0/min(sp));  d10 = int(10.0/min(sp))

roi5 = myo.copy()
for _ in range(d5):  roi5 = binary_dilation(roi5, st)
roi10= myo.copy()
for _ in range(d10): roi10= binary_dilation(roi10, st)

EAT = fat & roi5  & ~myo
PAT = fat & roi10 & ~myo

def save(mask, path): nib.save(nib.Nifti1Image(mask.astype(np.uint8), ct.affine, ct.header), path)
save(EAT, out_e); save(PAT, out_p)
print("EAT(ml)=",EAT.sum()*vx, " PAT(ml)=",PAT.sum()*vx)
```

実行：

```bash
$ python ~/ctrate_ws/tools/eat_pat_min.py
```

---

## ステップ 6：2D 投影（DRR/脂肪マップ）

Plastimatch で DRR、NumPy で脂肪投影（または Plastimatch で 0/1 マスクの線積分）を生成し、学習ペア（DRR, FatMap）を作ります。DRR は既にご利用の手順でOK。脂肪投影は **Y 軸和**（PA/AP 近似）でまずは十分です。

```python
# fat_mask の 3D（EAT/PAT どちらでも） -> 2D 投影
import nibabel as nib, numpy as np, imageio
m = nib.load('pericardial_10mm.nii.gz').get_fdata()>0
proj = m.sum(axis=1).astype(np.float32)    # Y軸和（LPSの前後）
proj = (proj/proj.max()*255).astype(np.uint8)
imageio.v2.imwrite('fatmap.png', proj)
```

---

# トラブルシューティング（詰まりポイント別）

* **WSL 内で `docker` が動かない**
  → Docker Desktop の **WSL Integration** を Ubuntu に有効化、`hello-world` が動くか確認。([Docker Documentation][3], [Microsoft Learn][4])

* **`--gpus all` が通らない／GPU が見えない**
  → Windows 側の NVIDIA ドライバ更新、`wsl --update`、Docker Desktop の WSL2 GPU 有効化（Prerequisites を満たす）。([Docker Documentation][1], [Microsoft Learn][2])

* **CT‑RATE で v1（未補正）を取ってしまう**
  → **`*_fixed/`（v2）** を指定してダウンロード。主催側ページでも v2 を推奨。([vlm3dchallenge.com][6])

* **TotalSegmentator heartchambers 系がエラー／結果空**
  → v1 を使っていないか、HU 範囲が正しいかを冒頭の検証スクリプトで確認。`--robust_crop --body_seg` を付与。**ライセンス登録**（`totalseg_set_license -l`）も必須。([GitHub][8], [3D Slicer Community][9])

* **BOA のコマンド／Compose で迷う**
  → リポジトリの **README「How to run?」と `example_scripts/`** を参照。`docker-compose-win.yml` が同梱されているので Windows 環境ではこちらを使う。([GitHub][10])

---

## この後のロードマップ（何をすればよいか）

1. **TotalSegmentator 公式 Docker で 1 ケース心臓セグメント**（上のステップ 3）→ マスクが妥当かを可視化で確認。([GitHub][8])
2. **EAT/PAT を最小スクリプトで算出**（ステップ 5）→ 体積のオーダが現実的かチェック。
3. **DRR＋脂肪マップの 2D ペア作成**（ステップ 6）。
4. ここまで OK なら、**BOA を常用**（ステップ 4）にシフトして、バッチ処理（複数症例）へ。BOA は TotalSegmentator と Body Composition を組み合わせたワークフローで、基礎文献や最近の解説でもその位置づけが説明されています。([GitHub][10], [PMC][11])

---

必要なら、この手順を**コピペで動くシェル**（`setup_and_test.sh`）にまとめた形でもお渡しできます。まずは **ステップ 3 の 1 例推論**まで終えた段階の出力一覧（`ls ~/ctrate_ws/outputs/ts_test`）を教えてください。そこから BOA の常用運用（Compose の永続ボリューム設計、ライセンス永続化、並列バッチ）に進めます。

[1]: https://docs.docker.com/desktop/features/gpu/?utm_source=chatgpt.com "GPU support in Docker Desktop for Windows"
[2]: https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl?utm_source=chatgpt.com "Enable NVIDIA CUDA on WSL 2"
[3]: https://docs.docker.com/desktop/features/wsl/?utm_source=chatgpt.com "Docker Desktop WSL 2 backend on Windows"
[4]: https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers?utm_source=chatgpt.com "Get started with Docker remote containers on WSL 2"
[5]: https://www.docker.com/blog/wsl-2-gpu-support-for-docker-desktop-on-nvidia-gpus/?utm_source=chatgpt.com "WSL 2 GPU Support for Docker Desktop on NVIDIA GPUs"
[6]: https://vlm3dchallenge.com/getting-started/?utm_source=chatgpt.com "Getting Started with VLM3D Challenge"
[7]: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE?utm_source=chatgpt.com "ibrahimhamamci/CT-RATE · Datasets at Hugging Face"
[8]: https://github.com/wasserth/TotalSegmentator?utm_source=chatgpt.com "wasserth/TotalSegmentator: Tool for robust segmentation ..."
[9]: https://discourse.slicer.org/t/total-segmentator-1-heartchambers-highres/32505?utm_source=chatgpt.com "Total Segmentator 1 - heartchambers highres - Support"
[10]: https://github.com/UMEssen/Body-and-Organ-Analysis?utm_source=chatgpt.com "GitHub - UMEssen/Body-and-Organ-Analysis: BOA is a ..."
[11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12138298/?utm_source=chatgpt.com "Exchange of Quantitative Computed Tomography ..."
