0) まとめ（最短ルート）

原因：DeepDRR 1.1.3 の投影カーネルは Ruijters 氏の「CubicInterpolationCUDA」をベースにしており、CUDA 12 で削除された legacy の texture reference API（texture<T, 3, ...> / tex3D(tex, ...)）に依存。CUDA 12.0+ ではこの構文がコンパイル不可です（identifier 'texture' is undefined など）
PyPI
GitHub
NVIDIA Developer Forums
Stack Overflow

現状のDeepDRR側のスタンス：README/ PyPI はCUDA 11 を推奨。pip の最新リリース 1.1.3 は 2023-07-14 のまま（CUDA 12 対応の明記なし）
GitHub
PyPI

最短で安定稼働：CUDA 11.8 ツールキットの環境（conda / Docker）を作り、その上で deepdrr==1.1.3 を使う。RTX 3090 + 近年のドライバは後方互換により、CUDA 11 でビルドしたアプリを動かせます（“driver は常に後方互換”の公式説明）
NVIDIA Docs
+1md_files/answer1.md

CUDA 12.9 固定で行く場合：Ruijters の cubic 補間コードをtexture object API（cudaTextureObject_t）へ移行するパッチが必要。実際にtexture reference→texture object に置き換えた移植例が公開されています（MMORF の MR）
GitLab

以下、詳細です。

1) 何が起きているか（エラーの正体）

DeepDRR の投影は Ruijters 氏の CUDA 3D cubic B-spline 補間を取り入れています（cubicTex3D 等）。このコードは texture<T, 3, ...> と tex3D を使う古典的な texture reference スタイルです 
PyPI
GitHub
dannyruijters.nl

CUDA 12 からは texture reference が完全削除され、コンパイル段階で texture is not a template / identifier 'texture' is undefined 等のエラーになります（NVIDIA 公式フォーラムの回答、OpenCV 側の同趣旨 issue/回答）
NVIDIA Developer Forums
Stack Overflow

よって pycuda の問題ではなく、カーネル側の構文が CUDA 12 で廃止されたことが原因です（pycuda はドライバAPIのラッパであり、kernel のビルドは結局 nvcc/NVRTC を用いるため）※CuPy でも texture reference は v10.1 時点で “非推奨” 注記があり、将来的な削除に言及されています 
docs.cupy.dev

2) DeepDRR の CUDA 12 対応状況

最新 pip リリース 1.1.3（2023/07/14）：インストール手順に**「CUDA 11 推奨」**と明記（開発ブランチの依存でも CUDA 11.1 を要求）。CUDA 12 対応は記載なし 
PyPI
GitHub

したがって CUDA 12.9 と相性の良い DeepDRR バージョンは未公表 です。現時点では CUDA 11 系での運用が “公式に一番楽” です。

3) 解決策A（最短・堅牢）：CUDA 11.8 環境を用意してその上で DeepDRR 1.1.3 を使う

ポイント：GPUドライバは後方互換なので、ホストに CUDA 12.9 なドライバが入っていても、ユーザー空間で CUDA 11.8 のツールキットを使ってビルドしたバイナリは動作します（公式 FAQ/Compatibility ガイド）
NVIDIA Docs
+1

3.1 conda 環境（推奨）
# environment-deepdrr-cuda11.yml
name: deepdrr-cuda11
channels:
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - cuda-toolkit=11.8        # nvcc 同梱の公式メタパッケージ
  - pycuda                   # conda-forge のビルド済み (2025年時点で配布あり)
  - numpy
  - pip
  - pip:
      - deepdrr==1.1.3
      - nibabel
      - pydicom
      - scikit-image

mamba env create -f environment-deepdrr-cuda11.yml
mamba activate deepdrr-cuda11

# 確認（nvccが11.8であること）
nvcc --version

# PyCUDA がGPUを見られるか簡易確認
python - <<'PY'
import pycuda.driver as drv; drv.init()
print("Device:", drv.Device(0).name())
PY


cuda-toolkit=11.8 は nvidia チャンネルで公式配布されています 
Anaconda

pycuda は conda-forge から入れるとビルド不要のホイールが使え、環境が安定しやすいです（配布状況）
Anaconda

3.2 Docker（再現性重視）
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip python3-dev && \
    pip3 install --upgrade pip && \
    pip3 install pycuda deepdrr==1.1.3 nibabel pydicom scikit-image


ベースイメージのタグ例（11.8.0-devel-ubuntu22.04）は公式で提供されています 
Docker Hub
+1

3.3 動作スモークテスト（DeepDRR最小例）
from deepdrr import geo, Volume, MobileCArm
from deepdrr.projector import Projector  # PyCUDA初期化
ct = Volume.from_nifti("/path/to/ct.nii.gz")
carm = MobileCArm()

with Projector(ct, carm=carm) as projector:
    ct.orient_patient(head_first=True, supine=True)
    ct.place_center(carm.isocenter_in_world)
    carm.move_to(alpha=30, beta=10, degrees=True)
    img = projector()  # numpy array


（この最小例は DeepDRR の README / PyPI に準拠）
PyPI