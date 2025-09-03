実装プラン：DiffDRRとTorchIOによる3Dから2Dへの定量的データ拡張パイプライン1. 目的とアーキテクチャ概要本実装プランは、3D胸部CT（NIfTI形式）とそのセグメンテーションマスクから、以下の2つの成果物を生成するエンドツーエンドのパイプラインを構築することを目的とします。擬似胸部レントゲン画像 (DRR): 臨床的に現実的なパラメータ設定と、多様なデータ拡張を適用した2D画像。定量的2D体積マスク: DRRと完全に空間的位置合わせがされた2Dマスク。各ピクセル値は、そのピクセルが表す領域の3Dマスクの体積（mm³）を反映しており、全ピクセル値の総和が元の3Dマスクの総体積と一致します。このパイプラインは、以下の2つの主要ライブラリを組み合わせることで、効率性と柔軟性を最大化します。TorchIO: 3Dボリューム（CTとマスク）に対する空間的・強度的なデータ拡張（解剖学的多様性のシミュレーション）を担当します 1。DiffDRR: 拡張された3Dボリュームを2Dに投影し、DRRと厚みマップを生成します。また、レンダリング時のカメラポーズを動的に変更することで、撮影角度の多様性をシミュレートします 2。2. ステップ1：環境構築まず、プロジェクトに必要なライブラリをインストールします。Python 3.8以上が推奨されます 3。Bash# PyTorchのインストール (ご自身のCUDAバージョンに合わせてください)
# https://pytorch.org/get-started/locally/ を参照
pip install torch torchvision torchaudio

# DiffDRRと関連ライブラリのインストール
pip install diffdrr torchio simpleitk numpy matplotlib
3. ステップ2：NIfTIデータの読み込みと準備TorchIOは、医用画像データ（特にNIfTI）とそれに付随するメタデータ（ボクセル間隔、座標系など）を効率的に扱うための優れたツールです 1。CT画像とマスク画像をSubjectオブジェクトとしてペアで管理することで、後のデータ拡張処理が極めて容易になります。Python# 01_load_data.py
import torchio as tio
import torch

def load_nifti_data(ct_path: str, mask_path: str) -> tio.Subject:
    """
    NIfTI形式のCTとマスクファイルを読み込み、TorchIOのSubjectオブジェクトを作成する。
    
    Args:
        ct_path (str): CT画像のファイルパス
        mask_path (str): マスク画像のファイルパス

    Returns:
        tio.Subject: CTとマスクを格納したSubjectオブジェクト
    """
    subject = tio.Subject(
        ct=tio.ScalarImage(ct_path),
        mask=tio.LabelMap(mask_path),
    )
    return subject

# --- 使用例 ---
if __name__ == '__main__':
    # ご自身のファイルパスに置き換えてください
    CT_NIFTI_PATH = "./path/to/your_ct.nii.gz"
    MASK_NIFTI_PATH = "./path/to/your_mask.nii.gz"

    # データをロード
    subject = load_nifti_data(CT_NIFTI_PATH, MASK_NIFTI_PATH)

    # 読み込んだ情報を表示
    print("--- Subject Information ---")
    print(subject)
    
    # CT画像の情報を表示
    print("\n--- CT Image ---")
    print(f"Shape: {subject.ct.shape}")
    print(f"Spacing (mm): {subject.ct.spacing}")
    print(f"Orientation: {subject.ct.orientation}")

    # マスク画像の情報を表示
    print("\n--- Mask Image ---")
    print(f"Shape: {subject.mask.shape}")
    print(f"Spacing (mm): {subject.mask.spacing}")

    # データのテンソルとアフィン行列にアクセス
    ct_tensor = subject.ct.data
    mask_tensor = subject.mask.data
    affine_matrix = subject.ct.affine

    print(f"\nCT Tensor Shape: {ct_tensor.shape}")
4. ステップ3：3Dデータ拡張 (TorchIO)解剖学的な多様性をシミュレートするため、DRRを生成する前に3Dボリューム自体に拡張を適用します。TorchIOでは、これらの処理をComposeで一連のパイプラインとして定義できます 5。重要な点は、Subjectオブジェクトに変換を適用すると、CTとマスクの両方に同じ空間変換が自動的に適用されるため、位置合わせが崩れる心配がないことです。Python# 02_augmentation.py
import torchio as tio

def get_augmentation_transform() -> tio.Compose:
    """
    3Dデータ拡張のための変換パイプラインを定義する。
    """
    augment = tio.Compose()
    return augment

# --- 使用例 ---
if __name__ == '__main__':
    from step_01_load_data import load_nifti_data

    CT_NIFTI_PATH = "./path/to/your_ct.nii.gz"
    MASK_NIFTI_PATH = "./path/to/your_mask.nii.gz"
    subject = load_nifti_data(CT_NIFTI_PATH, MASK_NIFTI_PATH)
    
    augmentation_pipeline = get_augmentation_transform()
    
    # 拡張を適用
    augmented_subject = augmentation_pipeline(subject)
    
    print("--- Original Subject ---")
    print(subject)
    print("\n--- Augmented Subject ---")
    print(augmented_subject)
    
    # 拡張後のデータは augmented_subject.ct.data などでアクセス可能
5. ステップ4：DiffDRRによる2D投影このステップがパイプラインの中核です。まず、臨床的に現実的なパラメータで仮想X線撮影装置（DRRモジュール）をセットアップします。次に、動的にカメラポーズを変更しながら、拡張された3Dボリュームを2Dに投影します。5.1. DiffDRRモジュールのセットアップDRRクラスの初期化には、撮影装置の物理的なジオメトリを指定するパラメータが必要です 6。これらのパラメータは、生成される画像のリアリズムに直接影響します。パラメータ説明推奨値（胸部X線）sdd線源-検出器間距離 (mm)。Source-to-Detector Distance。1800.0height検出器の高さ (ピクセル)。512, 1024, 2048 などwidth検出器の幅 (ピクセル)。512, 1024, 2048 などdelx, delyピクセル間隔 (mm/pixel)。検出器の物理的解像度。0.3 - 0.8 (解像度による)x0, y0主点オフセット (mm)。通常は中心 (0.0) で問題ありません。0.0device計算を行うデバイス。"cuda"5.2. カメラポーズの定義 (オンザフライ拡張)DiffDRRの真価は、forwardメソッドを呼び出すたびにカメラポーズ（回転と平行移動）を動的に変更できる点にあります 2。これにより、1つの3Dボリュームから無数の異なる視点の画像を効率的に生成できます。回転 (Rotations): 3つの角度（例：alpha, beta, gamma）で指定します。どの軸周りの回転かはconventionで定義します 6。平行移動 (Translations): X, Y, Z方向の移動量 (mm) で指定します。DiffDRRの座標系では、一般的にY軸が線源から検出器へ向かう方向です。5.3. 投影の実装コードPython# 03_projection.py
import torch
from diffdrr.drr import DRR
import numpy as np

def setup_drr_generator(subject: tio.Subject, height: int, delx: float, device: str) -> DRR:
    """
    DiffDRRジェネレータを臨床パラメータで初期化する。
    """
    # ToCanonicalを適用した後のspacingを取得
    spacing = subject.ct.spacing
    
    drr_generator = DRR(
        volume=subject.ct.data.squeeze().numpy(), # (C, D, H, W) -> (D, H, W)
        spacing=spacing,
        sdd=1800.0,
        height=height,
        delx=delx,
        device=device,
        # CTの原点を考慮に入れる
        origin=subject.ct.origin,
    )
    return drr_generator

def get_random_pose(batch_size: int, device: str) -> (torch.Tensor, torch.Tensor):
    """
    ランダムなカメラポーズ（回転と平行移動）を生成する。
    """
    # 回転 (alpha, beta, gamma) をラジアン単位で生成
    # 例: 各軸±5度の範囲
    rotations = torch.deg2rad(
        (torch.rand(batch_size, 3, device=device) - 0.5) * 10.0
    )
    
    # 平行移動 (x, y, z) をmm単位で生成
    # Y軸は線源から検出器方向。CTボリュームが中心付近に来るように調整
    # X, Zは±20mmの範囲でランダムにずらす
    translations = torch.tensor([0.0, -100.0, 0.0], device=device).expand(batch_size, -1)
    translations = translations + (torch.rand(batch_size, 3, device=device) - 0.5) * torch.tensor([40.0, 100.0, 40.0], device=device)
    
    return rotations, translations

# --- 使用例は次のセクションの完全なスクリプトに統合 ---
6. ステップ5：定量的スケーリングと最終出力DiffDRRでマスクを投影すると、各ピクセル値はX線がマスクを通過した**経路長（厚み、mm）になります。これを体積に変換するには、各ピクセルにそのピクセルが表す面積（mm²）**を乗算します。ピクセル面積 = delx * delyこの単純なスケーリングにより、2Dマスクのピクセル値の総和が、元の3Dマスクの体積と一致するようになります。7. ステップ6：完全なパイプラインスクリプトこれまでのステップをすべて統合し、NIfTIファイルを入力として、拡張されたDRRと定量的マスクのペアを生成・保存する完全なPythonスクリプトを以下に示します。Python# generate_augmented_drrs.py
import torch
import torchio as tio
from diffdrr.drr import DRR
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# -----------------------------------------------------------------------------
# STEP 1: 設定
# -----------------------------------------------------------------------------
# 入力ファイル
CT_NIFTI_PATH = "./path/to/your_ct.nii.gz"
MASK_NIFTI_PATH = "./path/to/your_mask.nii.gz"

# 出力設定
OUTPUT_DIR = "./augmented_data"
NUM_SAMPLES = 10  # 生成するサンプル数

# DRRパラメータ
DRR_HEIGHT = 512      # 生成画像の高さ (ピクセル)
PIXEL_SPACING = 0.7   # ピクセル間隔 (mm/pixel)

# デバイス設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# 出力ディレクトリを作成
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "drr"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "mask"), exist_ok=True)

# -----------------------------------------------------------------------------
# STEP 2: データ読み込み & 拡張パイプライン定義
# -----------------------------------------------------------------------------
def load_nifti_data(ct_path: str, mask_path: str) -> tio.Subject:
    subject = tio.Subject(
        ct=tio.ScalarImage(ct_path),
        mask=tio.LabelMap(mask_path),
    )
    return subject

def get_augmentation_transform() -> tio.Compose:
    return tio.Compose()

# -----------------------------------------------------------------------------
# STEP 3: DiffDRRジェネレータとポーズ生成の定義
# -----------------------------------------------------------------------------
def setup_drr_generator(subject: tio.Subject, height: int, delx: float, device: str) -> DRR:
    # ToCanonicalを適用した後のspacingを取得
    spacing = subject.ct.spacing
    
    # DiffDRRは (D, H, W) のnumpy配列を期待する
    volume_np = subject.ct.data.squeeze().numpy()
    
    drr_generator = DRR(
        volume=volume_np,
        spacing=spacing,
        sdd=1800.0,
        height=height,
        delx=delx,
        device=device,
        origin=subject.ct.origin,
        renderer='siddon', # Siddon's methodを使用 [2]
    )
    return drr_generator

def get_random_pose(device: str) -> (torch.Tensor, torch.Tensor):
    # 回転 (alpha, beta, gamma) をラジアン単位で生成 (バッチサイズ=1)
    rotations = torch.deg2rad((torch.rand(1, 3, device=device) - 0.5) * 20.0) # ±10度
    
    # 平行移動 (x, y, z) をmm単位で生成
    # Y軸は線源から検出器方向。CTボリュームが中心付近に来るように調整
    translations = torch.tensor([[0.0, -100.0, 0.0]], device=device)
    translations += (torch.rand(1, 3, device=device) - 0.5) * torch.tensor([50.0, 150.0, 50.0], device=device)
    
    return rotations, translations

# -----------------------------------------------------------------------------
# STEP 4: メイン処理ループ
# -----------------------------------------------------------------------------
# 1. データをロード
print("Loading NIfTI data...")
initial_subject = load_nifti_data(CT_NIFTI_PATH, MASK_NIFTI_PATH)

# 2. 3D拡張パイプラインを準備
augmentation_pipeline = get_augmentation_transform()

# 3. メインループ
for i in tqdm(range(NUM_SAMPLES), desc="Generating Samples"):
    # 4. 3D拡張を適用
    augmented_subject = augmentation_pipeline(initial_subject)
    
    # 5. DiffDRRジェネレータを拡張後のデータでセットアップ
    #    (毎回作り直すか、set_volumeで更新する)
    drr_generator = setup_drr_generator(augmented_subject, DRR_HEIGHT, PIXEL_SPACING, DEVICE)
    
    # 6. ランダムなカメラポーズを取得
    rotations, translations = get_random_pose(DEVICE)
    
    # 7. DRRと厚みマップを生成
    #    DiffDRRのforwardメソッドに直接テンソルを渡す
    ct_tensor = augmented_subject.ct.data.to(DEVICE)
    mask_tensor = augmented_subject.mask.data.to(DEVICE).float() # マスクはfloat型に変換

    # DRR生成
    drr_image = drr_generator(
        rotations=rotations, 
        translations=translations, 
        parameterization="euler_angles", 
        convention="ZXY",
        density=ct_tensor
    )
    
    # 厚みマップ生成
    thickness_map = drr_generator(
        rotations=rotations, 
        translations=translations, 
        parameterization="euler_angles", 
        convention="ZXY",
        density=mask_tensor
    )

    # 8. 定量的スケーリング
    pixel_area = drr_generator.delx * drr_generator.dely
    volumetric_mask = thickness_map * pixel_area

    # 9. 検証 (任意)
    voxel_volume = torch.prod(torch.tensor(augmented_subject.ct.spacing))
    original_volume_3d = torch.sum(mask_tensor) * voxel_volume
    projected_volume_2d = torch.sum(volumetric_mask)
    
    if i == 0: # 最初のサンプルで体積を比較表示
        print(f"\nOriginal 3D Mask Volume: {original_volume_3d.item():.2f} mm³")
        print(f"Projected 2D Mask Volume Sum: {projected_volume_2d.item():.2f} mm³")

    # 10. 結果を保存 (TensorをCPUに移動し、Numpy配列に変換)
    drr_np = drr_image.squeeze().cpu().numpy()
    mask_np = volumetric_mask.squeeze().cpu().numpy()
    
    plt.imsave(os.path.join(OUTPUT_DIR, "drr", f"sample_{i:04d}.png"), drr_np, cmap='gray')
    # マスクはNumpy配列として保存することも可能
    np.save(os.path.join(OUTPUT_DIR, "mask", f"sample_{i:04d}.npy"), mask_np)

print(f"\nSuccessfully generated {NUM_SAMPLES} samples in '{OUTPUT_DIR}'")

スクリプトの実行方法上記のコードをgenerate_augmented_drrs.pyとして保存します。CT_NIFTI_PATHとMASK_NIFTI_PATHを実際のファイルパスに書き換えます。ターミナルでスクリプトを実行します: python generate_augmented_drrs.py実行が完了すると、OUTPUT_DIRで指定したディレクトリ内にdrrとmaskのサブディレクトリが作成され、生成された画像とマスクデータが保存されます。この実装プランにより、要求されたすべての要件（データ拡張、DRR生成、定量的体積マスク生成）を満たす、再現可能でカスタマイズ容易なパイプラインが完成します。