#!/usr/bin/env python3
"""
DeepDRR単体テスト用スクリプト
CT-RATE v2の1つのNIfTIファイルからDRRを生成してPNG保存・可視化する
"""

from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector  # PyCUDA初期化を安定化

def to_uint16_xray(image: np.ndarray, method: str = "log") -> np.ndarray:
    """
    DeepDRRの出力をX線表示らしく（骨=白）する16bit PNG用画像へ
    """
    img = image.astype(np.float32)
    img /= (img.max() + 1e-6)
    
    if method == "log":
        img = -np.log(img + 1e-6)  # Beer-Lambert風の線積分表示
    elif method == "invert":
        img = 1.0 - img
    
    img -= img.min()
    img /= (img.max() + 1e-8)
    return (img * 65535.0).astype(np.uint16)

def visualize_drr(img_pa, img_lat, save_path: Path):
    """
    PA・LAT両方向のDRRを可視化して保存
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # PA (正面) 
    axes[0].imshow(img_pa, cmap='gray')
    axes[0].set_title('PA View (Frontal)')
    axes[0].axis('off')
    
    # LAT (側面)
    axes[1].imshow(img_lat, cmap='gray')
    axes[1].set_title('LAT View (Lateral)')  
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"可視化画像を保存: {save_path}")

def main():
    # 入力ファイルパス（CT-RATE v2の例）
    nifti_path = Path("/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    output_dir = Path("/home/soya/ctrate_ws/deepdrr/outputs")
    
    print(f"入力NIfTI: {nifti_path}")
    assert nifti_path.exists(), f"ファイルが見つかりません: {nifti_path}"
    
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DeepDRR初期化
    print("DeepDRRを初期化中...")
    carm = MobileCArm()
    ct = Volume.from_nifti(str(nifti_path))
    
    print(f"CTボリューム形状: {ct.data.shape}")
    print(f"CTボリューム間隔: {ct.spacing}")
    
    with Projector(ct, carm=carm) as projector:
        # 患者を世界座標に整列
        print("患者の体位を調整中...")
        ct.orient_patient(head_first=True, supine=True)
        ct.place_center(carm.isocenter_in_world)
        
        # ① PA（正面）相当
        print("PA view（正面）を生成中...")
        carm.move_to(alpha=0, beta=0, degrees=True)
        img_pa = projector()
        
        # ② LAT（側面）相当
        print("LAT view（側面）を生成中...")
        carm.move_to(alpha=90, beta=0, degrees=True)
        img_lat = projector()
        
        print(f"DRR画像サイズ: {img_pa.shape}")
        print(f"PA view値範囲: {img_pa.min():.4f} - {img_pa.max():.4f}")
        print(f"LAT view値範囲: {img_lat.min():.4f} - {img_lat.max():.4f}")
    
    # 16bit PNGで保存
    print("画像を保存中...")
    pa_16bit = to_uint16_xray(img_pa, method="log")
    lat_16bit = to_uint16_xray(img_lat, method="log")
    
    pa_path = output_dir / "test_PA.png"
    lat_path = output_dir / "test_LAT.png"
    
    imageio.imwrite(pa_path, pa_16bit)
    imageio.imwrite(lat_path, lat_16bit)
    
    print(f"PA view保存: {pa_path}")
    print(f"LAT view保存: {lat_path}")
    
    # 可視化
    print("可視化を生成中...")
    viz_path = output_dir / "drr_comparison.png"
    visualize_drr(pa_16bit, lat_16bit, viz_path)
    
    print("✅ DRR生成が完了しました！")

if __name__ == "__main__":
    main()