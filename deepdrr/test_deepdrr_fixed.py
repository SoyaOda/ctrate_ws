#!/usr/bin/env python3
"""
DeepDRR改善版テストスクリプト
md_files/answers/answer2.mdの改善案に基づく実装

主な改善点:
- SID=180cm, pixel_size=0.30mm, 2048x2048で胸部全体をカバー
- -log変換で骨=白、肺=黒の臨床的表示
- パーセンタイル窓でコントラスト改善
- 幾何更新直後のprojector()を2回実行
"""

from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector  # PyCUDA初期化を先に
import matplotlib.pyplot as plt

# 1) 見た目をX線らしく： -log + パーセンタイル窓で16bit化
def to_uint16_xray(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """DeepDRR出力を臨床X線画像らしく変換（骨=白、肺=黒）"""
    x = np.clip(img.astype(np.float32), 1e-6, None)   # 透過0回避
    x = -np.log(x)                                    # Beer-Lambert 表示（骨=白, 肺=黒）
    lo, hi = np.percentile(x, [1.0, 99.5])            # 外れ値に頑健
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma != 1.0:
        x = np.power(x, 1.0/gamma)                    # 必要なら階調補正
    return (x * 65535).astype(np.uint16)

# 2) 胸部撮影向けのCアームを作る：SID=180 cm, 0.30 mm/pix, 2048^2
def make_cxr_carm(isocenter):
    """胸部X線撮影用のC-arm幾何を作成"""
    return MobileCArm(
        isocenter=isocenter,
        rotate_camera_left=False,              # 左右が逆なら True に
        source_to_detector_distance=1800.0,    # mm（PA/LATの標準SID）
        source_to_isocenter_vertical_distance=900.0,  # ソース→アイソセンタ=SID/2
        pixel_size=0.30,                       # mm（FPDの代表値域）
        sensor_height=2048,                    # 検出器画素
        sensor_width=2048,
        min_alpha=-720, max_alpha=720,         # 可動域は広め
        min_beta=-720,  max_beta=720,
    )

# 3) 1ビュー投影（幾何を動かした直後は projector() を2回）
def render_one(ct: Volume, carm: MobileCArm, alpha, beta, gamma=0.0):
    """
    1つの角度でDRRを生成
    幾何更新直後は projector() を2回実行する必要がある
    """
    print(f"  📸 α={alpha}°, β={beta}° を生成中...")
    carm.move_to(alpha=alpha, beta=beta, gamma=gamma, degrees=True)
    with Projector(ct, carm=carm) as projector:
        _ = projector()          # 1回目は幾何更新の"慣らし"
        img = projector()        # 2回目を採用
    return img

def main():
    """改善版DeepDRRテスト実行"""
    print("🚀 DeepDRR改善版テスト開始")
    print("📋 md_files/answers/answer2.mdの改善案を適用")
    print("="*60)
    
    # ファイルパス
    nifti_path = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    out_dir = Path("/workspace/deepdrr/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 入力NIfTI: {nifti_path}")
    
    # CTボリューム読み込み
    print("🔄 CTボリュームを読み込み中...")
    ct = Volume.from_nifti(str(nifti_path))
    print(f"📊 CTボリューム形状: {ct.data.shape}")
    print(f"📏 CTボリューム間隔: {ct.spacing}")
    
    # 体位の規格化
    print("🔄 患者体位を規格化中...")
    ct.orient_patient(head_first=True, supine=True)        # 体位を規格化（READMEの流れ）
    
    # 胸部撮影向けC-arm作成
    print("🎥 胸部撮影向けC-arm幾何を設定中...")
    print("  - SID: 180cm（胸部撮影標準）")
    print("  - Pixel size: 0.30mm")
    print("  - Sensor: 2048×2048")
    print("  - FOV: 615×615mm")
    
    carm = make_cxr_carm(isocenter=ct.center_in_world)     # 体幹中心をアイソセンタへ
    ct.place_center(carm.isocenter_in_world)
    
    print("\n🎬 DRR投影を実行中...")
    print("  ⚠️  幾何更新直後はprojector()を2回実行します")
    
    # **PA（正面：後前方向）** と **LAT（側面）** を生成
    img_pa  = render_one(ct, carm, alpha=0.0,   beta=0.0,  gamma=0.0)   # PA
    img_lat = render_one(ct, carm, alpha=90.0,  beta=0.0,  gamma=0.0)   # 右側面（必要なら -90）
    
    print(f"📊 PA view値範囲: {img_pa.min():.6f} - {img_pa.max():.6f}")
    print(f"📊 LAT view値範囲: {img_lat.min():.6f} - {img_lat.max():.6f}")
    print(f"📐 DRR画像サイズ: {img_pa.shape}")
    
    # 画像変換とデバッグ情報
    print("\n🔄 臨床X線画像風に変換中...")
    print("  - Beer-Lambert変換（-log）で骨=白、肺=黒に")
    print("  - 1-99.5%パーセンタイル窓でコントラスト改善")
    
    pa_transformed = to_uint16_xray(img_pa, gamma=1.0)
    lat_transformed = to_uint16_xray(img_lat, gamma=1.0)
    
    # 5) 保存（16bit PNG）
    pa_path  = out_dir / "deepdrr_PA_fixed.png"
    lat_path = out_dir / "deepdrr_LAT_fixed.png"
    
    print("\n💾 画像を保存中...")
    imageio.imwrite(pa_path,  pa_transformed)
    imageio.imwrite(lat_path, lat_transformed)
    
    print(f"✅ PA view (改善版): {pa_path}")
    print(f"✅ LAT view (改善版): {lat_path}")
    
    # 6) 目視用の比較図
    print("🎨 比較画像を生成中...")
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(pa_transformed, cmap='gray')
    plt.title("PA (Fixed)\nSID=180cm, 0.30mm/px")
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(lat_transformed, cmap='gray')
    plt.title("LAT (Fixed)\nSID=180cm, 0.30mm/px")
    plt.axis('off')
    
    plt.suptitle('DeepDRR Improved (Chest-like DRR)', fontsize=14)
    viz = out_dir / "deepdrr_fixed_comparison.png"
    plt.tight_layout()
    plt.savefig(viz, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️  比較画像: {viz}")
    
    # 改善前との比較用の統計情報
    print("\n📊 変換統計:")
    print(f"  原画像値域: PA [{img_pa.min():.6f}, {img_pa.max():.6f}], LAT [{img_lat.min():.6f}, {img_lat.max():.6f}]")
    print(f"  変換後値域: PA [0, {pa_transformed.max()}], LAT [0, {lat_transformed.max()}]")
    print(f"  FOV: {2048 * 0.30:.0f}×{2048 * 0.30:.0f} mm")
    
    print("\n" + "="*60)
    print("🎉 DeepDRR改善版テスト完了!")
    print("📋 改善点:")
    print("  ✓ SID 180cm + pixel 0.30mm で胸部全体をカバー") 
    print("  ✓ Beer-Lambert変換で臨床的な白黒表示")
    print("  ✓ パーセンタイル窓でコントラスト改善")
    print("  ✓ 幾何更新後の2回projector()実行で安定化")
    print("="*60)

if __name__ == "__main__":
    main()