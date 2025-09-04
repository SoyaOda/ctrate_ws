#!/usr/bin/env python3
"""
DeepDRR最終修正版テストスクリプト
白黒・左右・拡大の3つの問題を同時に解決

修正点:
1. 白黒: DeepDRR出力(energy deposited)を-log無しで表示
2. 左右: rotate_camera_left=Trueで装置側反転
3. 拡大: SAD=1720mm(SIDに近い)でOID=80mmに縮小
"""

from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector  # PyCUDA 初期化
import matplotlib.pyplot as plt

# 1) DeepDRR出力(=平均エネルギー堆積)を臨床風に表示
def to_uint16_xray_energy(img: np.ndarray, window=(1.0, 99.5), gamma: float = 1.0) -> np.ndarray:
    """DeepDRR出力(energy deposited)を線形表示で変換（-log無し）"""
    x = img.astype(np.float32)
    # パーセンタイル窓で正規化
    mn, mx = np.percentile(x, window)
    x = np.clip((x - mn) / max(mx - mn, 1e-6), 0, 1)
    # 必要なら階調補正（gamma>1でソフト、<1でハード）
    if gamma != 1.0:
        x = np.power(x, 1.0 / gamma)
    return (x * 65535).astype(np.uint16)

# 2) 胸部PA/LAT向けの幾何（拡大を抑制）
def make_cxr_carm(isocenter, sid_mm=1800.0, sad_mm=1720.0, pix_mm=0.30, res=2048, flip_lr_in_camera=True):
    """
    胸部撮影用C-arm幾何（拡大抑制版）
    SAD を SID に近づけて OID を小さくすることで拡大を抑制
    """
    return MobileCArm(
        isocenter=isocenter,
        rotate_camera_left=flip_lr_in_camera,          # 左右反転を装置側で解決
        source_to_detector_distance=sid_mm,            # SID ≈ 180 cm（標準）
        source_to_isocenter_vertical_distance=sad_mm,  # SAD ≈ 172 cm（OID ≈ 8 cm）
        pixel_size=pix_mm,                             # 0.30 mm/px 程度
        sensor_height=res,
        sensor_width=res,
        min_alpha=-720, max_alpha=720,
        min_beta=-720,  max_beta=720,
    )

def render_one(ct: Volume, carm: MobileCArm, alpha, beta, gamma=0.0, degrees=True):
    """1つの角度でDRR生成（幾何更新後2回実行）"""
    print(f"  📸 α={alpha}°, β={beta}° を生成中...")
    carm.move_to(alpha=alpha, beta=beta, gamma=gamma, degrees=degrees)
    with Projector(ct, carm=carm) as projector:
        _ = projector()            # 1回目は幾何の更新が反映されないことがある
        img = projector()          # 2回目を採用（既知の回避策）
    return img

def main():
    """最終修正版DeepDRRテスト実行"""
    print("🚀 DeepDRR最終修正版テスト開始")
    print("🔧 白黒・左右・拡大の3つの問題を同時に解決")
    print("="*60)
    
    # ファイルパス
    nifti = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    out = Path("/workspace/deepdrr/outputs")
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 入力NIfTI: {nifti}")
    
    # CTボリューム読み込み
    print("🔄 CTボリュームを読み込み中...")
    ct = Volume.from_nifti(str(nifti))
    print(f"📊 CTボリューム形状: {ct.data.shape}")
    print(f"📏 CTボリューム間隔: {ct.spacing}")
    
    # 体位の規格化
    print("🔄 患者体位を規格化中...")
    ct.orient_patient(head_first=True, supine=True)
    
    # 修正版C-arm幾何設定
    print("🎥 修正版C-arm幾何を設定中...")
    print("  📏 SID: 180cm（胸部撮影標準）")
    print("  📏 SAD: 172cm（拡大抑制のためSIDに近い）")
    print("  📏 OID: 8cm（検出器に近接）")
    print("  📏 Magnification: SID/SOD ≈ 1800/1720 ≈ 1.05倍")
    print("  🔄 rotate_camera_left=True（左右反転を装置側で解決）")
    print("  🖼️  Pixel size: 0.30mm, Sensor: 2048×2048")
    print("  📐 FOV: 614×614mm")
    
    # SADをSIDに近づける（例：SID=1800, SAD=1720 ⇒ OID=80mm ≒ 検出器に近接）
    carm = make_cxr_carm(
        isocenter=ct.center_in_world, 
        sid_mm=1800.0, 
        sad_mm=1720.0,
        pix_mm=0.30, 
        res=2048, 
        flip_lr_in_camera=True
    )
    ct.place_center(carm.isocenter_in_world)
    
    print("\n🎬 DRR投影を実行中...")
    print("  ⚠️  幾何更新直後はprojector()を2回実行")
    
    # PA（正面）と LAT（側面）
    img_pa  = render_one(ct, carm, alpha=0.0,  beta=0.0,  gamma=0.0)   # PA
    img_lat = render_one(ct, carm, alpha=90.0, beta=0.0,  gamma=0.0)   # 右側面（向きが逆なら -90°）
    
    print(f"📊 PA view値範囲: {img_pa.min():.6f} - {img_pa.max():.6f}")
    print(f"📊 LAT view値範囲: {img_lat.min():.6f} - {img_lat.max():.6f}")
    print(f"📐 DRR画像サイズ: {img_pa.shape}")
    
    # 修正版画像変換（energy表示、-log無し）
    print("\n🔄 Energy表示に変換中...")
    print("  ✅ -log変換を使わない（DeepDRR出力=energy deposited前提）")
    print("  📊 1-99.5%パーセンタイル窓でコントラスト調整")
    
    pa_energy = to_uint16_xray_energy(img_pa, window=(1, 99.5), gamma=1.0)
    lat_energy = to_uint16_xray_energy(img_lat, window=(1, 99.5), gamma=1.0)
    
    # 16bit PNG 保存（energy表示）
    pa_png  = out / "deepdrr_PA_final.png"
    lat_png = out / "deepdrr_LAT_final.png"
    
    print(f"\n💾 最終修正版画像を保存中...")
    imageio.imwrite(pa_png,  pa_energy)
    imageio.imwrite(lat_png, lat_energy)
    
    print(f"✅ PA view (最終版): {pa_png}")
    print(f"✅ LAT view (最終版): {lat_png}")
    
    # 確認用の表示
    print("🎨 比較画像を生成中...")
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(pa_energy, cmap="gray")
    plt.title("PA (Final)\nEnergy Display, L/R Fixed, Reduced Magnification")
    plt.axis("off")
    
    plt.subplot(122)
    plt.imshow(lat_energy, cmap="gray")
    plt.title("LAT (Final)\nEnergy Display, L/R Fixed, Reduced Magnification")
    plt.axis("off")
    
    plt.suptitle('DeepDRR Final (3 Issues Fixed)', fontsize=14)
    viz = out / "deepdrr_final_comparison.png"
    plt.tight_layout()
    plt.savefig(viz, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"🖼️  比較画像: {viz}")
    
    # 修正内容の統計情報
    print("\n📊 修正統計:")
    print(f"  原画像値域: PA [{img_pa.min():.6f}, {img_pa.max():.6f}], LAT [{img_lat.min():.6f}, {img_lat.max():.6f}]")
    print(f"  変換後値域: PA [0, {pa_energy.max()}], LAT [0, {lat_energy.max()}]")
    print(f"  幾何倍率: SID/SOD = 1800/1720 ≈ {1800/1720:.2f}倍")
    print(f"  FOV: {2048 * 0.30:.0f}×{2048 * 0.30:.0f} mm")
    
    print("\n" + "="*60)
    print("🎉 DeepDRR最終修正版テスト完了!")
    print("📋 解決した問題:")
    print("  ✅ 白黒逆転: -log無しのenergy表示で正しい階調")
    print("  ✅ 左右反転: rotate_camera_left=Trueで装置側反転")
    print("  ✅ 過度の拡大: SAD=172cmでOID=8cm、倍率1.05倍に抑制")
    print("  ✅ 幾何更新: projector()を2回実行で安定化")
    print("="*60)

if __name__ == "__main__":
    main()