#!/usr/bin/env python3
"""
DeepDRR PA左右反転修正版テスト（クリーン環境）
md_files/deepdrr2.mdの指針に基づく実装

技術仕様:
- 座標系: LPS（配列とaffine同期変換）
- 撮影角度: PA=α0°,β0°,γ90°、LAT=α0°,β±90°,γ90°
- PA左右反転: img_pa[:, ::-1].copy() で radiographic view対応
- ヘッドレス実行: matplotlib Agg backend使用
"""

from pathlib import Path
import numpy as np
import imageio.v2 as imageio

# ヘッドレス実行のためmatplotlibバックエンド設定
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# DeepDRR（PyCUDA初期化を最優先）
from deepdrr.projector import Projector
from deepdrr import Volume, MobileCArm

def to_uint16_xray(image: np.ndarray, method: str = "log") -> np.ndarray:
    """DeepDRR出力(検出器読み=相対透過量)をX線表示(骨=白)に整形して16bitに."""
    img = image.astype(np.float32)
    img /= (img.max() + 1e-6)
    if method == "log":
        img = -np.log(img + 1e-6)  # Beer-Lambert風の線積分表示
    elif method == "invert":
        img = 1.0 - img
    img -= img.min()
    img /= (img.max() + 1e-8)
    return (img * 65535.0).astype(np.uint16)

def main():
    print("🏥 DeepDRR PA左右反転修正版テスト")
    print("=====================================")
    
    # 入出力パス
    nifti_path = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    outdir = Path("/workspace/outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    
    if not nifti_path.exists():
        print(f"❌ 入力ファイルが見つかりません: {nifti_path}")
        return
    
    print(f"📁 入力: {nifti_path}")
    print(f"📁 出力: {outdir}")
    
    try:
        # STEP 1: NIfTI読み込みと体位設定
        print("\n🔄 STEP 1: NIfTI読み込みと体位設定")
        carm = MobileCArm()
        ct = Volume.from_nifti(str(nifti_path))
        print(f"  📊 CT形状: {ct.data.shape}")
        
        with Projector(ct, carm=carm) as projector:
            # 患者体位を標準化（頭側先行・仰臥位）
            ct.orient_patient(head_first=True, supine=True)
            ct.place_center(carm.isocenter_in_world)
            print("  ✅ 体位設定完了")
            
            # STEP 2: DeepDRR角度設定とDRR生成
            print("\n🔄 STEP 2: DRR生成（γ=90°で縦向き）")
            
            # PA（正面）: α=0°, β=0°, γ=90°
            print("  📸 PA正面投影...")
            carm.move_to(alpha=0, beta=0, gamma=90, degrees=True)
            img_pa = projector()
            print(f"    形状: {img_pa.shape}, 範囲: {img_pa.min():.6f}-{img_pa.max():.6f}")
            
            # ★PA左右反転（radiographic view対応）
            print("  🔄 PA左右反転適用...")
            img_pa = img_pa[:, ::-1].copy()
            print("    ✅ 左右反転完了（患者の左=画像の右）")
            
            # LAT（左側面）: α=0°, β=90°, γ=90°
            print("  📸 LAT側面投影...")
            carm.move_to(alpha=0, beta=90, gamma=90, degrees=True)
            img_lat = projector()
            print(f"    形状: {img_lat.shape}, 範囲: {img_lat.min():.6f}-{img_lat.max():.6f}")
            
            # STEP 3: X線画像風に変換・保存
            print("\n🔄 STEP 3: X線画像変換・保存")
            
            # 16bit PNG変換
            pa16 = to_uint16_xray(img_pa, method="log")
            lat16 = to_uint16_xray(img_lat, method="log")
            
            # 保存
            pa_path = outdir / "deepdrr_PA_corrected.png"
            lat_path = outdir / "deepdrr_LAT_corrected.png"
            
            imageio.imwrite(pa_path, pa16)
            imageio.imwrite(lat_path, lat16)
            
            print(f"  💾 PA保存: {pa_path}")
            print(f"  💾 LAT保存: {lat_path}")
            
            # STEP 4: 比較可視化作成
            print("\n🔄 STEP 4: 比較可視化作成")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # PA表示
            axes[0].imshow(pa16, cmap='bone', aspect='equal')
            axes[0].set_title('PA正面（左右反転修正済み）\nγ=90° 縦向き', fontsize=12)
            axes[0].axis('off')
            
            # LAT表示
            axes[1].imshow(lat16, cmap='bone', aspect='equal')
            axes[1].set_title('LAT左側面\nγ=90° 縦向き', fontsize=12)
            axes[1].axis('off')
            
            plt.suptitle('DeepDRR PA左右反転修正版\nRadiographic View対応', fontsize=14, weight='bold')
            plt.tight_layout()
            
            comparison_path = outdir / "deepdrr_pa_correction_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  🖼️ 比較画像: {comparison_path}")
            
            # STEP 5: 結果確認
            print("\n✅ PA左右反転修正テスト完了")
            print("=" * 40)
            print("📊 技術仕様確認:")
            print("  - PA角度: α=0°, β=0°, γ=90° (縦向き)")
            print("  - LAT角度: α=0°, β=90°, γ=90° (縦向き)")
            print("  - PA左右: [:, ::-1]で反転（患者左=画像右）")
            print("  - 座標系: LPS正規化")
            print("  - 出力: 16bit PNG (骨=白)")
            
            print("\n📁 生成ファイル:")
            print(f"  - {pa_path.name}")
            print(f"  - {lat_path.name}")
            print(f"  - {comparison_path.name}")
            
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()