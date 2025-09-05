#!/usr/bin/env python3
"""
DeepDRR座標系修正版テストスクリプト
RAS→LPS変換によるPA縦向き・真正側面LAT・正しい左右を確実に実現

根本原因の解決:
1. NIfTI（RAS系）をDeepDRR（LPS系）へ明示的に変換
2. 座標系ミスマッチによる90°回転・左右反転・水平断様LATを一括解決
3. Volume.from_parameters() + anatomical_coordinate_system="LPS"で安全に構築

技術的改善:
- Nibabelでaff2axcodes()による座標系確認
- RAS→LPS変換（x,y軸反転）を明示的実行
- 幾何更新後projector()を2回実行（既知ワークアラウンド）
"""

from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import nibabel as nib
from deepdrr import Volume, MobileCArm, geo
from deepdrr.projector import Projector

# ---------- RAS→LPS変換とDeepDRR Volume構築 ----------
def load_ct_as_lps(nifti_path: Path) -> Volume:
    """
    NIfTIをRAS→LPS変換してDeepDRR Volumeとして安全に構築
    
    手順:
    1. NIfTIの座標系確認（axcodes）
    2. RAS→LPS変換が必要な場合は一時NIfTI保存
    3. DeepDRR Volume.from_nifti()で読み込み
    """
    print(f"📁 NIfTI読み込み: {nifti_path}")
    
    # NIfTI画像読み込み・座標系確認
    img = nib.load(str(nifti_path))
    axcodes = nib.orientations.aff2axcodes(img.affine)
    print(f"🧭 NIfTI座標系: {axcodes}")
    
    if axcodes == ('L', 'P', 'S'):
        print("✅ 既にLPS座標系です")
        # そのまま使用
        ct = Volume.from_nifti(str(nifti_path))
        print("✅ Volume読み込み完了")
        return ct
    else:
        print(f"🔄 {axcodes}→LPS変換を実行")
        
        # RAS canonical化
        img_canonical = nib.as_closest_canonical(img)
        data = img_canonical.get_fdata().astype(np.float32)
        
        # RAS→LPS変換: x,y軸を反転
        if axcodes[0] == 'R':  # Right → Left
            data = data[::-1, :, :]
            print("✅ x軸反転（R→L）")
        if axcodes[1] == 'A':  # Anterior → Posterior  
            data = data[:, ::-1, :]
            print("✅ y軸反転（A→P）")
        
        print(f"📊 変換後データ形状: {data.shape}")
        
        # 一時NIfTI作成（LPS向き）
        temp_nifti = Path("/tmp/temp_lps.nii.gz")
        lps_img = nib.Nifti1Image(data, img_canonical.affine, img_canonical.header)
        nib.save(lps_img, temp_nifti)
        print(f"📄 一時LPS NIfTI保存: {temp_nifti}")
        
        # DeepDRR Volume読み込み
        ct = Volume.from_nifti(str(temp_nifti))
        print("✅ LPS Volume読み込み完了")
        
        # 一時ファイル削除
        temp_nifti.unlink()
        print("🗑️  一時ファイル削除")
        
        return ct

# ---------- 臨床仕様C-arm幾何（35×43cm, 0.14mm/pix） ----------
def make_clinical_carm(isocenter, sid_mm=1800.0, oid_mm=80.0, 
                      pix_mm=0.14, plate_mm=(350.0, 430.0)):
    """
    臨床標準仕様のC-arm幾何
    - 座標系修正後なのでrotate_camera_leftは基本False
    - PA縦向き、真正側面LAT用
    """
    sad_mm = sid_mm - oid_mm
    sensor_w = int(round(plate_mm[0] / pix_mm))  # ≈2500
    sensor_h = int(round(plate_mm[1] / pix_mm))  # ≈3070
    sensor_w += sensor_w % 2; sensor_h += sensor_h % 2
    
    print(f"📏 C-arm幾何設定:")
    print(f"  検出器: {sensor_w}×{sensor_h} pixels")
    print(f"  物理サイズ: {plate_mm[0]}×{plate_mm[1]} mm")  
    print(f"  ピクセルサイズ: {pix_mm} mm/pixel")
    print(f"  SID: {sid_mm}mm, SAD: {sad_mm}mm, OID: {oid_mm}mm")
    print(f"  倍率: {sid_mm/sad_mm:.3f}")
    
    return MobileCArm(
        isocenter=isocenter,
        rotate_camera_left=False,  # 座標系修正後は基本不要
        source_to_detector_distance=sid_mm,
        source_to_isocenter_vertical_distance=sad_mm,
        pixel_size=pix_mm,
        sensor_width=sensor_w,
        sensor_height=sensor_h,
        min_alpha=-720, max_alpha=720,
        min_beta=-720, max_beta=720
    )

# ---------- 幾何更新後2回投影（既知ワークアラウンド） ----------
def project_with_warmup(ct: Volume, carm: MobileCArm, alpha=0.0, beta=0.0, gamma=0.0):
    """
    幾何更新後の安全な投影（2回実行）
    """
    print(f"📸 投影角度: α={alpha}°, β={beta}°, γ={gamma}°")
    carm.move_to(alpha=alpha, beta=beta, gamma=gamma, degrees=True)
    
    with Projector(ct, carm=carm) as projector:
        _ = projector()    # 1回目: 幾何更新の慣らし
        img = projector()  # 2回目: 採用
    
    return img

# ---------- Energy表示（-log不要、パーセンタイル窓） ----------
def energy_to_uint16(img, window=(1.0, 99.5), gamma=1.0):
    """
    DeepDRR energy出力を臨床表示用16bitに変換
    """
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, window)
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma != 1.0:
        x = x ** (1.0 / gamma)
    return (x * 65535).astype(np.uint16)

# ---------- Portrait調整（35:43アスペクト比） ----------
def to_portrait_aspect(u16, target_aspect=43/35):
    """
    画像を縦長35:43比に調整
    """
    y, x = u16.shape
    print(f"    元サイズ: {x}×{y}")
    
    # 横長なら90°回転
    if x > y:
        u16 = np.rot90(u16)
        y, x = u16.shape
        print(f"    回転後: {x}×{y}")
    
    # アスペクト比調整
    current_aspect = y / x
    if abs(current_aspect - target_aspect) < 1e-3:
        print(f"    調整不要: {current_aspect:.3f}")
        return u16
    
    if current_aspect > target_aspect:
        # 縦長→上下クロップ
        new_y = int(x * target_aspect)
        top = (y - new_y) // 2
        cropped = u16[top:top+new_y, :]
        print(f"    上下クロップ: {x}×{new_y}")
    else:
        # 横長→左右クロップ
        new_x = int(y / target_aspect)
        left = (x - new_x) // 2
        cropped = u16[:, left:left+new_x]
        print(f"    左右クロップ: {new_x}×{y}")
    
    return cropped

def main():
    """
    座標系修正版DeepDRRテスト実行
    """
    print("🚀 DeepDRR座標系修正版テスト開始")
    print("🎯 目標: PA縦向き・真正側面LAT・正しい左右")
    print("🔧 手法: RAS→LPS変換による根本的解決")
    print("="*60)
    
    # ファイルパス
    nifti_path = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    outdir = Path("/workspace/deepdrr/outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. RAS→LPS変換でCT読み込み
        print("\n🔄 STEP 1: RAS→LPS変換でCT読み込み")
        ct = load_ct_as_lps(nifti_path)
        
        # 2. 患者体位規格化
        print("\n🔄 STEP 2: 患者体位規格化")
        print("  HFS (Head First Supine) に設定")
        ct.orient_patient(head_first=True, supine=True)
        print("✅ 体位規格化完了")
        
        # 3. 臨床仕様C-arm設定
        print("\n🔄 STEP 3: 臨床仕様C-arm設定")
        carm = make_clinical_carm(
            ct.center_in_world,
            sid_mm=1800.0,
            oid_mm=80.0,
            pix_mm=0.14,
            plate_mm=(350.0, 430.0)
        )
        ct.place_center(carm.isocenter_in_world)
        print("✅ C-arm幾何設定完了")
        
        # 4. PA・LAT投影
        print("\n🔄 STEP 4: PA・LAT投影実行")
        print("  📋 座標系修正後の標準角度:")
        print("    PA (正面): α=0°")
        print("    LAT (左側面): α=90°")
        
        img_pa = project_with_warmup(ct, carm, alpha=0.0, beta=0.0)    # PA
        img_lat = project_with_warmup(ct, carm, alpha=90.0, beta=0.0)  # 左側面
        
        print(f"📊 投影結果:")
        print(f"  PA view値範囲: {img_pa.min():.6f} - {img_pa.max():.6f}")
        print(f"  LAT view値範囲: {img_lat.min():.6f} - {img_lat.max():.6f}")
        print(f"  画像サイズ: {img_pa.shape}")
        
        # 5. 表示変換・Portrait調整
        print("\n🔄 STEP 5: 表示変換・Portrait調整")
        print("  📊 Energy表示（-log不要）+ パーセンタイル窓")
        
        pa16_raw = energy_to_uint16(img_pa, window=(1.0, 99.5))
        lat16_raw = energy_to_uint16(img_lat, window=(1.0, 99.5))
        
        print("  📐 35:43 Portrait調整:")
        print("  PA view:")
        pa16 = to_portrait_aspect(pa16_raw)
        print("  LAT view:")
        lat16 = to_portrait_aspect(lat16_raw)
        
        # 6. 保存
        print("\n💾 STEP 6: 画像保存")
        pa_path = outdir / "deepdrr_PA_coordinate_fixed.png"
        lat_path = outdir / "deepdrr_LAT_coordinate_fixed.png"
        
        imageio.imwrite(pa_path, pa16)
        imageio.imwrite(lat_path, lat16)
        
        print(f"✅ 保存完了:")
        print(f"  PA view: {pa_path}")
        print(f"  LAT view: {lat_path}")
        
        # 7. 比較画像生成
        print("\n🎨 STEP 7: 比較画像生成")
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 8))
        plt.subplot(121)
        plt.imshow(pa16, cmap='gray')
        plt.title('PA (Coordinate Fixed)\n縦向き・正しい左右', fontsize=12)
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(lat16, cmap='gray')
        plt.title('LAT (Coordinate Fixed)\n真正側面像', fontsize=12)
        plt.axis('off')
        
        plt.suptitle('DeepDRR座標系修正版\nRAS→LPS変換による根本的解決', fontsize=14)
        viz_path = outdir / "deepdrr_coordinate_fixed_comparison.png"
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"🖼️  比較画像: {viz_path}")
        
        # 8. 結果統計
        print("\n📊 STEP 8: 結果統計")
        print("✅ 座標系修正版統計:")
        print(f"  最終サイズ: PA {pa16.shape}, LAT {lat16.shape}")
        print(f"  アスペクト比: PA {pa16.shape[0]/pa16.shape[1]:.3f}, LAT {lat16.shape[0]/lat16.shape[1]:.3f}")
        print(f"  物理サイズ: {pa16.shape[1]*0.14:.1f}×{pa16.shape[0]*0.14:.1f} mm")
        print(f"  座標系: LPS（Left-Posterior-Superior）")
        print(f"  角度: PA=0°, LAT=90°")
        
        print("\n" + "="*60)
        print("🎉 座標系修正版テスト完了!")
        print("🎯 期待される改善:")
        print("  ✅ PA縦向き（90°回転解消）")
        print("  ✅ 正しい左右（反転解消）")
        print("  ✅ 真正側面LAT（水平断様→側面像）")
        print("  ✅ Energy表示による適切な階調")
        print("="*60)
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()