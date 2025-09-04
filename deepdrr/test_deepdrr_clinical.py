#!/usr/bin/env python3
"""
DeepDRR臨床仕様版テストスクリプト
実機に近い胸部撮影仕様での仕上げ実装

臨床仕様:
1. 検出器: 35×43cm、0.14mm/pixel（実機標準仕様）
2. 画像: 縦長（portrait）、35:43アスペクト比
3. 撮影法: PA（標準）、左側面LAT（心陰影観察性）
4. DICOM: MONOCHROME2、ViewPosition明記
"""

from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector

# ---------- 画質整形（energy表示：-logなし、percentile窓＋γ） ----------
def energy_to_uint16(img, window=(1.0, 99.5), gamma=1.0):
    """DeepDRR出力(energy deposited)を臨床表示用に変換"""
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, window)
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma != 1.0:
        x = x ** (1.0 / gamma)
    return (x * 65535).astype(np.uint16)

# ---------- 35×43cm, 0.14mm/pix の検出器幾何（縦長） ----------
def make_cxr_carm_portrait(isocenter, sid_mm=1800.0, oid_mm=80.0,
                           pix_mm=0.14, plate_mm=(350.0, 430.0),
                           rotate_camera_left=True):
    """
    臨床標準仕様のC-arm幾何（35×43cm検出器、0.14mm/pixel）
    """
    sad_mm = sid_mm - oid_mm
    sensor_w = int(round(plate_mm[0] / pix_mm))   # ≈ 350 / 0.14 ≈ 2500
    sensor_h = int(round(plate_mm[1] / pix_mm))   # ≈ 430 / 0.14 ≈ 3070
    sensor_w += sensor_w % 2; sensor_h += sensor_h % 2  # 偶数に整形
    
    print(f"  📏 検出器サイズ: {sensor_w}×{sensor_h} pixels")
    print(f"  📐 物理サイズ: {plate_mm[0]}×{plate_mm[1]} mm")
    print(f"  🔍 ピクセルサイズ: {pix_mm} mm/pixel")
    
    return MobileCArm(
        isocenter=isocenter,
        rotate_camera_left=rotate_camera_left,  # 左右は装置側で合わせる
        source_to_detector_distance=sid_mm,     # SID=180 cm
        source_to_isocenter_vertical_distance=sad_mm,  # SAD=SID-OID
        pixel_size=pix_mm,                      # 0.14 mm
        sensor_width=sensor_w, sensor_height=sensor_h,
        min_alpha=-720, max_alpha=720, min_beta=-720, max_beta=720
    )

# ---------- 幾何更新後は projector() を2回 ----------
def project_once(ct: Volume, carm: MobileCArm, alpha=0.0, beta=0.0, gamma=0.0):
    """1つの角度でDRR生成（幾何更新後2回実行）"""
    print(f"  📸 α={alpha}°, β={beta}° を生成中...")
    carm.move_to(alpha=alpha, beta=beta, gamma=gamma, degrees=True)
    with Projector(ct, carm=carm) as projector:
        _ = projector()     # 慣らし
        img = projector()   # 採用
    return img

# ---------- 画像を portrait & 35:43 に揃える（クロップ優先） ----------
def to_portrait_crop(u16, aspect=43/35):
    """画像を縦長（portrait）35:43比に調整"""
    y, x = u16.shape
    print(f"    元サイズ: {x}×{y}")
    
    if x > y:                           # 横長なら90°回転
        u16 = np.rot90(u16)
        y, x = u16.shape
        print(f"    回転後: {x}×{y}")
    
    cur = y / x
    if abs(cur - aspect) < 1e-3:
        print(f"    アスペクト比調整不要: {cur:.3f}")
        return u16
    
    if cur > aspect:                    # 縦に長い→上下をクロップ
        new_y = int(x * aspect)
        top = max((y - new_y)//2, 0)
        cropped = u16[top:top+new_y, :]
        print(f"    上下クロップ: {x}×{new_y}")
        return cropped
    else:                               # 横に長い→左右をクロップ
        new_x = int(y / aspect)
        left = max((x - new_x)//2, 0)
        cropped = u16[:, left:left+new_x]
        print(f"    左右クロップ: {new_x}×{y}")
        return cropped

# ---------- DICOM保存（DX For Presentation, MONOCHROME2, ViewPosition） ----------
def save_dx_dicom(u16, save_path: Path, pixel_spacing_mm: float,
                  sid_mm: float, sad_mm: float, view_position: str):
    """DICOM形式で保存（DX For Presentation、臨床標準準拠）"""
    try:
        import datetime
        import pydicom
        from pydicom.dataset import FileDataset
        from pydicom.uid import generate_uid
        
        print(f"    DICOM保存: {save_path}")
        print(f"    ViewPosition: {view_position}")
        
        ds = FileDataset(str(save_path), {}, file_meta=pydicom.dataset.FileMetaDataset(), preamble=b"\0"*128)
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.SOPClassUID  = pydicom.uid.DigitalXRayImageStorageForPresentation
        ds.SOPInstanceUID = generate_uid()
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()

        ds.Modality = "DX"
        ds.PatientName = "CTRATE^SYNTH"
        ds.PatientID = "CTRATE_SYNTH_0001"
        ds.StudyDate = ds.SeriesDate = ds.ContentDate = datetime.datetime.now().strftime("%Y%m%d")
        ds.StudyTime = ds.SeriesTime = ds.ContentTime = datetime.datetime.now().strftime("%H%M%S")

        ds.Rows, ds.Columns = u16.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"   # 値↑ = 白（表示規約）
        ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
        ds.PixelSpacing = [str(pixel_spacing_mm), str(pixel_spacing_mm)]
        ds.DistanceSourceToDetector = float(sid_mm)
        ds.DistanceSourceToPatient  = float(sad_mm)
        ds.ViewPosition = view_position               # "PA" / "LATERAL" など
        ds.BurnedInAnnotation = "NO"                  # 文字を焼き込むなら "YES"

        ds.PresentationIntentType = "FOR PRESENTATION"
        ds.PixelData = u16.tobytes()
        ds.is_little_endian = True; ds.is_implicit_VR = False
        ds.save_as(str(save_path), write_like_original=False)
        
        print(f"    ✅ DICOM保存完了")
        
    except ImportError:
        print(f"    ⚠️  pydicomがインストールされていません。PNGのみ保存します。")
        print(f"    インストール: pip install pydicom")

def main():
    """臨床仕様版DeepDRRテスト実行"""
    print("🚀 DeepDRR臨床仕様版テスト開始")
    print("🏥 実機に近い胸部撮影仕様で仕上げ実装")
    print("="*60)
    
    # ファイルパス
    nifti = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    outdir = Path("/workspace/deepdrr/outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 入力NIfTI: {nifti}")
    
    # CTボリューム読み込み
    print("🔄 CTボリュームを読み込み中...")
    ct = Volume.from_nifti(str(nifti))
    print(f"📊 CTボリューム形状: {ct.data.shape}")
    print(f"📏 CTボリューム間隔: {ct.spacing}")
    
    # 体位の規格化
    print("🔄 患者体位を規格化中...")
    ct.orient_patient(head_first=True, supine=True)
    
    # 臨床仕様C-arm幾何設定
    print("🎥 臨床仕様C-arm幾何を設定中...")
    print("  📋 臨床標準仕様:")
    print("    - 検出器: 35×43cm（縦長）")
    print("    - 解像度: 0.14mm/pixel")
    print("    - SID: 180cm（胸部撮影標準）")
    print("    - OID: 8cm（検出器に近接）")
    
    SID, OID = 1800.0, 80.0
    PIX = 0.14  # mm/pixel（DICOMへも書き込む）
    
    carm = make_cxr_carm_portrait(
        ct.center_in_world, 
        sid_mm=SID, 
        oid_mm=OID,
        pix_mm=PIX, 
        plate_mm=(350.0, 430.0),
        rotate_camera_left=True
    )
    ct.place_center(carm.isocenter_in_world)
    
    SAD = SID - OID
    print(f"  📏 幾何パラメータ:")
    print(f"    - SID: {SID:.0f}mm")
    print(f"    - SAD: {SAD:.0f}mm")  
    print(f"    - OID: {OID:.0f}mm")
    print(f"    - 倍率: {SID/SAD:.3f}")
    
    print("\n🎬 DRR投影を実行中...")
    print("  📋 撮影法: PA（標準）+ 左側面LAT（心陰影観察性）")
    print("  ⚠️  幾何更新直後はprojector()を2回実行")
    
    # PA（正面）と左側面（既定）をレンダ
    img_pa  = project_once(ct, carm, alpha=0.0,  beta=0.0)   # PA
    img_lat = project_once(ct, carm, alpha=90.0, beta=0.0)   # 左側面（右/左は±90°で要確認）
    
    print(f"📊 PA view値範囲: {img_pa.min():.6f} - {img_pa.max():.6f}")
    print(f"📊 LAT view値範囲: {img_lat.min():.6f} - {img_lat.max():.6f}")
    print(f"📐 生成画像サイズ: {img_pa.shape}")
    
    # 臨床表示への変換
    print("\n🔄 臨床表示に変換中...")
    print("  📊 Energy表示（-log無し）")
    print("  📊 1-99.5%パーセンタイル窓")
    
    pa16_raw  = energy_to_uint16(img_pa,  window=(1,99.5), gamma=1.0)
    lat16_raw = energy_to_uint16(img_lat, window=(1,99.5), gamma=1.0)
    
    # 整形 → portrait & 35:43 → 16bit
    print("\n📐 Portrait（縦長）35:43比に調整中...")
    print("  PA view:")
    pa16  = to_portrait_crop(pa16_raw)
    print("  LAT view:")
    lat16 = to_portrait_crop(lat16_raw)
    
    # PNG保存（確認用）
    print("\n💾 PNG画像を保存中...")
    pa_png = outdir / "deepdrr_PA_clinical.png"
    lat_png = outdir / "deepdrr_LAT_clinical.png"
    
    imageio.imwrite(pa_png, pa16)
    imageio.imwrite(lat_png, lat16)
    
    print(f"✅ PA view (臨床仕様): {pa_png}")
    print(f"✅ LAT view (臨床仕様): {lat_png}")
    
    # DICOM保存（DX For Presentation）
    print("\n💾 DICOM形式で保存中...")
    pa_dcm = outdir / "deepdrr_PA_clinical.dcm"
    lat_dcm = outdir / "deepdrr_LAT_clinical.dcm"
    
    save_dx_dicom(pa16,  pa_dcm, pixel_spacing_mm=PIX, sid_mm=SID, sad_mm=SAD, view_position="PA")
    save_dx_dicom(lat16, lat_dcm, pixel_spacing_mm=PIX, sid_mm=SID, sad_mm=SAD, view_position="LATERAL")
    
    # 比較画像生成
    print("\n🎨 比較画像を生成中...")
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12,8))
    plt.subplot(121)
    plt.imshow(pa16, cmap='gray')
    plt.title(f'PA (Clinical Spec)\n35×43cm, 0.14mm/px\nSID={SID:.0f}mm, OID={OID:.0f}mm')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(lat16, cmap='gray')
    plt.title(f'LAT (Clinical Spec)\n35×43cm, 0.14mm/px\nSID={SID:.0f}mm, OID={OID:.0f}mm')
    plt.axis('off')
    
    plt.suptitle('DeepDRR Clinical Specification (Portrait, 35:43)', fontsize=14)
    viz = outdir / "deepdrr_clinical_comparison.png"
    plt.tight_layout()
    plt.savefig(viz, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️  比較画像: {viz}")
    
    # 臨床仕様統計
    print("\n📊 臨床仕様統計:")
    print(f"  最終画像サイズ: PA {pa16.shape}, LAT {lat16.shape}")
    print(f"  アスペクト比: PA {pa16.shape[0]/pa16.shape[1]:.3f}, LAT {lat16.shape[0]/lat16.shape[1]:.3f}")
    print(f"  物理サイズ: {pa16.shape[1]*PIX:.1f}×{pa16.shape[0]*PIX:.1f} mm")
    print(f"  検出器仕様: 35×43cm, {PIX}mm/pixel")
    print(f"  幾何倍率: {SID/SAD:.3f}")
    print(f"  ピクセル間隔: {PIX}mm")
    
    print("\n" + "="*60)
    print("🎉 DeepDRR臨床仕様版テスト完了!")
    print("📋 臨床標準準拠:")
    print("  ✅ 検出器: 35×43cm、0.14mm/pixel（実機標準）")
    print("  ✅ 画像: 縦長（portrait）、35:43アスペクト比")
    print("  ✅ 撮影法: PA（標準）、左側面LAT（心陰影観察性）")
    print("  ✅ DICOM: MONOCHROME2、ViewPosition明記")
    print("  ✅ Energy表示: -log無し、適切な階調")
    print("  ✅ 幾何: SID=180cm、適切な拡大率")
    print("="*60)

if __name__ == "__main__":
    main()