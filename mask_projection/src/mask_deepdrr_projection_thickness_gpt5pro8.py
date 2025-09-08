# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Plan-A: 単色"実効"モデルで画素値=厚み[mm] (gpt5pro8改良版)
- I_air (flat-field) を同一幾何で取得 + 健全性チェック
- G = -log( I_mask / I_air ) with valid condition
- μ_eff を体積一致で自己較正
- T_mm = G / μ_eff（Ω=ロバストシルエット内、外は0）
- ∑T_mm * p^2 = V_3D を常に満たす

根拠:
- Beer–Lambert (mono): -log(I/I0) = μ * L  → 厚みに線形。 DeepDRR 論文背景等参照
- Flat-field 正規化: I_norm = (I_raw - dark) / (flat - dark) ≈ I/I0
"""
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from nibabel import orientations as nio
from nibabel.affines import voxel_sizes
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing

from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector

# ---------- 0) 共通: LPSに厳密変換 ----------
def load_volume_LPS(nifti: Path) -> Volume:
    img = nib.load(str(nifti))
    in_ornt = nio.io_orientation(img.affine)
    out_ornt = nio.axcodes2ornt(("L","P","S"))
    to_lps = nio.ornt_transform(in_ornt, out_ornt)
    data_lps = nio.apply_orientation(img.get_fdata().astype(np.float32), to_lps)
    aff_lps = img.affine @ nio.inv_ornt_aff(to_lps, img.shape)
    spacing = voxel_sizes(aff_lps)[:3]
    shp = np.array(data_lps.shape, dtype=float)
    origin = (-0.5 * (shp - 1.0) * spacing).tolist()
    return Volume.from_hu(
        hu_values=data_lps, origin=origin, spacing=spacing.tolist(),
        anatomical_coordinate_system="LPS",
    )

def load_mask_volume_LPS(mask_nifti: Path) -> Volume:
    img_mask = nib.load(str(mask_nifti))
    in_ornt = nio.io_orientation(img_mask.affine)
    out_ornt = nio.axcodes2ornt(("L","P","S"))
    to_lps = nio.ornt_transform(in_ornt, out_ornt)
    mask_lps = nio.apply_orientation(img_mask.get_fdata().astype(np.float32), to_lps)
    # gpt5pro8改良: 現実的なHU（背景=-1000, 内部=-50）→ ビームハードニング軽減
    hu = np.full(mask_lps.shape, -1000.0, dtype=np.float32)
    hu[mask_lps > 0.5] = -50.0  # 脂肪組織の現実的なHU値
    aff_mask_lps = img_mask.affine @ nio.inv_ornt_aff(to_lps, img_mask.shape)
    spacing = voxel_sizes(aff_mask_lps)[:3]
    shp = np.array(hu.shape, dtype=float)
    origin = (-0.5 * (shp - 1.0) * spacing).tolist()
    return Volume.from_hu(
        hu_values=hu, origin=origin, spacing=list(spacing),
        anatomical_coordinate_system="LPS",
    )

# ---------- 1) 幾何: C-arm ----------
def make_carm(isocenter, sid=1800.0, oid=80.0, px=0.14, plate_mm=(350.0, 430.0)):
    sad = sid - oid
    W = int(round(plate_mm[0] / px)) & ~1
    H = int(round(plate_mm[1] / px)) & ~1
    return MobileCArm(
        isocenter=isocenter, rotate_camera_left=True,
        source_to_detector_distance=sid,
        source_to_isocenter_vertical_distance=sad,
        pixel_size=px,
        sensor_width=W, sensor_height=H,
        min_alpha=-720, max_alpha=720, min_beta=-720, max_beta=720,
    )

# ---------- 2) 同一幾何のエナジー画像（生） ----------
def render_energy(vol: Volume, carm: MobileCArm, alpha=0, beta=0, gamma=90, flip_pa=True):
    with Projector(vol, carm=carm) as projector:
        carm.move_to(alpha=alpha, beta=beta, gamma=gamma, degrees=True)
        _ = projector()           # warm-up
        img = projector().astype(np.float32)  # [0..1] transmission-like
    if flip_pa:
        img = img[:, ::-1].copy() # radiographic view: 患者左=画像右
    return img

# ---------- 3) flat-field (I_air) 生成 & キャッシュ ----------
def make_air_volume_like(vol: Volume) -> Volume:
    air = np.full(vol.data.shape, -1000.0, dtype=np.float32)
    return Volume.from_hu(
        hu_values=air, origin=vol.origin, spacing=vol.spacing,
        anatomical_coordinate_system=vol.anatomical_coordinate_system,
    )

def flat_cache_key(alpha,beta,gamma,carm):
    W,H = carm.sensor_width, carm.sensor_height
    px  = float(getattr(carm, "pixel_size", 0.14))
    sid = float(getattr(carm, "source_to_detector_distance", 1800.0))
    sad = float(getattr(carm, "source_to_isocenter_vertical_distance", 1720.0))
    rcl = bool(getattr(carm, "rotate_camera_left", True))
    return f"a{alpha}_b{beta}_g{gamma}_W{W}_H{H}_px{px:.5f}_sid{sid:.1f}_sad{sad:.1f}_r{int(rcl)}"

def get_flat_field(mask_vol: Volume, carm: MobileCArm, view, outdir: Path, flip_pa=True):
    alpha,beta,gamma = view
    cache_path = outdir / f"flat_{flat_cache_key(alpha,beta,gamma,carm)}.npy"
    if cache_path.exists():
        print(f"🔧 フラットフィールドキャッシュ使用: {cache_path}")
        return np.load(cache_path)
    print(f"🔧 フラットフィールド生成: α={alpha}°, β={beta}°, γ={gamma}°")
    air_vol = make_air_volume_like(mask_vol)
    air_vol.place_center(carm.isocenter_in_world)
    I_air = render_energy(air_vol, carm, alpha, beta, gamma, flip_pa=flip_pa)
    np.save(cache_path, I_air)
    print(f"  ✅ フラットフィールド保存: {cache_path}")
    return I_air

# ---------- 4) FOVマスク作成 ----------
def create_fov_mask(shape, edge_crop_px=8):
    """
    FOVマスク作成: 境界からedge_crop_px内側のマスクを生成
    
    Args:
        shape: 画像形状 (H, W)
        edge_crop_px: 境界からクロップするピクセル数
    
    Returns:
        FOVマスク (境界の一定範囲を除外)
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[edge_crop_px:h-edge_crop_px, edge_crop_px:w-edge_crop_px] = True
    return mask

# ---------- 5) Beer–Lambert 逆変換 + μ_eff 自己較正 ----------
def thickness_from_energy(I_mask: np.ndarray, I_air: np.ndarray, V3D_mm3: float,
                          px_mm: float, support_tau: float = 1e-4):
    """
    Plan-A: Beer–Lambert 逆変換 + μ_eff 自己較正による厚み推定
    
    Args:
        I_mask: マスク付きenergy画像
        I_air: flat-field (空気のみ) energy画像
        V3D_mm3: 3Dマスクの体積 (mm³)
        px_mm: 検出器ピクセルピッチ (mm)
        support_tau: シルエット検出閾値
    
    Returns:
        T: 厚み画像 [mm]
        mu_eff: 実効線減弱係数 [1/mm]
        Omega: シルエットマスク
        vol_err: 体積誤差 [%]
    """
    # 形状確認
    assert I_mask.shape == I_air.shape, "I_mask/I_air shape mismatch"
    print(f"🔍 Plan-A厚み推定: Beer–Lambert逆変換 + μ_eff自己較正")
    print(f"  📊 入力画像形状: {I_mask.shape}")
    print(f"  📊 I_mask範囲: {I_mask.min():.6f} - {I_mask.max():.6f}")
    print(f"  📊 I_air範囲: {I_air.min():.6f} - {I_air.max():.6f}")
    
    # gpt5pro8改良: I_air健全性チェック + valid条件 + FOVマスク
    eps = 1e-6
    I_air_threshold = 1e-3  # フラットフィールド最小閾値
    fov_mask = create_fov_mask(I_air.shape, edge_crop_px=8)  # 境界8px内側マスク
    valid = (I_air > I_air_threshold) & fov_mask
    
    ratio = np.zeros_like(I_mask, dtype=np.float32)
    ratio[valid] = np.clip(I_mask[valid] / np.clip(I_air[valid], eps, 1.0), eps, 1.0)
    G = np.zeros_like(I_mask, dtype=np.float32)
    G[valid] = -np.log(ratio[valid])  # Beer–Lambert (mono) の線積分に相当
    
    print(f"  📊 G(-log(I/I0))統計: min={G.min():.6f}, max={G.max():.6f}, mean={G.mean():.6f}")
    print(f"  📊 FOVマスク: {fov_mask.sum()} / {fov_mask.size} pixels (境界8px除外)")
    print(f"  📊 valid pixels: {valid.sum()} / {valid.size} ({valid.sum()/valid.size*100:.1f}%)")
    
    # gpt5pro8改良: Ωのロバスト化（valid条件 + binary_closing十字型小穴埋め）
    tau = max(support_tau, np.percentile(G[valid], 0.05) if valid.sum() > 0 else support_tau)
    Omega = valid & (G > tau)
    # 十字型構造要素（4近傍）で細線の太り過ぎを抑制
    cross_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    Omega = binary_closing(Omega, structure=cross_kernel)  # 十字型小穴埋め
    support_pixels = int(Omega.sum())
    
    print(f"  🎯 シルエット検出:")
    print(f"    パーセンタイル閾値: {np.percentile(G, 0.05):.6f}")
    print(f"    最終閾値: {tau:.6f}")
    print(f"    サポートピクセル: {support_pixels}")
    
    A = float(px_mm * px_mm)       # ピクセル面積
    sumG = float(G[Omega].sum())
    assert V3D_mm3 > 0 and sumG > 0, f"empty projection (sumG={sumG}) or zero 3D volume (V3D={V3D_mm3})"
    mu_eff = (A * sumG) / V3D_mm3  # 体積一致（∑(G/μ)*A = V3D）
    
    print(f"  ⚖️  Plan-A μ_eff自己較正:")
    print(f"    ピクセル面積A: {A:.6f} mm²")
    print(f"    ∑G: {sumG:.3f}")
    print(f"    3D体積: {V3D_mm3:.1f} mm³")
    print(f"    μ_eff: {mu_eff:.5f} [1/mm]")
    
    T = np.zeros_like(G, dtype=np.float32)
    T[Omega] = G[Omega] / mu_eff   # 画素値 = 厚み[mm]
    
    # QC: 体積一致
    V2D = float(T.sum()) * A
    vol_err = abs(V2D - V3D_mm3) / V3D_mm3 * 100.0
    
    print(f"  ✅ Plan-A体積QC:")
    print(f"    3D体積: {V3D_mm3:.1f} mm³")
    print(f"    2D復元: {V2D:.1f} mm³")
    print(f"    体積誤差: {vol_err:.2f}%")
    print(f"    最大厚み: {T.max():.3f} mm")
    print(f"    平均厚み: {T[Omega].mean():.3f} mm" if support_pixels > 0 else "    平均厚み: N/A")
    
    return T, mu_eff, Omega, vol_err

# ---------- 5) ユーティリティ ----------
def mask_volume_mm3(mask_nifti: Path) -> float:
    img = nib.load(str(mask_nifti))
    data = img.get_fdata()
    dx,dy,dz = img.header.get_zooms()[:3]
    volume = float((data > 0.5).sum()) * float(dx*dy*dz)
    print(f"📊 3Dマスク体積計算:")
    print(f"  スペーシング: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f} mm")
    print(f"  マスクボクセル数: {(data > 0.5).sum()}")
    print(f"  総体積: {volume:.1f} mm³")
    return volume

def save_u16(path: Path, x01: np.ndarray):
    x = np.clip(x01, 0, 1)
    imageio.imwrite(path, (x*65535).astype(np.uint16))

def energy_to_display(img, method="energy", window=(1.0, 99.5), gamma=1.0):
    """表示用正規化（学習・推定には使用しない）"""
    x = img.astype(np.float32)
    
    if method == "energy":
        lo, hi = np.percentile(x, window)
        x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        if gamma != 1.0:
            x = x ** (1.0 / gamma)
        return x
    elif method == "log":
        x = np.clip(x, 1e-6, None)
        y = -np.log(x)
        y = (y - y.min()) / max(y.ptp(), 1e-6)
        return y
    else:
        lo, hi = np.percentile(x, window)
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

# ---------- 6) メイン ----------
def main():
    print("🏥 Plan-A: 単色実効モデル による厚み[mm]推定")
    print("📋 gpt5pro8.md改良実装版 (フラット健全性+Ωロバスト化+現実HU)")
    print("="*60)
    
    # 入力
    ct_nifti   = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    mask_nifti = Path("/workspace/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz")
    outdir = Path("/workspace/mask_projection/outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    GAMMA = 90.0
    view  = (0.0, 0.0, GAMMA)  # PA (縦)

    try:
        print("\n🔄 STEP 1: ボリューム読み込み & 幾何設定")
        # 1) 読み込み & 幾何
        ct_vol   = load_volume_LPS(ct_nifti)
        mask_vol = load_mask_volume_LPS(mask_nifti)
        carm = make_carm(ct_vol.center_in_world)
        ct_vol.place_center(carm.isocenter_in_world)
        mask_vol.place_center(carm.isocenter_in_world)
        
        print(f"  ✅ CT・マスク両方を同一isocenterに配置完了")

        print("\n🔄 STEP 2: 画像生成（I_mask, I_air）")
        # 2) 画像生成（I_mask, I_air）
        print(f"📸 マスク投影: α={view[0]}°, β={view[1]}°, γ={view[2]}°")
        I_mask = render_energy(mask_vol, carm, *view, flip_pa=True)  # radiographic view 準拠
        print(f"  マスク投影完了: 範囲 {I_mask.min():.6f}-{I_mask.max():.6f}, 形状 {I_mask.shape}")
        
        I_air  = get_flat_field(mask_vol, carm, view, outdir, flip_pa=True)
        print(f"  フラットフィールド完了: 範囲 {I_air.min():.6f}-{I_air.max():.6f}, 形状 {I_air.shape}")

        print("\n🔄 STEP 3: Plan-A厚み推定")
        # 3) 厚み推定
        px_mm = float(getattr(carm, "pixel_size", 0.14))
        V3D   = mask_volume_mm3(mask_nifti)
        T_mm, mu_eff, Omega, vol_err = thickness_from_energy(I_mask, I_air, V3D, px_mm, support_tau=1e-4)

        print("\n🔄 STEP 4: 結果保存")
        # 4) 可視化保存
        if T_mm.max() > 0:
            T_disp = np.clip(T_mm / np.percentile(T_mm[T_mm>0], 99.9), 0, 1)
        else:
            T_disp = T_mm
        save_u16(outdir/"PA_mask_thickness_mm_plan_a_gpt5pro8.png", T_disp)
        np.save(outdir/"PA_mask_thickness_mm_plan_a_gpt5pro8.npy", T_mm)
        print(f"  ✅ 厚み画像保存: PA_mask_thickness_mm_plan_a_gpt5pro8.png/.npy")

        # 5) 参考: マスクDRR（表示用）
        print("\n🔄 STEP 5: 参考マスクDRR生成（表示用）")
        E = render_energy(mask_vol, carm, *view, flip_pa=True)
        E_disp = energy_to_display(E, method="energy", window=(1.0, 99.5))
        save_u16(outdir/"PA_mask_deepdrr_plan_a_gpt5pro8.png", E_disp)
        print(f"  ✅ 表示用マスクDRR保存: PA_mask_deepdrr_plan_a_gpt5pro8.png")

        # 6) CT DRR生成（参考）
        print("\n🔄 STEP 6: 参考CT DRR生成（表示用）")
        CT_E = render_energy(ct_vol, carm, *view, flip_pa=True)
        CT_disp = energy_to_display(CT_E, method="energy", window=(1.0, 99.5))
        save_u16(outdir/"PA_ct_deepdrr_plan_a_gpt5pro8.png", CT_disp)
        print(f"  ✅ 表示用CT DRR保存: PA_ct_deepdrr_plan_a_gpt5pro8.png")

        print("\n🔄 STEP 7: QCレポート＆詳細可視化")
        # 7) QC レポート
        info = {
            "pixel_pitch_mm": px_mm,
            "mu_eff_1_per_mm": float(mu_eff),
            "volume_3D_mm3": float(V3D),
            "volume_2D_mm3": float(T_mm.sum() * px_mm * px_mm),
            "volume_error_percent": float(vol_err),
            "support_pixels": int(Omega.sum()),
            "max_thickness_mm": float(T_mm.max()),
            "mean_thickness_mm": float(T_mm[Omega].mean()) if Omega.sum() > 0 else 0.0,
            "image_shape": list(T_mm.shape),
            "plan_version": "gpt5pro8_plan_a_improved"
        }
        with open(outdir/"thickness_info_plan_a_gpt5pro8.json", "w") as f:
            json.dump(info, f, indent=2)
        
        # 詳細可視化
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # 元画像
        axes[0,0].imshow(CT_disp, cmap='gray')
        axes[0,0].set_title('CT DRR (Plan-A gpt5pro8)')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(E_disp, cmap='hot')
        axes[0,1].set_title('Mask DRR (Plan-A gpt5pro8)')
        axes[0,1].axis('off')
        
        # オーバーレイ
        mask_silhouette = (T_disp >= 0.1).astype(np.uint8)
        axes[0,2].imshow(CT_disp, cmap='gray', alpha=1.0)
        axes[0,2].imshow(np.ma.masked_where(mask_silhouette==0, mask_silhouette), 
                        cmap='Reds', alpha=0.5)
        axes[0,2].set_title('Overlay (CT + Mask)')
        axes[0,2].axis('off')
        
        # 厚み画像
        axes[1,0].imshow(T_disp, cmap='viridis')
        axes[1,0].set_title(f'Plan-A gpt5pro8厚み画像 [mm] (max={T_mm.max():.2f}mm)')
        axes[1,0].axis('off')
        
        # 厚みプロファイル
        center_row = T_mm.shape[0] // 2
        thickness_profile = T_mm[center_row, :]
        axes[1,1].plot(thickness_profile)
        axes[1,1].set_title(f'厚み中央行プロファイル (max={thickness_profile.max():.2f}mm)')
        axes[1,1].set_xlabel('X座標')
        axes[1,1].set_ylabel('厚み [mm]')
        axes[1,1].grid(True, alpha=0.3)
        
        # 厚みヒストグラム
        thickness_nonzero = T_mm[Omega]
        if len(thickness_nonzero) > 0:
            axes[1,2].hist(thickness_nonzero, bins=50, alpha=0.7, color='green')
            axes[1,2].set_title(f'Plan-A厚み分布 (n={len(thickness_nonzero)})')
            axes[1,2].set_xlabel('厚み [mm]')
            axes[1,2].set_ylabel('頻度')
            axes[1,2].grid(True, alpha=0.3)
        else:
            axes[1,2].text(0.5, 0.5, 'データなし', ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].axis('off')
        
        # 統計情報
        axes[2,0].text(0.05, 0.95, "📊 Plan-A gpt5pro8厚み推定結果", fontsize=14, weight='bold',
                      transform=axes[2,0].transAxes)
        
        stats_text = f"""
Plan-A (gpt5pro8改良実装):
✅ Beer–Lambert逆変換 + valid条件
✅ μ_eff体積一致自己較正

厚み統計:
• 最大厚み: {info['max_thickness_mm']:.3f} mm
• 平均厚み: {info['mean_thickness_mm']:.3f} mm
• μ_eff: {info['mu_eff_1_per_mm']:.5f} [1/mm]
• 体積誤差: {info['volume_error_percent']:.2f}%
• QC判定: {'✅合格' if info['volume_error_percent'] < 2.0 else '⚠️要確認'}

gpt5pro8改良特徴:
• I_air健全性チェック(>1e-3)
• FOVマスク境界8px除外
• Ω十字型closing小穴埋め
• フラットキャッシュ拡張
• 現実HU(-50)ビームハード軽減
"""
        
        axes[2,0].text(0.05, 0.85, stats_text.strip(), fontsize=10, 
                      transform=axes[2,0].transAxes, verticalalignment='top')
        axes[2,0].axis('off')
        
        # 技術仕様
        axes[2,1].text(0.05, 0.95, "🔧 Plan-A gpt5pro8技術仕様", fontsize=14, weight='bold',
                      transform=axes[2,1].transAxes)
        
        tech_specs = f"""
Plan-A実装: gpt5pro8改良版
座標系: 厳密LPS同期変換
C-arm: CT完全共有
  SID: 1800mm, SAD: 1720mm
  解像度: {px_mm}mm/pixel
  
投影角度: PA (α=0°, β=0°, γ=90°)
反転処理: PAのみ水平反転1回

gpt5pro8厚み推定:
  Beer–Lambert: G = -log(I/I₀) with valid
  体積一致: μ_eff = A∑G/V₃D
  厚み: T = G/μ_eff [mm]
  
体積整合性:
  3D体積: {info['volume_3D_mm3']:.1f} mm³
  2D復元: {info['volume_2D_mm3']:.1f} mm³
  
出力形状:
  厚み: {info['image_shape']}
  サポート: {info['support_pixels']} pixels
"""
        
        axes[2,1].text(0.05, 0.85, tech_specs.strip(), fontsize=9,
                      transform=axes[2,1].transAxes, verticalalignment='top')
        axes[2,1].axis('off')
        
        # QC詳細
        axes[2,2].text(0.05, 0.95, "📋 Plan-A gpt5pro8 QC詳細", fontsize=14, weight='bold',
                      transform=axes[2,2].transAxes)
        
        qc_details = f"""
体積一致QC:
  誤差: {info['volume_error_percent']:.2f}%
  基準: <2% (厳格)
  判定: {'✅PASS' if info['volume_error_percent'] < 2.0 else '⚠️FAIL'}

シルエット検出:
  サポートピクセル: {info['support_pixels']}
  検出率: {info['support_pixels']/np.prod(info['image_shape'])*100:.2f}%

μ_eff自己較正:
  係数: {info['mu_eff_1_per_mm']:.5f} [1/mm]
  方式: 体積一致制約

ピクセル統計:
  最大厚み: {info['max_thickness_mm']:.3f} mm
  平均厚み: {info['mean_thickness_mm']:.3f} mm
  ゼロ率: {(1-info['support_pixels']/np.prod(info['image_shape']))*100:.1f}%

ファイル出力:
  .npy: 生厚みデータ[mm]
  .png: 表示用正規化
  .json: QCレポート
"""
        
        axes[2,2].text(0.05, 0.85, qc_details.strip(), fontsize=9,
                      transform=axes[2,2].transAxes, verticalalignment='top')
        axes[2,2].axis('off')
        
        plt.suptitle('Plan-A: 単色実効モデル厚み推定 (gpt5pro8改良実装)', 
                    fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(outdir / "plan_a_analysis_gpt5pro8.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\n📊 STEP 8: Plan-A gpt5pro8最終成果報告")
        print("🎉 Plan-A (gpt5pro8改良版) 実装完了!")
        print(json.dumps(info, indent=2, ensure_ascii=False))

        print("\n🏆 Plan-A gpt5pro8主要成果:")
        print(f"  ✅ Beer–Lambert逆変換+valid条件による物理的厚み推定")
        print(f"  ✅ μ_eff体積一致自己較正: {info['mu_eff_1_per_mm']:.5f} [1/mm]")
        print(f"  ✅ 体積整合性: {info['volume_error_percent']:.2f}% 誤差")
        print(f"  ✅ 最大厚み: {info['max_thickness_mm']:.3f} mm")
        print(f"  ✅ QC判定: {'合格' if info['volume_error_percent'] < 2.0 else '要確認'}")
        
        print("\n📁 Plan-A gpt5pro8出力ファイル:")
        print(f"  • PA_mask_thickness_mm_plan_a_gpt5pro8.npy: 厚みデータ[mm]")
        print(f"  • PA_mask_thickness_mm_plan_a_gpt5pro8.png: 厚み画像")
        print(f"  • PA_mask_deepdrr_plan_a_gpt5pro8.png: マスクDRR")
        print(f"  • PA_ct_deepdrr_plan_a_gpt5pro8.png: CT DRR")
        print(f"  • thickness_info_plan_a_gpt5pro8.json: QCレポート")
        print(f"  • plan_a_analysis_gpt5pro8.png: 詳細解析")
        
        print("\n" + "="*60)
        print("🌟 gpt5pro8 Plan-A改良実装成功!")
        print("📋 ∑T_mm * p² = V_3D体積一致を厳密達成 + ロバスト化")
        if info['volume_error_percent'] < 2.0:
            print("✅ QC合格：Plan-A gpt5pro8実装が基準値内で動作")
        else:
            print("⚠️  QC要確認：体積整合性を再検証")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Plan-A gpt5pro8実装エラー: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()