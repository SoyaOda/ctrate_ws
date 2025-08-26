#!/usr/bin/env python3
"""
EAT抽出結果の可視化スクリプト
心臓セグメンテーション、6mm殻、最終EATをPNGとして出力
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from PIL import Image
import matplotlib.cm as cm

def load_nifti_data(nifti_path):
    """NIfTIファイルをnumpyで読み込み（SimpleITK/nibabel不要版）"""
    import gzip
    import struct
    
    with gzip.open(nifti_path, 'rb') as f:
        # NIfTIヘッダーをスキップ（352バイト）
        f.read(352)
        # データを読み込み
        data = f.read()
    
    # float32として解釈（一般的なNIfTI形式）
    # 実際のデータ型はヘッダーから取得すべきだが、簡略化
    return None  # この実装は複雑なため、Dockerで実行

def create_visualization_docker(ct_nifti, totalseg_dir, eat_nifti, output_dir, 
                              shell_thickness=6.0, num_slices=10):
    """
    Dockerコンテナ内で可視化を実行
    """
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')  # バックエンドを設定
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    print(f"[INFO] Loading data...")
    
    # CTデータ読み込み
    ct_img = nib.load(ct_nifti)
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]
    
    # 心臓構造を読み込み
    heart_parts = []
    components = [
        "heart_myocardium",
        "heart_ventricle_left", 
        "heart_ventricle_right",
        "heart_atrium_left",
        "heart_atrium_right"
    ]
    
    for comp in components:
        comp_path = Path(totalseg_dir) / f"{comp}.nii.gz"
        if comp_path.exists():
            print(f"[INFO] Loading {comp}...")
            mask = nib.load(str(comp_path)).get_fdata() > 0
            heart_parts.append(mask)
    
    # 心臓全体のマスク
    heart = np.logical_or.reduce(heart_parts) if heart_parts else np.zeros_like(ct_data, dtype=bool)
    
    # 心筋マスク
    myo_path = Path(totalseg_dir) / "heart_myocardium.nii.gz"
    if myo_path.exists():
        myo = nib.load(str(myo_path)).get_fdata() > 0
    else:
        myo = heart
    
    # 6mm殻を生成
    print(f"[INFO] Creating {shell_thickness}mm shell...")
    outside = ~myo
    dist_mm = distance_transform_edt(outside, sampling=spacing)
    shell_eat = (dist_mm > 0) & (dist_mm <= shell_thickness)
    struct = generate_binary_structure(3, 2)
    shell_eat &= binary_dilation(heart, structure=struct, iterations=2)
    
    # EATマスク読み込み
    if eat_nifti and Path(eat_nifti).exists():
        eat_img = nib.load(eat_nifti)
        eat_data = eat_img.get_fdata() > 0
    else:
        eat_data = np.zeros_like(ct_data, dtype=bool)
    
    # 出力ディレクトリ作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリ作成
    (output_dir / "ct_slices").mkdir(exist_ok=True)
    (output_dir / "heart").mkdir(exist_ok=True)
    (output_dir / "shell").mkdir(exist_ok=True)
    (output_dir / "eat").mkdir(exist_ok=True)
    (output_dir / "combined").mkdir(exist_ok=True)
    
    # スライス選択（等間隔）
    z_size = ct_data.shape[2]
    slice_indices = np.linspace(0, z_size-1, num_slices, dtype=int)
    
    print(f"[INFO] Creating visualizations for {num_slices} slices...")
    
    for idx, z in enumerate(slice_indices):
        print(f"  Processing slice {z}/{z_size}...")
        
        # CTスライス
        ct_slice = ct_data[:, :, z]
        
        # Window/Level調整（胸部CT用）
        window_center = 40
        window_width = 400
        ct_windowed = np.clip(
            (ct_slice - (window_center - window_width/2)) / window_width * 255,
            0, 255
        ).astype(np.uint8)
        
        # 各マスクのスライス
        heart_slice = heart[:, :, z].astype(np.uint8)
        shell_slice = shell_eat[:, :, z].astype(np.uint8)
        eat_slice = eat_data[:, :, z].astype(np.uint8)
        
        # 1. CT画像
        plt.figure(figsize=(8, 8))
        plt.imshow(ct_windowed, cmap='gray')
        plt.axis('off')
        plt.title(f'CT Slice {z}')
        plt.savefig(output_dir / "ct_slices" / f"slice_{z:03d}_ct.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 2. 心臓セグメンテーション
        plt.figure(figsize=(8, 8))
        plt.imshow(ct_windowed, cmap='gray', alpha=0.7)
        plt.imshow(heart_slice, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f'Heart Segmentation - Slice {z}')
        plt.savefig(output_dir / "heart" / f"slice_{z:03d}_heart.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 3. 6mm殻
        plt.figure(figsize=(8, 8))
        plt.imshow(ct_windowed, cmap='gray', alpha=0.7)
        plt.imshow(shell_slice, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f'{shell_thickness}mm Shell - Slice {z}')
        plt.savefig(output_dir / "shell" / f"slice_{z:03d}_shell.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 4. EAT
        plt.figure(figsize=(8, 8))
        plt.imshow(ct_windowed, cmap='gray', alpha=0.7)
        plt.imshow(eat_slice, cmap='Greens', alpha=0.7, vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f'EAT - Slice {z}')
        plt.savefig(output_dir / "eat" / f"slice_{z:03d}_eat.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 5. 統合画像
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # CT
        axes[0].imshow(ct_windowed, cmap='gray')
        axes[0].set_title('CT')
        axes[0].axis('off')
        
        # 心臓
        axes[1].imshow(ct_windowed, cmap='gray', alpha=0.5)
        axes[1].imshow(heart_slice, cmap='Reds', alpha=0.5)
        axes[1].set_title('Heart')
        axes[1].axis('off')
        
        # 殻
        axes[2].imshow(ct_windowed, cmap='gray', alpha=0.5)
        axes[2].imshow(shell_slice, cmap='Blues', alpha=0.5)
        axes[2].set_title(f'{shell_thickness}mm Shell')
        axes[2].axis('off')
        
        # EAT
        axes[3].imshow(ct_windowed, cmap='gray', alpha=0.5)
        axes[3].imshow(eat_slice, cmap='Greens', alpha=0.7)
        axes[3].set_title('EAT')
        axes[3].axis('off')
        
        plt.suptitle(f'Slice {z}')
        plt.tight_layout()
        plt.savefig(output_dir / "combined" / f"slice_{z:03d}_combined.png", 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        # オーバーレイ画像（全要素を重ねた画像）
        overlay_img = np.stack([ct_windowed, ct_windowed, ct_windowed], axis=-1)
        
        # 心臓を赤で追加
        overlay_img[:, :, 0] = np.where(heart_slice > 0, 
                                        np.minimum(overlay_img[:, :, 0] + 100, 255),
                                        overlay_img[:, :, 0])
        
        # 殻を青で追加
        overlay_img[:, :, 2] = np.where(shell_slice > 0,
                                        np.minimum(overlay_img[:, :, 2] + 100, 255),
                                        overlay_img[:, :, 2])
        
        # EATを緑で追加
        overlay_img[:, :, 1] = np.where(eat_slice > 0,
                                        np.minimum(overlay_img[:, :, 1] + 150, 255),
                                        overlay_img[:, :, 1])
        
        # オーバーレイ保存
        Image.fromarray(overlay_img.astype(np.uint8)).save(
            output_dir / "combined" / f"slice_{z:03d}_overlay.png"
        )
    
    # 統計情報を保存
    stats = {
        "heart_voxels": int(np.sum(heart)),
        "shell_voxels": int(np.sum(shell_eat)),
        "eat_voxels": int(np.sum(eat_data)),
        "shell_thickness_mm": shell_thickness,
        "num_slices_visualized": num_slices,
        "slice_indices": slice_indices.tolist()
    }
    
    stats_path = output_dir / "visualization_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"[INFO] Visualizations saved to {output_dir}")
    print(f"[STATS] Heart: {stats['heart_voxels']} voxels")
    print(f"[STATS] Shell: {stats['shell_voxels']} voxels")
    print(f"[STATS] EAT: {stats['eat_voxels']} voxels")
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Visualize EAT extraction results"
    )
    parser.add_argument("--ct_nifti", required=True,
                       help="Path to CT NIfTI file")
    parser.add_argument("--totalseg_dir", required=True,
                       help="TotalSegmentator output directory")
    parser.add_argument("--eat_nifti", required=True,
                       help="Path to final EAT NIfTI file")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for visualizations")
    parser.add_argument("--shell_thickness", type=float, default=6.0,
                       help="Shell thickness in mm (default: 6.0)")
    parser.add_argument("--num_slices", type=int, default=20,
                       help="Number of slices to visualize (default: 20)")
    
    args = parser.parse_args()
    
    # Dockerで実行する必要がある
    print("[INFO] This script needs to be run inside Docker container with nibabel")
    print("[INFO] Use visualize_eat_docker.sh instead")
    
if __name__ == "__main__":
    main()