#!/usr/bin/env python3
"""
HARTA出力ディレクトリの内容を確認
"""
import numpy as np
from PIL import Image
from pathlib import Path

def analyze_png(png_path):
    """PNG画像の内容を分析"""
    img = Image.open(png_path)
    arr = np.array(img)
    
    if len(arr.shape) == 3 and arr.shape[2] >= 3:
        # RGBチャンネルの分析
        rgb = arr[:, :, :3]
        # 各チャンネルの非ゼロピクセル数
        r_nonzero = np.count_nonzero(rgb[:, :, 0])
        g_nonzero = np.count_nonzero(rgb[:, :, 1])
        b_nonzero = np.count_nonzero(rgb[:, :, 2])
        
        # 白いピクセル（255,255,255）の数
        white_pixels = np.sum((rgb[:, :, 0] == 255) & 
                              (rgb[:, :, 1] == 255) & 
                              (rgb[:, :, 2] == 255))
        
        # 赤いピクセル（255,0,0）の数
        red_pixels = np.sum((rgb[:, :, 0] == 255) & 
                           (rgb[:, :, 1] == 0) & 
                           (rgb[:, :, 2] == 0))
        
        return {
            'shape': arr.shape,
            'r_nonzero': r_nonzero,
            'g_nonzero': g_nonzero,
            'b_nonzero': b_nonzero,
            'white_pixels': white_pixels,
            'red_pixels': red_pixels,
            'unique_colors': len(np.unique(rgb.reshape(-1, 3), axis=0))
        }
    else:
        return {
            'shape': arr.shape,
            'nonzero': np.count_nonzero(arr),
            'unique_values': len(np.unique(arr))
        }

def main():
    base_dir = Path("/home/soya/ctrate_ws/outputs/test_harta_fixed/harta_output/test_valid_2_a_1_fixed")
    
    # テストするスライス番号
    test_slices = [50, 100]
    
    print("="*80)
    print("HARTA各ディレクトリの内容分析")
    print("="*80)
    
    for slice_num in test_slices:
        print(f"\n■ スライス {slice_num} の分析:")
        print("-"*40)
        
        # 1. slices/ - 元のCT画像
        slice_png = base_dir / "slices" / f"CTRATE001_{slice_num}.png"
        if slice_png.exists():
            info = analyze_png(slice_png)
            print(f"slices/CTRATE001_{slice_num}.png:")
            print(f"  形状: {info['shape']}")
            print(f"  説明: 元のCTスライス画像（グレースケール変換済み）")
            print(f"  非ゼロRGB: R={info['r_nonzero']}, G={info['g_nonzero']}, B={info['b_nonzero']}")
        
        # 2. contours/ - 心臓領域のコンター
        contour_png = base_dir / "contours" / f"CTRATE001_{slice_num}_c.png"
        if contour_png.exists():
            info = analyze_png(contour_png)
            print(f"\ncontours/CTRATE001_{slice_num}_c.png:")
            print(f"  形状: {info['shape']}")
            print(f"  説明: 心臓領域のコンター（ROI）")
            print(f"  白ピクセル(255,255,255): {info['white_pixels']} ({100*info['white_pixels']/(512*512):.1f}%)")
            print(f"  → 心臓領域の境界/内部を示すマスク")
        
        # 3. fat/ - EATマスク
        fat_png = base_dir / "fat" / f"CTRATE001_{slice_num}_fat.png"
        if fat_png.exists():
            info = analyze_png(fat_png)
            print(f"\nfat/CTRATE001_{slice_num}_fat.png:")
            print(f"  形状: {info['shape']}")
            print(f"  説明: 脂肪（EAT）マスク")
            print(f"  白ピクセル(255,255,255): {info['white_pixels']} ({100*info['white_pixels']/(512*512):.1f}%)")
            print(f"  → これがEATセグメンテーション結果！")
        
        # 4. combined/ - オーバーレイ画像
        combined_png = base_dir / "combined" / f"CTRATE001_{slice_num}_combined.png"
        if combined_png.exists():
            info = analyze_png(combined_png)
            print(f"\ncombined/CTRATE001_{slice_num}_combined.png:")
            print(f"  形状: {info['shape']}")
            print(f"  説明: 元画像＋EATマスクのオーバーレイ")
            print(f"  赤ピクセル(255,0,0): {info['red_pixels']} ({100*info['red_pixels']/(512*512):.1f}%)")
            print(f"  色の種類: {info['unique_colors']}")
            print(f"  → 視覚化用（赤い部分がEAT）")
    
    print("\n" + "="*80)
    print("ディレクトリの役割まとめ:")
    print("="*80)
    print("• slices/   : 元のCTスライス画像")
    print("• contours/ : 心臓領域のコンター（ROI）")
    print("• fat/      : 【重要】EATセグメンテーションマスク（白=EAT、黒=背景）")
    print("• combined/ : 視覚化用オーバーレイ（赤=EAT）")
    print("\n→ EATそのものは fat/ ディレクトリに白色(255,255,255)として保存されています")

if __name__ == "__main__":
    main()