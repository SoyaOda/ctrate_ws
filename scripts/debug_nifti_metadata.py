#!/usr/bin/env python3
"""
NIfTIメタデータ比較デバッグツール
オリジナルとHARTA出力のNIfTIファイルのメタデータを比較
"""
import nibabel as nib
import numpy as np
import sys
import argparse

def analyze_nifti(file_path, label=""):
    """NIfTIファイルのメタデータを分析"""
    print(f"\n{'='*60}")
    print(f"分析対象 [{label}]: {file_path}")
    print('='*60)
    
    try:
        nii = nib.load(file_path)
        data = nii.get_fdata()
        header = nii.header
        affine = nii.affine
        
        print(f"データ形状: {data.shape}")
        print(f"データ型: {data.dtype}")
        print(f"最小値/最大値: {np.min(data):.2f} / {np.max(data):.2f}")
        print(f"非ゼロ要素数: {np.count_nonzero(data)}")
        
        print(f"\nヘッダー情報:")
        print(f"  次元: {header.get_data_shape()}")
        print(f"  ボクセルサイズ: {header.get_zooms()[:3]}")
        print(f"  データ型: {header.get_data_dtype()}")
        print(f"  スライス順序: {header.get_slice_duration()}")
        
        print(f"\nアフィン変換行列:")
        print(affine)
        
        print(f"\n向き情報:")
        print(f"  q-form code: {header['qform_code']}")
        print(f"  s-form code: {header['sform_code']}")
        
        # RAS+ 座標系の確認
        print(f"\n座標系:")
        print(f"  原点: {affine[:3, 3]}")
        print(f"  X軸方向: {affine[:3, 0]}")
        print(f"  Y軸方向: {affine[:3, 1]}")
        print(f"  Z軸方向: {affine[:3, 2]}")
        
        return {
            'shape': data.shape,
            'affine': affine,
            'spacing': header.get_zooms()[:3],
            'origin': affine[:3, 3],
            'nonzero': np.count_nonzero(data)
        }
        
    except Exception as e:
        print(f"エラー: {str(e)}")
        return None

def compare_nifti_files(original_path, segmentation_path):
    """2つのNIfTIファイルを比較"""
    print("\n" + "="*80)
    print("NIfTIファイル比較分析")
    print("="*80)
    
    orig_info = analyze_nifti(original_path, "オリジナルCT")
    seg_info = analyze_nifti(segmentation_path, "HARTAセグメンテーション")
    
    if orig_info and seg_info:
        print("\n" + "="*80)
        print("比較結果:")
        print("="*80)
        
        # 形状比較
        if orig_info['shape'] != seg_info['shape']:
            print(f"⚠️ 形状不一致: {orig_info['shape']} vs {seg_info['shape']}")
        else:
            print(f"✓ 形状一致: {orig_info['shape']}")
        
        # スペーシング比較
        orig_spacing = np.array(orig_info['spacing'])
        seg_spacing = np.array(seg_info['spacing'])
        spacing_diff = np.abs(orig_spacing - seg_spacing)
        if np.max(spacing_diff) > 0.01:
            print(f"⚠️ スペーシング不一致:")
            print(f"  オリジナル: {orig_spacing}")
            print(f"  セグメント: {seg_spacing}")
            print(f"  差: {spacing_diff}")
        else:
            print(f"✓ スペーシング一致: {orig_spacing}")
        
        # アフィン行列比較
        affine_diff = np.abs(orig_info['affine'] - seg_info['affine'])
        if np.max(affine_diff) > 0.01:
            print(f"⚠️ アフィン変換行列不一致:")
            print(f"  最大差: {np.max(affine_diff):.4f}")
            print(f"  差分行列:")
            print(affine_diff)
        else:
            print(f"✓ アフィン変換行列一致")
        
        # 原点比較
        origin_diff = np.abs(orig_info['origin'] - seg_info['origin'])
        if np.max(origin_diff) > 1.0:
            print(f"⚠️ 原点位置不一致:")
            print(f"  オリジナル: {orig_info['origin']}")
            print(f"  セグメント: {seg_info['origin']}")
            print(f"  差: {origin_diff}")
        else:
            print(f"✓ 原点位置一致")

def main():
    parser = argparse.ArgumentParser(description="NIfTIメタデータ比較ツール")
    parser.add_argument("--original", required=True, help="オリジナルCT NIfTIファイル")
    parser.add_argument("--segmentation", required=True, help="セグメンテーションNIfTIファイル")
    args = parser.parse_args()
    
    compare_nifti_files(args.original, args.segmentation)

if __name__ == "__main__":
    main()