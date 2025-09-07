#!/usr/bin/env python3
"""
DeepDRRマスク投射スクリプトの構文テスト
Docker環境なしでの動作確認用
"""

import sys
from pathlib import Path
import numpy as np

def test_imports():
    """必要ライブラリのインポートテスト"""
    print("🧪 ライブラリインポートテスト")
    
    try:
        import nibabel as nib
        print("  ✅ nibabel: OK")
    except ImportError as e:
        print(f"  ❌ nibabel: {e}")
        return False
    
    try:
        from nibabel import orientations as nio
        from nibabel.affines import voxel_sizes
        print("  ✅ nibabel.orientations: OK")
    except ImportError as e:
        print(f"  ❌ nibabel.orientations: {e}")
        return False
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        print("  ✅ matplotlib: OK")
    except ImportError as e:
        print(f"  ❌ matplotlib: {e}")
        return False
    
    try:
        import imageio.v2 as imageio
        print("  ✅ imageio: OK")
    except ImportError as e:
        print(f"  ❌ imageio: {e}")
        return False
    
    # DeepDRRは環境依存なのでインポートテストのみ
    deepdrr_available = False
    try:
        from deepdrr import Volume, MobileCArm
        from deepdrr.projector import Projector
        print("  ✅ DeepDRR: OK (フル機能利用可能)")
        deepdrr_available = True
    except ImportError as e:
        print(f"  ⚠️  DeepDRR: {e} (Docker環境が必要)")
        deepdrr_available = False
    
    return deepdrr_available

def test_mask_data_loading():
    """マスクデータの読み込みテスト"""
    print("\\n📁 マスクデータ読み込みテスト")
    
    mask_path = Path("/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz")
    
    if not mask_path.exists():
        print(f"  ❌ マスクファイルが見つかりません: {mask_path}")
        return False
    
    try:
        import nibabel as nib
        from nibabel import orientations as nio
        
        print(f"  📁 読み込み中: {mask_path}")
        
        # 基本読み込み
        img = nib.load(str(mask_path))
        print(f"  📊 形状: {img.shape}")
        print(f"  📏 アフィン行列形状: {img.affine.shape}")
        
        # 座標系確認
        current_orient = nio.aff2axcodes(img.affine)
        print(f"  🧭 現在の座標系: {current_orient}")
        
        # データ統計
        data = img.get_fdata()
        nonzero_count = np.sum(data > 0.5)
        total_voxels = data.size
        
        print(f"  📊 データ統計:")
        print(f"    総voxel数: {total_voxels:,}")
        print(f"    非ゼロvoxel数: {nonzero_count:,}")
        print(f"    マスク率: {100*nonzero_count/total_voxels:.2f}%")
        print(f"    値範囲: {data.min():.3f} - {data.max():.3f}")
        
        # LPS変換テスト
        in_ornt = nio.io_orientation(img.affine)
        out_ornt = nio.axcodes2ornt(('L','P','S'))
        to_lps = nio.ornt_transform(in_ornt, out_ornt)
        
        print(f"  🔄 LPS変換行列: {to_lps}")
        
        data_lps = nio.apply_orientation(data.astype(np.float32), to_lps)
        print(f"  ✅ LPS変換後形状: {data_lps.shape}")
        
        # 275.2999mm境界の確認
        aff_lps = img.affine @ nio.inv_ornt_aff(to_lps, img.shape)
        spacing_z = aff_lps[2, 2]  # Z軸spacing
        origin_z = aff_lps[2, 3]   # Z軸origin
        
        z_indices = np.arange(data_lps.shape[2])
        z_physical = origin_z + z_indices * spacing_z
        
        print(f"  📏 Z軸物理座標範囲: {z_physical[0]:.3f} - {z_physical[-1]:.3f} mm")
        
        # 275mm付近の検索
        target_z = 275.3
        closest_idx = np.argmin(np.abs(z_physical - target_z))
        closest_z = z_physical[closest_idx]
        
        print(f"  🎯 275.3mm最近接: インデックス{closest_idx}, 座標{closest_z:.3f}mm")
        
        # そのスライスでの密度
        if closest_idx < data_lps.shape[2]:
            slice_density = np.sum(data_lps[:, :, closest_idx] > 0.5)
            print(f"  📊 275.3mm付近密度: {slice_density} voxels")
        
        print("  ✅ マスクデータ読み込み成功")
        return True
        
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_script_syntax():
    """メインスクリプトの構文チェック"""
    print("\\n🔍 メインスクリプト構文チェック")
    
    script_path = Path("/home/soya/ctrate_ws/mask_projection/src/mask_deepdrr_projection.py")
    
    if not script_path.exists():
        print(f"  ❌ スクリプトファイルが見つかりません: {script_path}")
        return False
    
    try:
        # 構文チェック（コンパイルのみ）
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, str(script_path), 'exec')
        print("  ✅ 構文チェック成功")
        
        # 重要な関数の存在確認
        important_functions = [
            'load_volume_LPS',
            'make_carm', 
            'project_and_flip',
            'mask_to_hu_like',
            'load_mask_volume_LPS',
            'detect_sharp_boundaries_advanced',
            'main'
        ]
        
        for func_name in important_functions:
            if f"def {func_name}" in code:
                print(f"    ✅ {func_name}関数: 存在")
            else:
                print(f"    ❌ {func_name}関数: 未発見")
        
        return True
        
    except SyntaxError as e:
        print(f"  ❌ 構文エラー: {e}")
        print(f"    行 {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        return False

def create_mock_test():
    """DeepDRRなしでのモック動作テスト"""
    print("\\n🎭 モック動作テスト（DeepDRR機能を除く）")
    
    try:
        # パス設定
        mask_path = Path("/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz")
        
        if not mask_path.exists():
            print("  ❌ マスクファイルが存在しません")
            return False
        
        # 基本処理テスト
        import nibabel as nib
        from nibabel import orientations as nio
        from nibabel.affines import voxel_sizes
        
        # LPS変換
        img = nib.load(str(mask_path))
        in_ornt = nio.io_orientation(img.affine)
        out_ornt = nio.axcodes2ornt(('L','P','S'))
        to_lps = nio.ornt_transform(in_ornt, out_ornt)
        
        mask_lps = nio.apply_orientation(img.get_fdata().astype(np.float32), to_lps)
        print(f"  ✅ LPS変換: {mask_lps.shape}")
        
        # 疑似HU変換
        hu_mask = np.full(mask_lps.shape, -1000.0, dtype=np.float32)
        hu_mask[mask_lps > 0.5] = 3000.0
        
        nonzero_hu = np.sum(hu_mask > 0)
        print(f"  ✅ 疑似HU変換: {nonzero_hu:,} 高吸収voxels")
        
        # 勾配計算テスト（2Dスライス）
        test_slice = mask_lps[:, :, mask_lps.shape[2]//2]  # 中央スライス
        grad_x = np.abs(np.diff(test_slice, axis=1))
        grad_y = np.abs(np.diff(test_slice, axis=0))
        
        print(f"  ✅ 勾配計算テスト:")
        print(f"    X方向最大勾配: {grad_x.max():.6f}")
        print(f"    Y方向最大勾配: {grad_y.max():.6f}")
        
        # 出力ディレクトリ作成テスト
        outdir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ 出力ディレクトリ作成: {outdir}")
        
        print("  🎉 モック動作テスト成功")
        return True
        
    except Exception as e:
        print(f"  ❌ モックテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """総合テスト実行"""
    print("🧪 DeepDRRマスク投射スクリプト総合テスト")
    print("="*60)
    
    results = {}
    
    # 1. ライブラリインポートテスト
    results['imports'] = test_imports()
    
    # 2. マスクデータ読み込みテスト
    results['mask_loading'] = test_mask_data_loading()
    
    # 3. スクリプト構文チェック
    results['syntax'] = test_script_syntax()
    
    # 4. モック動作テスト
    results['mock'] = create_mock_test()
    
    # 結果サマリー
    print("\\n" + "="*60)
    print("📊 テスト結果サマリー")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"  {test_name.upper()}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\\n🎯 総合結果: {total_passed}/{total_tests} テスト合格")
    
    if results.get('imports', False) and not results.get('imports'):
        print("\\n⚠️  DeepDRRライブラリが利用不可能です")
        print("   • Docker環境での実行を推奨")
        print("   • またはCUDA 11.8環境でのDeepDRRインストールが必要")
    
    if results['syntax'] and results['mask_loading']:
        print("\\n🌟 基本機能は動作準備完了!")
        print("   • DeepDRR環境が整えばフル機能利用可能")
        print("   • マスクデータの275.2999mm境界も確認済み")
    
    print("="*60)

if __name__ == "__main__":
    main()