#!/usr/bin/env python3
"""
DeepDRR CUDA 11.8環境用テストスクリプト
md_files/answer1.mdの解決策に基づく実装
"""

from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

def test_deepdrr_import():
    """DeepDRRのインポートテスト"""
    try:
        # PyCUDA初期化を安定化するため最初にimport
        from deepdrr.projector import Projector
        from deepdrr import Volume, MobileCArm
        print("✅ DeepDRRのインポートに成功しました")
        return True
    except ImportError as e:
        print(f"❌ DeepDRRのインポートに失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ DeepDRRの初期化エラー: {e}")
        return False

def test_cuda_environment():
    """CUDA環境のテスト"""
    try:
        import pycuda.driver as drv
        drv.init()
        device = drv.Device(0)
        print(f"✅ GPU検出: {device.name()}")
        
        # CUDA context作成テスト
        ctx = device.make_context()
        ctx.pop()
        print("✅ CUDA contextの作成に成功")
        return True
    except Exception as e:
        print(f"❌ CUDA環境エラー: {e}")
        return False

def to_uint16_xray(image: np.ndarray, method: str = "log") -> np.ndarray:
    """
    DeepDRRの出力をX線表示らしく（骨=白）する16bit PNG用画像へ
    """
    img = image.astype(np.float32)
    img /= (img.max() + 1e-6)
    
    if method == "log":
        img = -np.log(img + 1e-6)  # Beer-Lambert風の線積分表示
    elif method == "invert":
        img = 1.0 - img
    
    img -= img.min()
    img /= (img.max() + 1e-8)
    return (img * 65535.0).astype(np.uint16)

def generate_drr_minimal():
    """
    DeepDRRを使った最小DRR生成テスト
    md_files/answer1.mdの最小実行例に基づく
    """
    print("\n" + "="*60)
    print("DeepDRR最小DRR生成テスト")
    print("="*60)
    
    # ファイルパス
    nifti_path = Path("/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    output_dir = Path("/workspace/deepdrr/outputs")
    
    if not nifti_path.exists():
        print(f"❌ CTファイルが見つかりません: {nifti_path}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # DeepDRRコンポーネントのインポート
        from deepdrr import Volume, MobileCArm
        from deepdrr.projector import Projector
        
        print(f"📁 入力NIfTI: {nifti_path}")
        print("🔄 CTボリュームを読み込み中...")
        
        # CTボリューム読み込み
        ct = Volume.from_nifti(str(nifti_path))
        print(f"📊 CTボリューム形状: {ct.data.shape}")
        print(f"📏 CTボリューム間隔: {ct.spacing}")
        
        # C-armジオメトリ
        carm = MobileCArm()
        
        print("🎥 DRR投影を実行中...")
        
        with Projector(ct, carm=carm) as projector:
            # 患者体位の調整（answer1.mdの推奨設定）
            ct.orient_patient(head_first=True, supine=True)
            ct.place_center(carm.isocenter_in_world)
            
            # PA（正面）投影
            print("  📸 PA view（正面）を生成中...")
            carm.move_to(alpha=0, beta=0, degrees=True)
            img_pa = projector()
            
            # LAT（側面）投影
            print("  📸 LAT view（側面）を生成中...")
            carm.move_to(alpha=90, beta=0, degrees=True)  
            img_lat = projector()
            
            print(f"📐 DRR画像サイズ: {img_pa.shape}")
            print(f"📊 PA view値範囲: {img_pa.min():.4f} - {img_pa.max():.4f}")
            print(f"📊 LAT view値範囲: {img_lat.min():.4f} - {img_lat.max():.4f}")
        
        # 16bit PNG形式で保存
        print("💾 画像を保存中...")
        pa_16bit = to_uint16_xray(img_pa, method="log")
        lat_16bit = to_uint16_xray(img_lat, method="log")
        
        pa_path = output_dir / "deepdrr_PA.png"
        lat_path = output_dir / "deepdrr_LAT.png"
        
        imageio.imwrite(pa_path, pa_16bit)
        imageio.imwrite(lat_path, lat_16bit)
        
        print(f"✅ PA view保存: {pa_path}")
        print(f"✅ LAT view保存: {lat_path}")
        
        # 可視化生成
        print("🎨 可視化を生成中...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # PA view
        axes[0].imshow(pa_16bit, cmap='gray')
        axes[0].set_title('PA View (Frontal)\nDeepDRR Generated')
        axes[0].axis('off')
        
        # LAT view  
        axes[1].imshow(lat_16bit, cmap='gray')
        axes[1].set_title('LAT View (Lateral)\nDeepDRR Generated')
        axes[1].axis('off')
        
        plt.suptitle('DeepDRR X-ray Generation Test', fontsize=14)
        plt.tight_layout()
        
        viz_path = output_dir / "deepdrr_comparison.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"🖼️  可視化画像保存: {viz_path}")
        
        print("\n✅ DeepDRR DRR生成テスト完了!")
        return True
        
    except Exception as e:
        print(f"❌ DRR生成エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    print("🚀 DeepDRR CUDA 11.8環境テスト開始")
    print("📋 answer1.mdの解決策に基づく実装\n")
    
    # ステップ1: DeepDRRインポートテスト
    if not test_deepdrr_import():
        print("\n❌ DeepDRRのインポートに失敗。CUDA 11.8環境が必要です。")
        print("📖 解決策: answer1.mdのconda環境またはDocker環境を使用してください")
        return False
    
    # ステップ2: CUDA環境テスト
    if not test_cuda_environment():
        print("\n❌ CUDA環境の初期化に失敗。")
        return False
    
    # ステップ3: DRR生成テスト
    if not generate_drr_minimal():
        print("\n❌ DRR生成テストに失敗。")
        return False
    
    print("\n" + "="*60)
    print("🎉 全てのテストが正常に完了しました！")
    print("📁 出力ファイル: /home/soya/ctrate_ws/deepdrr/outputs/")
    print("="*60)
    return True

if __name__ == "__main__":
    main()