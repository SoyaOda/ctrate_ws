#!/usr/bin/env python3
"""
統合パイプラインのドライランテスト
実際のDockerコマンドは実行せず、フローとログ出力を確認
"""

import sys
import tempfile
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append('/home/soya/ctrate_ws')

def test_pipeline_structure():
    """パイプラインの構造テスト"""
    print("=" * 60)
    print("統合パイプライン構造テスト")
    print("=" * 60)
    
    # モジュールのインポートテスト
    try:
        from mask_generation.integrated_mask_pipeline import IntegratedMaskPipeline
        print("✓ IntegratedMaskPipelineクラスのインポート成功")
    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False
    
    # 一時ディレクトリで初期化テスト
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # 入力CTパス（実際のファイル）
            input_ct = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
            
            # パイプラインインスタンスの作成
            pipeline = IntegratedMaskPipeline(
                input_ct_path=input_ct,
                output_dir=tmpdir,
                verbose=True
            )
            print("✓ パイプラインインスタンスの作成成功")
            
            # 各メソッドの存在確認
            methods = [
                'step1_heartchambers_segmentation',
                'step2_total_segmentation', 
                'step3_tissue_segmentation',
                'step4_integrate_segmentations',
                'step5_extract_eat_pat'
            ]
            
            for method_name in methods:
                if hasattr(pipeline, method_name):
                    print(f"✓ メソッド {method_name} が存在")
                else:
                    print(f"✗ メソッド {method_name} が見つかりません")
                    return False
            
            # 時間測定機能のテスト
            print("\n時間測定機能のテスト:")
            pipeline.timing_results['test_step'] = 1.23
            pipeline.save_timing_results()
            
            timing_file = Path(tmpdir) / "timing_results.json"
            if timing_file.exists():
                print("✓ 時間測定結果の保存成功")
            else:
                print("✗ 時間測定結果の保存失敗")
                return False
            
        except Exception as e:
            print(f"✗ エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("全てのテストが成功しました！")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_pipeline_structure()
    sys.exit(0 if success else 1)