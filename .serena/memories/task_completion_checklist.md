# タスク完了時のチェックリスト

## コード変更後の必須確認項目

### 1. 動作テスト
```bash
# 変更したスクリプトのテスト実行
./scripts/quick_test_totalseg.sh

# Docker環境の確認
docker ps
nvidia-smi
```

### 2. ファイル出力確認
```bash
# 出力ディレクトリの確認
ls -la outputs/
ls -la outputs/ts_test/

# NIfTIファイルの妥当性確認（Python）
python -c "
import nibabel as nib
import numpy as np
img = nib.load('path/to/output.nii.gz')
data = img.get_fdata()
print(f'Shape: {data.shape}')
print(f'Range: [{data.min():.0f}, {data.max():.0f}]')
"
```

### 3. エラーログ確認
```bash
# Dockerログ確認
docker compose logs

# システムログ確認（必要時）
journalctl --since "1 hour ago"
```

### 4. リソース使用量確認
```bash
# GPU使用量
nvidia-smi

# ディスク容量
df -h

# メモリ使用量
free -h
```

### 5. 設定ファイル確認
- `.env` ファイルの設定値
- TotalSegmentatorライセンス設定
- Dockerボリュームマウント設定

### 6. ドキュメント更新
- README.md の手順更新（必要時）
- エラーログの記録
- トラブルシューティング情報追加

## コミット前チェック
- 機密情報（ライセンスキー）の除去
- 不要なデバッグ出力の削除
- コメント・ドキュメントの更新