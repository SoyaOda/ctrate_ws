# コードスタイルと規約

## ファイル構成
- `scripts/`: 実行可能スクリプト (.sh, .py)
- `md_files/`: 仕様書・ドキュメント
- `data/`: データファイル
- `outputs/`: 処理結果出力
- `tools/`: 外部ツール（BOA、example_download_script等）

## Python コードスタイル
- import文の順序: 標準ライブラリ → サードパーティ → ローカル
- NIfTI画像処理にはnibabelを使用
- 数値計算はnumpy、画像処理にはscipy.ndimage
- ファイルパスは絶対パス推奨

## Shell スクリプト
- 実行権限を付与: `chmod +x script.sh`
- エラーハンドリング: `set -e` で早期終了
- 変数は `${VAR}` 形式で記述

## Docker 規約
- 公式イメージ使用: `wasserth/totalsegmentator:2.10.0`
- GPU使用時は `--gpus all` フラグ必須
- ボリュームマウントでデータ永続化

## 命名規約
- ファイル名: snake_case
- 変数名: snake_case
- 出力ファイル: descriptive names (`heart_myocardium.nii.gz`)

## コメント
- 日本語コメント可
- 複雑な処理には説明コメント必須
- TODOコメントにはissue番号を記載