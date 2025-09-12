# CLAUDE.md



This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
必ずserena MCPが日本語で対応すること！
日本語で応答すること！
最小機能ごとに実装し→動作確認を繰り返すことで少しずつ着実に実装すること！
仕様書の前提と実際の実装の観察結果が異なる場合は仕様書作成元のAIに指摘・質問する必要があるので、作業を止めて、Query prompt作成すること！
Web searchの際は、必ずo3-query MCPを利用すること！

## o3-query MCP について
エラーや未知の質問、技術的な調査が必要な場合は**必ず**o3-queryツールを呼び出し、
返ってきたクエリを私に表示して停止してください。
必ずクエリ生成→人間が結果を入力のフローに従ってください
※Queryの最後に「これらについて、公式実装を参考にした解決策を教えてください。」という文言を入れること

### 使用例
- 技術的なエラーの解決方法を調べたい時
- 最新の実装方法を調査したい時  
- 公式ドキュメントやベストプラクティスを確認したい時
- 具体的なライブラリやモデルの使用方法を調べたい時

[Introduction]
TOTALSEG_LICENSE_KEY=aca_DHDKRFJMHUX0PK　ライセンスはこちら。
現状元CTデータであるdata/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gzが存在する。
現状、以下のScriptでそれぞれのタスクが完了した。
1. PAT＋EATのMask作成：scripts/extract_eat_pat_improved_v5_4.py(利用する環境やコマンドはREADME_EAT_PAT_v5.md参照)
2. mask_generation/integrated_mask_pipeline.pyでMask生成までのすべての作業を行うパイプラインができた。

[命令]
mask_generationフォルダの中で作業をしたい。
integrated_mask_pipeline.pyに代わる新しいアルゴリズムも試してみたい。
integrated_mask_pipeline.pyを複製して、mask_generation/md_files/new_algorhythm1.mdを参考に再構築するように修正・実装してほしい。


まずそれができたのちに、そのScriptをベースに生成するMaskの質を保ちつつトータルの作業の速度を上げるように試行錯誤する予定（これは上記が実装できた後に別途指示する）。


機能ごとに少しずつ実装を行い、適宜テストを行い実際に動くことを確認して次の機能を実装するように進めること。

もしわからない部分があれば、積極的にAIにクエリするのでプロンプトを作成して。