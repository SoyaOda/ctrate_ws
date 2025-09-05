# CLAUDE.md



This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
必ずserena MCPが日本語で対応すること！
日本語で応答すること！
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
現状元CTデータであるdata/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gzとそのMaskでーたであるoutputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gzが存在する。
deepdrr/README.mdにあるように、deepdrr/test_deepdrr_final_pa_fixed.pyでCTのNiftiをPAのDRRにすることはできた。次はMask dataをDRRに完全位置対応するように投射して2D画像を作成したい。

[命令]
mask_projection/md_files/gpt5pro1.mdとmask_projection/md_files/gpt5pro2.mdに沿って実装を進めたい。このプロジェクトはmask_projectionフォルダの中で進めたい。

機能ごとに少しずつ実装を行い、適宜テストを行い実際に動くことを確認して次の機能を実装するように進めること。

もしわからない部分があれば、積極的にAIにクエリするのでプロンプトを作成して。