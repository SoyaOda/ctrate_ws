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
※Queryの最後に「これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。」という文言を入れること

### 使用例
- 技術的なエラーの解決方法を調べたい時
- 最新の実装方法を調査したい時  
- 公式ドキュメントやベストプラクティスを確認したい時
- 具体的なライブラリやモデルの使用方法を調べたい時

[Introduction]
現状scripts/batch_process_all.shを用いて元CT画像のNIFTIファイルを入力してEATを得るScriptを実装した。

[命令]
現状README_EAT_PAT_EXTRACTION.mdのようにアルゴリズムとVisualization Scriptを実装できた。
そのあと、scripts/extract_eat_pat_improved_v3_backup.pyのようにアルゴリズムを実装できた。
現状のアルゴリズムへのフィードバックとして、md_files/lung_ROI_2.mdをえたので、extract_eat_pat_improved_v3_backup.pyを複製したのち、lung_ROI_2.mdを参考に修正して。


もしわからない部分があればAIにクエリするのでプロンプトを作成して。