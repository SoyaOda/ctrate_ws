# プロジェクト概要

## プロジェクトの目的
CT-RATE v2データセットから心臓周囲脂肪（EAT/PAT）を抽出し、2D投影画像を生成するパイプラインの開発。医療画像解析における心臓脂肪組織の定量化を目的としている。

## 技術スタック
- **Python**: 主要プログラミング言語
- **Docker**: TotalSegmentator、BOA（Body-and-Organ-Analysis）の実行環境
- **NVIDIA GPU**: CUDA対応のセグメンテーション処理
- **主要ライブラリ**:
  - nibabel: NIfTI画像ファイル処理
  - numpy: 数値計算
  - scipy: 画像処理（morphological operations）
  - matplotlib: 可視化
  - imageio: 画像IO

## 主要コンポーネント
1. **TotalSegmentator**: 心臓構造セグメンテーション
2. **BOA (Body-and-Organ-Analysis)**: TotalSegmentatorと体組成解析の統合
3. **EAT/PAT抽出**: 心筋からの距離ベース脂肪組織抽出
4. **2D投影生成**: DRR（デジタル再構成X線画像）と脂肪マップ

## データフロー
CT-RATE v2 → 心臓セグメンテーション → EAT/PAT抽出 → 2D投影生成

## 実行環境
- WSL2 Ubuntu (Linux)
- Docker Desktop with GPU support
- NVIDIA GPU with CUDA support