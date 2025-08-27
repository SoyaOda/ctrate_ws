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

EAT＋PATの対象領域（Shell）に関して、下記の方針で上端と下端を適応させて。

「上端」
改善提案：上限のクリップを“解剖学的に”安定化

目的：EDT膨張の“帽子（cap）”で心膜外の高さまで Shell がにじむのを防ぎ、症例ごとに近位大血管が被覆される高さで機械的に止める。

1) 「近位大血管・心膜近接」から Z 上限を決める

TotalSegmentator の aorta と pulmonary_artery を使い、心臓マスクを r=5 mm 膨張した近接帯とANDを取ります（= 「心膜が被覆し得る近位部」の近似）。その存在する最後のZを**上限（+1〜2スライスの安全マージン）**にする方法です。

パッチ（あなたの統合版にそのまま差し込める最小例）：

from scipy.ndimage import binary_dilation, generate_binary_structure
import nibabel as nib
import numpy as np
from pathlib import Path

def compute_intrapericardial_zmax(totalseg_dir, heart_mask, spacing,
                                  prox_mm=5.0, z_margin_slices=1, verbose=False):
    se3d = generate_binary_structure(3, 2)
    it = max(1, int(np.ceil(prox_mm / min(spacing))))
    prox_band = binary_dilation(heart_mask, structure=se3d, iterations=it)

    def _load(name):
        p = Path(totalseg_dir) / f"{name}.nii.gz"
        return (nib.load(str(p)).get_fdata() > 0) if p.exists() else None

    aorta = _load("aorta")
    pa    = _load("pulmonary_artery")

    intrapericardial = heart_mask.copy()
    for v in [aorta, pa]:
        if v is not None:
            intrapericardial |= (v & prox_band)

    z_any = np.any(intrapericardial, axis=(0,1))
    z_idx = np.where(z_any)[0]
    if z_idx.size == 0:
        return np.where(np.any(heart_mask, axis=(0,1)))[0][-1]  # 心臓だけで決定
    zmax = min(z_idx[-1] + z_margin_slices, heart_mask.shape[2]-1)
    if verbose:
        print(f"  Intrapericardial zmax (with proximal great vessels): {zmax}")
    return zmax


適用：

z_top = compute_intrapericardial_zmax(totalseg_dir, heart_mask, spacing, prox_mm=5.0, z_margin_slices=1)
# Shell と fat の両方をこの上限でゼロ化
shell_mask[:, :, z_top+1:] = False
fat_near_heart[:, :, z_top+1:] = False


根拠：心膜は近位の上行大動脈・肺動脈幹の周囲で反転・被覆し、胸骨角で終わる。上式はその機能的近似です。
Radiopaedia
+1


「下端」：z_bottom_heart = 心臓マスクが存在する最も尾側のZ

もしわからない部分があればAIにクエリするのでプロンプトを作成して。