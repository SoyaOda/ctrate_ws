# DiffDRR v0.5.1 APIの混乱：正しい初期化方法について

## 環境
- Python 3.12
- DiffDRR v0.5.1
- TorchIO 0.20.21
- PyTorch 2.8.0
- CUDA対応環境

## 問題の経緯

DiffDRRを使用して胸部CTデータからDRR（Digitally Reconstructed Radiograph）を生成しようとしていますが、APIの正しい使用方法について混乱しています。

### 問題1: DRRクラスのコンストラクタシグネチャ

`DRR`クラスの初期化方法について、相反する情報があります：

#### A. inspect.signatureで確認した結果
```python
from diffdrr.drr import DRR
import inspect
print(inspect.signature(DRR.__init__))
# 出力: (self, subject: 'Subject', sdd: 'float', height: 'int', delx: 'float', ...)
```
これによると、第一引数は`subject`です。

#### B. 直接テンソルを渡す方法（推奨された方法）
```python
drr_generator = DRR(
    volume=volume_lac,          # 3Dテンソル (D, H, W)
    spacing=spacing_tensor,     # 3Dテンソル (sx, sy, sz)
    sdd=1800.0,
    height=512,
    width=512,
    delx=0.7,
    dely=0.7,
)
```
しかし、この方法では以下のエラーが発生します：
```
TypeError: DRR.__init__() missing 1 required positional argument: 'subject'
```

### 問題2: Subjectオブジェクトの作成方法

#### A. read関数を使用（動作確認済み）
```python
from diffdrr.data import read
subject = read('/path/to/ct.nii.gz')
# これは動作します
drr = DRR(subject, sdd=1800.0, height=512, width=512, delx=0.7, dely=0.7)
```

#### B. 手動でSubjectを作成（失敗）
```python
import torchio as tio

# 様々な試行錯誤
# 試行1: TorchIOのSubjectを作成
drr_subject = tio.Subject(
    volume=tio.ScalarImage(tensor=volume_4d, affine=affine),
    density=tio.ScalarImage(tensor=volume_4d, affine=affine),
    spacing=spacing_xyz,
)
drr_subject.reorient = reorient
drr_subject.mask = None
drr_subject.fiducials = None

# 結果: AttributeError（様々な属性が不足）
```

### 問題3: データフローの複雑さ

現在のデータフローは非常に複雑です：

1. **TorchIOでNIfTIを読み込み** → `tio.Subject`（CTとマスク）
2. **3D拡張を適用** → 拡張された`tio.Subject`
3. **DiffDRR用に変換が必要** → ？？？
4. **DRRを生成** → 2D投影画像

ステップ3で、TorchIOの`Subject`をDiffDRRが期待する形式に変換する必要がありますが、その方法が不明瞭です。

## 現在の実装（setup_drr_generator）

```python
def setup_drr_generator(self, subject: tio.Subject) -> DRR:
    """
    TorchIO SubjectからDRRジェネレータを正しくセットアップする。
    ★ DRRクラスにテンソルを直接渡す方式に修正
    """
    # Step A: TorchIOからデータとメタ情報を抽出
    # TorchIO tensor shape: (C, D, H, W)
    # squeeze(0) -> (D, H, W)
    ct_tensor_hu = subject.ct.data.squeeze(0)  # Shape: (D, H, W)
    
    # TorchIO spacing: (X, Y, Z)
    spacing_xyz = subject.ct.spacing
    # DRRクラスのspacing引数が期待する順序 (X, Y, Z)
    spacing_tensor = torch.tensor(spacing_xyz, dtype=torch.float32)
    
    # Step B & C: HU値を線減弱係数(LAC)に変換
    volume_clipped = torch.clamp(ct_tensor_hu, -1024, 3071)
    mu_water = 0.0192  # [mm^-1]
    volume_lac = (volume_clipped + 1024) * (mu_water / 1000.0)
    
    # GPUに転送
    volume_lac = volume_lac.to(self.device)
    spacing_tensor = spacing_tensor.to(self.device)
    
    # ★★★【問題】ここで失敗 ★★★
    # 以下のコードはTypeError: missing 1 required positional argument: 'subject'
    drr_generator = DRR(
        volume=volume_lac,          # 3Dテンソル (D, H, W)
        spacing=spacing_tensor,     # 3Dテンソル (sx, sy, sz)
        sdd=1800.0,
        height=512,
        width=512,
        delx=0.7,
        dely=0.7,
    ).to(self.device)
    
    return drr_generator
```

## 質問

1. **DiffDRR v0.5.1の正しいAPI使用方法は？**
   - `DRR`クラスは`subject`引数を必須としているのか？
   - それとも`volume`と`spacing`を直接渡せるのか？
   - バージョンによってAPIが異なるのか？

2. **TorchIOのSubjectをDiffDRR用に変換する方法**
   - `read`関数を使わずに、TorchIOで拡張したデータをDiffDRRに渡す方法
   - 必要な属性（reorient, mask, fiducials等）の正しい設定方法
   - 軸順序とspacing順序の正しい扱い方

3. **一時ファイルアプローチの是非**
   - 拡張されたデータを一時的にNIfTIファイルに保存してreadする方法：
   ```python
   with tempfile.NamedTemporaryFile(suffix='.nii.gz') as tmp:
       nib.save(nib.Nifti1Image(ct_data, affine), tmp.name)
       diffdrr_subject = read(tmp.name)
   ```
   - この方法は効率的でないが、確実に動作するのか？

4. **代替案の検討**
   - DiffDRR以外のDRR生成ライブラリの使用
   - カスタムレイキャスティング実装
   - DeepDRRなど他のツールへの移行

## 期待される解決策

以下のいずれかの明確な解決策を求めています：

1. **DiffDRR v0.5.1での正しいSubject作成方法**
   - read関数と同等のSubjectオブジェクトを手動で作成する方法
   - 必要な全属性のリストと設定方法

2. **テンソル直接渡しのAPIが存在する場合**
   - 正しいインポートパス
   - 正しい引数の順序と形式

3. **ワークアラウンド**
   - 一時ファイル経由でも良いので、確実に動作する実装方法
   - パフォーマンスを犠牲にしても、まず動作することを優先

具体的なコード例とともに、DiffDRR v0.5.1で確実に動作する実装方法を教えてください。

## 追加情報

デバッグ出力から確認できている情報：
- LAC値の変換は正常（範囲: [0.000000, 0.078624]）
- データ形状は正しい（Shape: torch.Size([1024, 1024, 251])）
- Spacingも正しい（(0.342, 0.342, 1.5)）

しかし、DRRクラスの初期化で失敗しています。

これらについて、公式実装を参考にした解決策を教えてください。