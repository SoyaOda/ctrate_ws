#!/usr/bin/env python3
"""
マスク投射での線状境界問題の詳細デバッグシステム

Query回答に基づいて：
1. 単一レイの詳細追跡
2. 自作Siddonアルゴリズムとの比較
3. voxel_shift、Euler順序、主点設定の検証
4. 線状境界の定量的検出
"""

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert
from skimage.metrics import structural_similarity as ssim

class MaskProjectionDebugger:
    def __init__(self, mask_path, output_dir="outputs/debug_mask_projection"):
        self.mask_path = Path(mask_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # DRRパラメータ（メモリ節約のため低解像度）
        self.SID = 1800.0
        self.SOD = 1720.0
        self.PX = 0.14 * (3070/512)  # 解像度に合わせて調整
        self.H = 512
        self.W = 512
        
        # GPUメモリクリア
        torch.cuda.empty_cache()
        
        # マスクデータ読み込み
        print("🔍 マスクデータ読み込み中...")
        self.subject = read(str(mask_path), orientation="PA", center_volume=True)
        print(f"✅ マスクデータ読み込み完了: {self.subject['volume'].data.shape}")
        
        # GPUメモリクリア
        torch.cuda.empty_cache()
        
        # 物理パラメータ取得
        self.spacing = np.array(self.subject["volume"].spacing)
        self.origin = np.array(self.subject["volume"].origin)
        self.shape = self.subject["volume"].data.shape[1:]  # (D,H,W) -> (H,W,D)
        
        print(f"📏 Spacing: {self.spacing}")
        print(f"📍 Origin: {self.origin}")
        print(f"📐 Shape: {self.shape}")

    def create_drr(self, reverse_x_axis=False, convention="YZX"):
        """DRRインスタンス作成"""
        drr = DRR(
            self.subject,
            sdd=self.SID,
            height=self.H,
            width=self.W,
            delx=self.PX,
            dely=self.PX,
            x0=+0.5*self.PX if self.W%2==0 else 0.0,  # 偶数サイズ半画素補正
            y0=-0.5*self.PX if self.H%2==0 else 0.0,  # 偶数サイズ半画素補正
            renderer="siddon",
            reducefn="sum",
            reverse_x_axis=reverse_x_axis
        ).to("cuda", dtype=torch.float32)
        
        return drr

    def get_pose(self, convention="YZX"):
        """PA投射のポーズ取得"""
        # DeepDRR→DiffDRRのPA角度
        alpha, beta, gamma = 0., 90., -90.
        rot = torch.tensor([[torch.deg2rad(torch.tensor(alpha)),
                             torch.deg2rad(torch.tensor(beta)),
                             torch.deg2rad(torch.tensor(gamma))]], device="cuda")
        xyz = torch.tensor([[0., -self.SOD, 0.]], device="cuda")
        
        pose = convert(rot, xyz, parameterization="euler_angles", convention=convention)
        return pose

    def render_mask_drr(self, reverse_x_axis=False, convention="YZX"):
        """マスクDRRレンダリング"""
        drr = self.create_drr(reverse_x_axis, convention)
        pose = self.get_pose(convention)
        
        with torch.inference_mode():
            img = drr(pose)
        
        img = img[0,0].detach().cpu()
        
        # radiologic表現の場合は水平反転
        if not reverse_x_axis:
            img = torch.flip(img, dims=[1])
            
        return img, drr, pose

    def trace_single_ray(self, drr, pose, u, v):
        """単一レイの詳細追跡"""
        print(f"\n🔍 レイ追跡: ピクセル({u}, {v})")
        
        # detector()でX線源と各画素の3D座標取得
        src, target = drr.detector(pose, calibration=None)
        
        source = src[0,0].detach().cpu().numpy()
        target_point = target[v,u].detach().cpu().numpy()
        
        print(f"📍 X線源: {source}")
        print(f"🎯 ターゲット: {target_point}")
        
        # 自作Siddonで交差長計算
        total_mm, segments = self.siddon_trace(source, target_point)
        
        # DiffDRRでの同一ピクセル値
        with torch.inference_mode():
            img = drr(pose)[0,0].detach().cpu()
            if not drr.reverse_x_axis:
                img = torch.flip(img, dims=[1])
            diffdrr_value = img[v, u].item()
        
        print(f"🧮 自作Siddon厚み: {total_mm:.6f} mm")
        print(f"🖥️  DiffDRR値: {diffdrr_value:.6f} mm")
        print(f"📊 差分: {abs(total_mm - diffdrr_value):.6f} mm")
        
        return {
            'source': source,
            'target': target_point,
            'siddon_thickness': total_mm,
            'diffdrr_value': diffdrr_value,
            'difference': abs(total_mm - diffdrr_value),
            'segments': segments
        }

    def siddon_trace(self, source, target):
        """自作Siddonアルゴリズムで交差長計算"""
        # Ray parametric form: p(t) = source + t * (target - source)
        d = target - source
        inv_d = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        
        # 体積の最小/最大境界（世界座標）
        vol_min = self.origin
        vol_max = self.origin + self.spacing * (np.array(self.shape) - 1)
        
        # 体積との交差範囲
        tmins = (vol_min - source) * inv_d
        tmaxs = (vol_max - source) * inv_d
        t0 = np.max(np.minimum(tmins, tmaxs))
        t1 = np.min(np.maximum(tmins, tmaxs))
        
        if t0 >= t1:
            return 0.0, []
        
        # Siddon: 各軸のセル境界でのt値
        def plane_params(axis):
            idx = np.arange(self.shape[axis]+1)
            # voxel_shift=0.5（ボクセル中心）基準
            plane = self.origin[axis] + self.spacing[axis]*(idx - 0.5)
            ts = (plane - source[axis]) * inv_d[axis]
            return ts
        
        ts = [t0, t1]
        for ax in range(3):
            ts.extend(plane_params(ax))
        ts = np.array(sorted(t for t in ts if t0 <= t <= t1))
        
        # 各区間での交差長計算
        segments = []
        total_mm = 0.0
        vol = self.subject["volume"].data[0].cpu().numpy()  # (D,H,W)
        
        for i in range(len(ts)-1):
            tm = 0.5*(ts[i]+ts[i+1])
            pm = source + tm*d
            # voxel index（center基準）
            idx = np.floor((pm - self.origin)/self.spacing + 0.5).astype(int)
            
            if np.any(idx < 0) or np.any(idx >= np.array(self.shape)):
                continue
                
            seg_len_mm = np.linalg.norm((ts[i+1]-ts[i]) * d)
            segments.append((tuple(idx.tolist()), seg_len_mm))
            
            # マスクが1の場合のみ厚みに加算
            if vol[idx[2], idx[1], idx[0]] > 0.5:  # (D,H,W)順序
                total_mm += seg_len_mm
                
        return total_mm, segments

    def detect_sharp_boundaries(self, img, save_analysis=True):
        """線状境界の定量的検出"""
        print("\n🔍 線状境界検出開始...")
        
        # 中央行での勾配解析
        v0 = self.H // 2
        u_range = np.arange(self.W // 2 - 200, self.W // 2 + 200)
        
        line = img[v0, u_range].numpy()
        gradient = np.abs(np.diff(line))
        max_gradient = gradient.max()
        max_pos = u_range[np.argmax(gradient)]
        
        print(f"📊 最大勾配: {max_gradient:.6f}")
        print(f"📍 最大勾配位置: u={max_pos}")
        
        # 全画像での勾配統計
        grad_x = torch.abs(torch.diff(img, dim=1))
        grad_y = torch.abs(torch.diff(img, dim=0))
        
        print(f"📈 X方向勾配統計 - 平均: {grad_x.mean():.6f}, 最大: {grad_x.max():.6f}")
        print(f"📈 Y方向勾配統計 - 平均: {grad_y.mean():.6f}, 最大: {grad_y.max():.6f}")
        
        if save_analysis:
            # 勾配解析の可視化
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 元画像
            im1 = axes[0,0].imshow(img, cmap='gray')
            axes[0,0].set_title('マスク厚みDRR')
            axes[0,0].axhline(y=v0, color='r', linestyle='--', alpha=0.7)
            plt.colorbar(im1, ax=axes[0,0])
            
            # 中央行のプロファイル
            axes[0,1].plot(u_range[:-1], gradient)
            axes[0,1].set_title(f'中央行勾配 (y={v0})')
            axes[0,1].set_xlabel('X座標')
            axes[0,1].set_ylabel('勾配強度')
            axes[0,1].axvline(x=max_pos, color='r', linestyle='--', alpha=0.7)
            
            # X方向勾配
            im3 = axes[1,0].imshow(grad_x, cmap='hot')
            axes[1,0].set_title('X方向勾配')
            plt.colorbar(im3, ax=axes[1,0])
            
            # Y方向勾配
            im4 = axes[1,1].imshow(grad_y, cmap='hot')
            axes[1,1].set_title('Y方向勾配')
            plt.colorbar(im4, ax=axes[1,1])
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'gradient_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        return {
            'max_gradient': max_gradient,
            'max_gradient_position': max_pos,
            'x_gradient_stats': {'mean': grad_x.mean().item(), 'max': grad_x.max().item()},
            'y_gradient_stats': {'mean': grad_y.mean().item(), 'max': grad_y.max().item()}
        }

    def parameter_sweep(self):
        """パラメータ検証（reverse_x_axis、Euler順序、主点設定）"""
        print("\n🧪 パラメータ検証開始...")
        
        results = {}
        
        # reverse_x_axis比較
        for rxa in [False, True]:
            print(f"\n🔄 reverse_x_axis = {rxa}")
            img, drr, pose = self.render_mask_drr(reverse_x_axis=rxa)
            
            # 中央ピクセルでのレイ追跡
            u, v = self.W//2, self.H//2
            trace_result = self.trace_single_ray(drr, pose, u, v)
            
            # 境界検出
            boundary_result = self.detect_sharp_boundaries(img, save_analysis=False)
            
            results[f'reverse_x_axis_{rxa}'] = {
                'trace': trace_result,
                'boundaries': boundary_result,
                'image': img
            }
        
        # Euler順序比較
        conventions = ["ZXY", "YZX", "ZYX", "XYZ"]
        for conv in conventions:
            print(f"\n🔄 Euler順序 = {conv}")
            try:
                img, drr, pose = self.render_mask_drr(convention=conv)
                
                # 中央ピクセルでのレイ追跡
                u, v = self.W//2, self.H//2
                trace_result = self.trace_single_ray(drr, pose, u, v)
                
                # 境界検出
                boundary_result = self.detect_sharp_boundaries(img, save_analysis=False)
                
                results[f'euler_{conv}'] = {
                    'trace': trace_result,
                    'boundaries': boundary_result,
                    'image': img
                }
            except Exception as e:
                print(f"❌ {conv} でエラー: {e}")
        
        return results

def main():
    # マスクパス
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    if not Path(mask_path).exists():
        print(f"❌ マスクファイルが見つかりません: {mask_path}")
        return
    
    # デバッガー初期化
    debugger = MaskProjectionDebugger(mask_path)
    
    print("\n" + "="*50)
    print("🚀 マスク投射デバッグ開始")
    print("="*50)
    
    # 1. 基本的なマスクDRR生成と境界検出
    print("\n1️⃣ 基本マスクDRR生成...")
    img, drr, pose = debugger.render_mask_drr()
    
    # DRR保存
    plt.figure(figsize=(12, 8))
    plt.imshow(img, cmap='gray')
    plt.title('マスク厚みDRR (基本設定)')
    plt.colorbar(label='厚み [mm]')
    plt.savefig(debugger.output_dir / 'mask_drr_baseline.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 線状境界検出
    print("\n2️⃣ 線状境界検出...")
    boundary_results = debugger.detect_sharp_boundaries(img)
    
    # 3. 中央ピクセルでのレイ追跡
    print("\n3️⃣ 中央ピクセルレイ追跡...")
    u, v = debugger.W//2, debugger.H//2
    trace_results = debugger.trace_single_ray(drr, pose, u, v)
    
    # 4. パラメータ検証
    print("\n4️⃣ パラメータ検証...")
    param_results = debugger.parameter_sweep()
    
    # 結果サマリー
    print("\n" + "="*50)
    print("📋 デバッグ結果サマリー")
    print("="*50)
    
    print(f"\n🎯 境界検出結果:")
    print(f"  最大勾配: {boundary_results['max_gradient']:.6f}")
    print(f"  線状境界の判定: {'✅ 検出' if boundary_results['max_gradient'] > 0.1 else '❌ 未検出'}")
    
    print(f"\n🧮 レイ追跡結果:")
    print(f"  自作Siddon: {trace_results['siddon_thickness']:.6f} mm")
    print(f"  DiffDRR: {trace_results['diffdrr_value']:.6f} mm")
    print(f"  整合性: {'✅ 一致' if trace_results['difference'] < 0.001 else '❌ 不一致'}")
    
    print(f"\n📁 出力ディレクトリ: {debugger.output_dir}")
    print("🏁 デバッグ完了")

if __name__ == "__main__":
    main()