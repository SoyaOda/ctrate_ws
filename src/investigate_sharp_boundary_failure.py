#!/usr/bin/env python3
"""
急激境界が線状にならない根本原因調査

Queryの技術的疑問に基づいた詳細調査:
1. Siddonアルゴリズムでの補間効果
2. center_volume=Trueの座標変換影響  
3. 単一レイでの交点計算詳細追跡
4. 275.2999mm境界の物理座標→投射座標変換精度
"""

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert

class SharpBoundaryInvestigator:
    def __init__(self, mask_path):
        self.mask_path = Path(mask_path)
        
        # 低解像度でメモリ節約
        self.SID = 1800.0
        self.SOD = 1720.0
        self.PX = 0.14 * 2  # 低解像度
        self.H = 256
        self.W = 256
        
        print("🔍 マスクデータ読み込み...")
        torch.cuda.empty_cache()
        
        # center_volume=Falseで座標変換影響を除外
        self.subject_centered = read(str(mask_path), orientation="PA", center_volume=True)
        self.subject_original = read(str(mask_path), orientation="PA", center_volume=False)
        
        print(f"✅ データ読み込み完了")
        print(f"   Centered shape: {self.subject_centered['volume'].data.shape}")
        print(f"   Original shape: {self.subject_original['volume'].data.shape}")

    def analyze_275mm_boundary_location(self):
        """275.2999mm境界の物理座標での正確な位置を解析"""
        print("\n🎯 275.2999mm境界の位置解析")
        
        for name, subject in [("Centered", self.subject_centered), ("Original", self.subject_original)]:
            print(f"\n=== {name} Volume ===")
            
            volume_data = subject["volume"].data[0].cpu().numpy()  # (D,H,W)
            spacing = np.array(subject["volume"].spacing)
            origin = np.array(subject["volume"].origin)
            
            print(f"📏 Spacing: {spacing}")
            print(f"📍 Origin: {origin}")
            print(f"📐 Shape: {volume_data.shape}")
            
            # Z軸での物理座標計算
            z_indices = np.arange(volume_data.shape[0])  # D方向
            z_physical = origin[2] + z_indices * spacing[2]
            
            print(f"🔢 Z物理座標範囲: {z_physical[0]:.3f} - {z_physical[-1]:.3f} mm")
            
            # 275.2999mm付近の検索
            target_z = 275.2999
            closest_idx = np.argmin(np.abs(z_physical - target_z))
            closest_z = z_physical[closest_idx]
            
            print(f"🎯 目標275.2999mmに最も近い位置:")
            print(f"   インデックス: {closest_idx}")
            print(f"   物理座標: {closest_z:.6f}mm")
            print(f"   差分: {abs(closest_z - target_z):.6f}mm")
            
            # 境界付近の密度プロファイル
            z_range = slice(max(0, closest_idx-5), min(volume_data.shape[0], closest_idx+5))
            density_profile = []
            
            for z_idx in range(z_range.start, z_range.stop):
                slice_data = volume_data[z_idx]
                total_voxels = np.sum(slice_data > 0.5)
                density_profile.append((z_physical[z_idx], total_voxels))
                
            print(f"📊 境界付近密度プロファイル:")
            for z_pos, density in density_profile:
                print(f"   {z_pos:.3f}mm: {density} voxels")
            
            # 急激変化の検出
            densities = [d[1] for d in density_profile]
            if len(densities) > 1:
                max_change = max(abs(densities[i+1] - densities[i]) for i in range(len(densities)-1))
                print(f"💥 最大密度変化: {max_change} voxels")

    def create_boundary_focused_drr(self, use_centered=True):
        """境界領域に焦点を当てたDRR生成"""
        print(f"\n🔬 境界焦点DRR生成 (centered={use_centered})")
        
        subject = self.subject_centered if use_centered else self.subject_original
        
        drr = DRR(
            subject,
            sdd=self.SID,
            height=self.H,
            width=self.W,
            delx=self.PX,
            dely=self.PX,
            x0=0.0,  # 中心配置
            y0=0.0,
            renderer="siddon",
            reducefn="sum",
            reverse_x_axis=False
        ).to("cuda", dtype=torch.float32)
        
        # PA投射ポーズ
        alpha, beta, gamma = 0., 90., -90.
        rot = torch.tensor([[torch.deg2rad(torch.tensor(alpha)),
                             torch.deg2rad(torch.tensor(beta)),
                             torch.deg2rad(torch.tensor(gamma))]], device="cuda")
        xyz = torch.tensor([[0., -self.SOD, 0.]], device="cuda")
        pose = convert(rot, xyz, parameterization="euler_angles", convention="YZX")
        
        with torch.inference_mode():
            img = drr(pose)[0,0].detach().cpu()
            # radiologic表現
            img = torch.flip(img, dims=[1])
        
        return img, drr, pose

    def create_ideal_sharp_boundary_phantom(self):
        """理想的な急激境界ファントム（275mm以下のみ1）を作成"""
        print("\n🧪 理想急激境界ファントム作成")
        
        # オリジナルマスクをベースに275mm以上を完全にゼロ化
        volume_data = self.subject_centered["volume"].data[0].clone()  # (D,H,W)
        spacing = np.array(self.subject_centered["volume"].spacing)
        origin = np.array(self.subject_centered["volume"].origin)
        
        # Z軸物理座標
        z_indices = np.arange(volume_data.shape[0])
        z_physical = origin[2] + z_indices * spacing[2]
        
        # 275mm以上を完全ゼロ化
        cutoff_z = 275.0
        mask_indices = z_physical > cutoff_z
        
        print(f"📐 カットオフ: {cutoff_z}mm")
        print(f"🔄 ゼロ化スライス数: {mask_indices.sum()}")
        
        # ファントムデータ作成
        phantom_data = volume_data.clone()
        phantom_data[mask_indices] = 0.0
        
        # 新しいSubject作成（TorchIOのtensor引数を使用）
        phantom_subject = {
            'volume': type(self.subject_centered['volume'])(
                tensor=phantom_data.unsqueeze(0),  # (1,D,H,W)に戻す
                affine=self.subject_centered['volume'].affine,
                spacing=self.subject_centered['volume'].spacing,
                origin=self.subject_centered['volume'].origin
            )
        }
        
        return phantom_subject

    def compare_original_vs_phantom(self):
        """オリジナルマスクと理想ファントムの投射比較"""
        print("\n🔄 オリジナル vs ファントム比較")
        
        # 理想ファントム作成
        phantom_subject = self.create_ideal_sharp_boundary_phantom()
        
        results = {}
        
        # オリジナル投射
        img_original, _, _ = self.create_boundary_focused_drr(use_centered=True)
        results['original'] = img_original
        
        # ファントム投射（DRRを新しく作成）
        drr_phantom = DRR(
            phantom_subject,
            sdd=self.SID,
            height=self.H,
            width=self.W,
            delx=self.PX,
            dely=self.PX,
            x0=0.0,
            y0=0.0,
            renderer="siddon",
            reducefn="sum",
            reverse_x_axis=False
        ).to("cuda", dtype=torch.float32)
        
        alpha, beta, gamma = 0., 90., -90.
        rot = torch.tensor([[torch.deg2rad(torch.tensor(alpha)),
                             torch.deg2rad(torch.tensor(beta)),
                             torch.deg2rad(torch.tensor(gamma))]], device="cuda")
        xyz = torch.tensor([[0., -self.SOD, 0.]], device="cuda")
        pose = convert(rot, xyz, parameterization="euler_angles", convention="YZX")
        
        with torch.inference_mode():
            img_phantom = drr_phantom(pose)[0,0].detach().cpu()
            img_phantom = torch.flip(img_phantom, dims=[1])
        
        results['phantom'] = img_phantom
        
        return results

    def analyze_gradient_comparison(self, results):
        """勾配比較解析"""
        print("\n📊 勾配比較解析")
        
        for name, img in results.items():
            print(f"\n=== {name.upper()} ===")
            
            # 中央行での勾配
            v0 = self.H // 2
            u_range = np.arange(self.W // 4, 3 * self.W // 4)
            
            line = img[v0, u_range].numpy()
            gradient = np.abs(np.diff(line))
            max_gradient = gradient.max() if len(gradient) > 0 else 0
            max_pos = u_range[np.argmax(gradient)] if len(gradient) > 0 else 0
            
            print(f"📈 最大勾配: {max_gradient:.6f}")
            print(f"📍 最大勾配位置: u={max_pos}")
            
            # 全体勾配統計
            grad_x = torch.abs(torch.diff(img, dim=1))
            grad_y = torch.abs(torch.diff(img, dim=0))
            
            print(f"🌍 X方向勾配 - 平均: {grad_x.mean():.6f}, 最大: {grad_x.max():.6f}")
            print(f"🌍 Y方向勾配 - 平均: {grad_y.mean():.6f}, 最大: {grad_y.max():.6f}")

def main():
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    if not Path(mask_path).exists():
        print(f"❌ マスクファイルが見つかりません: {mask_path}")
        return
        
    investigator = SharpBoundaryInvestigator(mask_path)
    
    print("="*60)
    print("🔬 急激境界問題の根本原因調査")
    print("="*60)
    
    # 1. 275.2999mm境界の物理位置解析
    investigator.analyze_275mm_boundary_location()
    
    # 2. オリジナル vs 理想ファントム比較
    print("\n" + "="*40)
    print("🧪 投射比較実験")
    print("="*40)
    
    comparison_results = investigator.compare_original_vs_phantom()
    
    # 3. 勾配解析
    investigator.analyze_gradient_comparison(comparison_results)
    
    # 4. 視覚化保存
    output_dir = Path("outputs/sharp_boundary_investigation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # オリジナル
    im1 = axes[0,0].imshow(comparison_results['original'], cmap='gray')
    axes[0,0].set_title('Original Mask DRR')
    plt.colorbar(im1, ax=axes[0,0])
    
    # ファントム
    im2 = axes[0,1].imshow(comparison_results['phantom'], cmap='gray')  
    axes[0,1].set_title('Ideal Sharp Phantom DRR')
    plt.colorbar(im2, ax=axes[0,1])
    
    # 差分
    diff = torch.abs(comparison_results['original'] - comparison_results['phantom'])
    im3 = axes[1,0].imshow(diff, cmap='hot')
    axes[1,0].set_title('Absolute Difference')
    plt.colorbar(im3, ax=axes[1,0])
    
    # 中央行プロファイル比較
    v0 = investigator.H // 2
    axes[1,1].plot(comparison_results['original'][v0], label='Original', alpha=0.7)
    axes[1,1].plot(comparison_results['phantom'][v0], label='Phantom', alpha=0.7)
    axes[1,1].set_title(f'Central Line Profile (y={v0})')
    axes[1,1].legend()
    axes[1,1].set_xlabel('X coordinate')
    axes[1,1].set_ylabel('Thickness [mm]')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sharp_boundary_investigation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📁 結果保存: {output_dir}/sharp_boundary_investigation.png")
    print("🏁 根本原因調査完了")

if __name__ == "__main__":
    main()