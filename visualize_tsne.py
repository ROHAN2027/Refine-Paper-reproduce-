"""
t-SNE Feature-Space Visualization for REFINE Defense
=====================================================
Fully standalone — no dependency on `core` module (avoids lpips issue).
Creates poisoned images manually and loads models directly.

Usage:
  python visualize_tsne.py
"""

import os
import sys
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import cv2
from torchvision import transforms
from torchvision.datasets import DatasetFolder
import time

# ========================
# ResNet18 (copied from core/models/resnet.py to avoid core imports)
# ========================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.linear = nn.Linear(512, num_classes)
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)
    def from_input_to_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

# ========================
# UNet (from utils/unet.py)
# ========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch: mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dY = x2.size(2) - x1.size(2); dX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNetLittle(nn.Module):
    def __init__(self, first_channels=64):
        super().__init__()
        fc = first_channels
        self.inc = DoubleConv(3, fc)
        self.down1 = Down(fc, fc*2)
        self.down2 = Down(fc*2, fc*4)
        self.down3 = Down(fc*4, fc*8)
        self.down4 = Down(fc*8, fc*16 // 2)
        self.up1 = Up(fc*16, fc*8 // 2)
        self.up2 = Up(fc*8, fc*4 // 2)
        self.up3 = Up(fc*4, fc*2 // 2)
        self.up4 = Up(fc*2, fc)
        self.outc = OutConv(fc, 3)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3)
        x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# ========================
# Poisoned dataset wrapper (manual BadNets trigger injection)
# ========================
class PoisonedDatasetFolder(Dataset):
    """Wraps a DatasetFolder and injects BadNets trigger to ALL samples."""
    def __init__(self, clean_dataset, pattern, weight):
        self.dataset = clean_dataset
        self.pattern = pattern  # (C, H, W) uint8 trigger pattern
        self.weight = weight    # (H, W) float mask
        # Pre-compute: res = weight * pattern, w = 1 - weight
        if self.pattern.dim() == 2:
            self.pattern = self.pattern.unsqueeze(0)
        if self.weight.dim() == 2:
            self.weight = self.weight.unsqueeze(0)
        self.res = self.weight * self.pattern
        self.w = 1.0 - self.weight

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, target = self.dataset.samples[idx]
        img = self.dataset.loader(path)  # numpy HxWxC (cv2)
        # Apply trigger: img = (1-weight)*img + weight*pattern
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # CxHxW uint8
        img_t = (self.w * img_t + self.res).to(torch.uint8)
        img_t = img_t.permute(1, 2, 0).numpy()  # back to HxWxC
        # Apply the original transform
        if self.dataset.transform:
            img_t = self.dataset.transform(img_t)
        # Target stays the same (we want to see where poisoned inputs map)
        return img_t, target

# ========================
# Configuration
# ========================
DATA_ROOT = './data/CIFAR10'
REFINE_RES_BASE = '../refine_res/CIFAR10/RestNet18/BadNets'
ATTACK_TRIGGERS = './attack/triggers/CIFAR10_pattern.pth'
OUTPUT_DIR = './tsne_results'
NUM_SAMPLES = 500
TSNE_PERPLEXITY = 25
BATCH_SIZE = 32
IMG_SIZE = 32
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]

torch.manual_seed(666)
np.random.seed(666)
random.seed(666)


def find_latest_refine_dir():
    if not os.path.exists(REFINE_RES_BASE):
        raise FileNotFoundError(f"Not found: {REFINE_RES_BASE}")
    subdirs = sorted([d for d in os.listdir(REFINE_RES_BASE) 
                       if os.path.isdir(os.path.join(REFINE_RES_BASE, d))])
    return os.path.join(REFINE_RES_BASE, subdirs[-1])


def find_attack_checkpoint():
    """Look for the backdoored model checkpoint."""
    attack_dir = './attack/CIFAR10/ResNet18/BadNets'
    if os.path.exists(attack_dir):
        for item in os.listdir(attack_dir):
            item_path = os.path.join(attack_dir, item)
            if os.path.isdir(item_path) and item.startswith("Normalize"):
                ckpts = [f for f in os.listdir(item_path) if f.endswith('.pth') and 'ckpt' in f]
                if ckpts:
                    # Sort by the epoch number (e.g., ckpt_epoch_150.pth -> 150)
                    ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    return os.path.join(item_path, ckpts[-1])
    return None


def get_dummy_model(device):
    """Return an untrained model if the checkpoint is missing."""
    print("\n" + "!"*60)
    print("WARNING: BACKDOORED MODEL CHECKPOINT NOT FOUND!")
    print("The script needs the ResNet18 weights from your Kaggle run:")
    print("  -> attack/CIFAR10/ResNet18/BadNets/Normalize_.../ckpt_epoch_150.pth")
    print("\nBecause it's missing, using an UNTRAINED ResNet18 for now.")
    print("The t-SNE plots will NOT show a strong backdoor cluster until")
    print("you download the ckpt_epoch_150.pth file and place it in the")
    print("attack/ directory.")
    print("!"*60 + "\n")
    model = ResNet18(num_classes=10).to(device)
    return model

def extract_features(model, dataloader, device, max_samples=200):
    model.eval()
    feats, labs = [], []
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="  Extracting", leave=False):
            if total >= max_samples: break
            imgs = imgs.to(device)
            f = model.from_input_to_features(imgs)
            f = F.avg_pool2d(f, f.size(-1)).view(f.size(0), -1)
            feats.append(f.cpu().numpy())
            labs.append(labels.numpy())
            total += imgs.size(0)
    return np.concatenate(feats)[:max_samples], np.concatenate(labs)[:max_samples]


def extract_features_unet(model, unet, dataloader, device, max_samples=200):
    model.eval(); unet.eval()
    feats, labs = [], []
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="  Extracting (REFINE)", leave=False):
            if total >= max_samples: break
            imgs = imgs.to(device)
            transformed = torch.clamp(unet(imgs), 0, 1)
            f = model.from_input_to_features(transformed)
            f = F.avg_pool2d(f, f.size(-1)).view(f.size(0), -1)
            feats.append(f.cpu().numpy())
            labs.append(labels.numpy())
            total += imgs.size(0)
    return np.concatenate(feats)[:max_samples], np.concatenate(labs)[:max_samples]


def plot_panel(ax, emb, labels, title, num_classes=10, show_legend=False):
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    cmap = ListedColormap(colors[:num_classes])
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=cmap,
                    s=15, alpha=0.7, edgecolors='none', rasterized=True)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    if show_legend:
        ax.legend(*sc.legend_elements(num=num_classes), title="Class",
                  loc='upper right', fontsize=7, title_fontsize=8,
                  markerscale=1.5, framealpha=0.8)
    return sc


def plot_combined(ax, ec, lc, ep, lp, title, nc=10):
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    cmap = ListedColormap(colors[:nc])
    ax.scatter(ec[:, 0], ec[:, 1], c=lc, cmap=cmap, s=12, alpha=0.5,
               edgecolors='none', marker='o', rasterized=True)
    ax.scatter(ep[:, 0], ep[:, 1], c='black', s=25, alpha=0.8,
               marker='*', label='Poisoned', rasterized=True)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.8)


def main():
    device = torch.device('cpu')
    print(f"Device: {device}")
    start_time = time.time()
    
    # ========== Step 1: Load backdoored model ==========
    print("\n[1/5] Loading backdoored ResNet18...")
    ckpt = find_attack_checkpoint()
    if ckpt:
        print(f"  Found checkpoint: {ckpt}")
        model = ResNet18(num_classes=10)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        model = model.to(device)
        model.eval()
    else:
        model = get_dummy_model(device)
    print("  ✅ Backdoored model ready")
    
    # ========== Step 2: Load UNet ==========
    print("\n[2/5] Loading REFINE UNet...")
    refine_dir = find_latest_refine_dir()
    unet_files = sorted([f for f in os.listdir(refine_dir) 
                          if f.startswith('unet_epoch') and f.endswith('.pth')])
    unet_path = os.path.join(refine_dir, unet_files[-1])
    print(f"  Loading: {unet_path}")
    
    unet = UNetLittle(first_channels=64)
    unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=False))
    unet = unet.to(device)
    unet.eval()
    print(f"  ✅ UNet loaded ({unet_files[-1]})")
    
    # ========== Step 3: Load datasets ==========
    print("\n[3/5] Loading datasets...")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMG_SIZE),
        transforms.Normalize(MEAN, STD),
    ])
    
    testset = DatasetFolder(root=os.path.join(DATA_ROOT, 'test'),
                            transform=transform_test, loader=cv2.imread,
                            extensions=('png', 'jpeg',))
    
    ptestset = DatasetFolder(root=os.path.join(DATA_ROOT, 'test_remove_0'),
                             transform=transform_test, loader=cv2.imread,
                             extensions=('png', 'jpeg',))
    
    # Create poisoned version (100% poisoned for ASR testing)
    pattern = torch.load(ATTACK_TRIGGERS, weights_only=False)
    weight = torch.zeros((IMG_SIZE, IMG_SIZE), dtype=torch.float32)
    weight[-3:, -3:] = 1.0
    poisoned_testset = PoisonedDatasetFolder(ptestset, pattern, weight)
    
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    poison_loader = DataLoader(poisoned_testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"  ✅ Clean test: {len(testset)} | Poisoned test: {len(poisoned_testset)}")
    
    # ========== Step 3.5: Visualize Image Transformations ==========
    print("\n[3.5/5] Generating Image Transformation Grids...")
    def visualize_images():
        model.eval()
        unet.eval()
        
        # Get one batch of clean and one of poisoned
        clean_imgs, clean_labels = next(iter(test_loader))
        poison_imgs, poison_labels = next(iter(poison_loader))
        
        # Take 4 images
        c_imgs = clean_imgs[:4].to(device)
        p_imgs = poison_imgs[:4].to(device)
        
        # Apply REFINE (UNet)
        with torch.no_grad():
            ref_c_imgs = torch.clamp(unet(c_imgs), 0, 1)
            ref_p_imgs = torch.clamp(unet(p_imgs), 0, 1)
            
        # Denormalize for plotting
        def denorm(t):
            t = t.clone().cpu()
            for i in range(3):
                t[:, i, :, :] = t[:, i, :, :] * STD[i] + MEAN[i]
            return torch.clamp(t, 0, 1).permute(0, 2, 3, 1).numpy()
            
        c_np = denorm(c_imgs)
        p_np = denorm(p_imgs)
        rc_np = ref_c_imgs.clone().cpu().permute(0, 2, 3, 1).numpy()
        rp_np = ref_p_imgs.clone().cpu().permute(0, 2, 3, 1).numpy()
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle('Image Transformations Before & After REFINE', fontsize=16, fontweight='bold')
        
        row_titles = ['Clean Original', 'Poisoned (Trigger added)', 'REFINE Clean', 'REFINE Poisoned']
        img_sets = [c_np, p_np, rc_np, rp_np]
        
        for row in range(4):
            for col in range(4):
                ax = axes[row, col]
                ax.imshow(img_sets[row][col])
                ax.axis('off')
                if col == 0:
                    ax.set_title(row_titles[row], loc='left', fontsize=12, fontweight='bold', pad=10)
        
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        img_path = os.path.join(OUTPUT_DIR, 'image_transformations.png')
        fig.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✅ Saved: {img_path}")
        
    visualize_images()
    
    # ========== Step 4: Extract features ==========
    print(f"\n[4/5] Extracting features ({NUM_SAMPLES} samples each)...")
    
    print("  (a) Clean → Backdoored ResNet18")
    feat_c, lab_c = extract_features(model, test_loader, device, NUM_SAMPLES)
    
    print("  (b) Poisoned → Backdoored ResNet18")
    feat_p, lab_p = extract_features(model, poison_loader, device, NUM_SAMPLES)
    
    print("  (c) REFINE(Clean) → Backdoored ResNet18")
    feat_rc, lab_rc = extract_features_unet(model, unet, test_loader, device, NUM_SAMPLES)
    
    print("  (d) REFINE(Poisoned) → Backdoored ResNet18")
    feat_rp, lab_rp = extract_features_unet(model, unet, poison_loader, device, NUM_SAMPLES)
    
    print(f"  Feature dim: {feat_c.shape[1]}")
    
    # ========== Step 5: t-SNE + Plot ==========
    print(f"\n[5/5] Running t-SNE & generating plots...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tsne_kw = dict(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=42,
                   max_iter=1000, learning_rate='auto', init='pca')
    
    print("  Computing t-SNE for 4 scenarios...")
    emb_a = TSNE(**tsne_kw).fit_transform(feat_c)
    emb_b = TSNE(**tsne_kw).fit_transform(feat_p)
    emb_c = TSNE(**tsne_kw).fit_transform(feat_rc)
    emb_d = TSNE(**tsne_kw).fit_transform(feat_rp)
    
    # --- Figure 1: 4-panel ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('t-SNE Feature Visualization — CIFAR-10 / ResNet18 / BadNets',
                 fontsize=15, fontweight='bold', y=1.02)
    plot_panel(axes[0], emb_a, lab_c, '(a) Clean Inputs\n(No Defense)', show_legend=True)
    plot_panel(axes[1], emb_b, lab_p, '(b) Poisoned Inputs\n(No Defense)')
    plot_panel(axes[2], emb_c, lab_rc, '(c) Clean Inputs\n(After REFINE)')
    plot_panel(axes[3], emb_d, lab_rp, '(d) Poisoned Inputs\n(After REFINE)')
    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, 'tsne_4panel.png')
    fig.savefig(p1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Saved: {p1}")
    
    # --- Figure 2: Before vs After ---
    print("  Computing t-SNE for before/after comparison...")
    feat_before = np.concatenate([feat_c, feat_p])
    emb_before = TSNE(**tsne_kw).fit_transform(feat_before)
    feat_after = np.concatenate([feat_rc, feat_rp])
    emb_after = TSNE(**tsne_kw).fit_transform(feat_after)
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('t-SNE — Backdoor Cluster Before vs After REFINE',
                  fontsize=15, fontweight='bold', y=1.02)
    n_c = len(feat_c)
    plot_combined(axes2[0], emb_before[:n_c], lab_c, emb_before[n_c:], lab_p,
                  'Before REFINE\n(Poisoned ★ form tight cluster)')
    n_rc = len(feat_rc)
    plot_combined(axes2[1], emb_after[:n_rc], lab_rc, emb_after[n_rc:], lab_rp,
                  'After REFINE\n(Poisoned ★ cluster disrupted)')
    plt.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, 'tsne_before_after.png')
    fig2.savefig(p2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"  ✅ Saved: {p2}")
    
    # --- Figure 3: Individual high-res ---
    for name, emb, labels, title in [
        ('clean', emb_a, lab_c, 'Clean → Backdoored Model'),
        ('poisoned', emb_b, lab_p, 'Poisoned → Backdoored Model'),
        ('refine_clean', emb_c, lab_rc, 'REFINE(Clean) → Backdoored Model'),
        ('refine_poisoned', emb_d, lab_rp, 'REFINE(Poisoned) → Backdoored Model'),
    ]:
        fig_s, ax_s = plt.subplots(1, 1, figsize=(7, 6))
        plot_panel(ax_s, emb, labels, title, show_legend=True)
        plt.tight_layout()
        ps = os.path.join(OUTPUT_DIR, f'tsne_{name}.png')
        fig_s.savefig(ps, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_s)
        print(f"  ✅ Saved: {ps}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"✅ All 6 plots saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"   Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
