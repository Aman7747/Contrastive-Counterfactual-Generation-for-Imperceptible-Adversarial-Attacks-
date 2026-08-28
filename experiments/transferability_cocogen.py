"""
CoCoGen (CCG) + High-Frequency Attack — Transferability and Robustness Analysis
================================================================================
Generates two tables:
  Table 1 – Cross-Architecture Transferability
            (ResNet-50, ConvNeXt-B, VisionMamba-Small, ViT-B/16 as source models)
  Table 2 – Effect of Optimization Steps T on ResNet-50 → MobileNet-V2 / ShuffleNet-V2

Pixel selection : Contrastive Counterfactual Generation (CCG)
Perturbation domain : High-Frequency (FFT high-pass projected), MI-PGD optimizer

Run on Kaggle with a GPU, ImageNet-Mini at:
  /kaggle/input/imagenetmini-1000/imagenet-mini/val
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import time
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import lpips as lpips_lib
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("WARNING: lpips not installed. LPIPS will be skipped. pip install lpips")

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("WARNING: timm not installed. pip install timm")


# ==========================================
# 1. DATASET
# ==========================================

class ImageNetMiniDataset(Dataset):
    def __init__(self, data_dir, transform=None, limit=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        if not self.data_dir.exists():
            kaggle_path = Path('/kaggle/input/imagenetmini-1000/imagenet-mini/val')
            if kaggle_path.exists():
                self.data_dir = kaggle_path
                print(f"Using Kaggle path: {self.data_dir}")
            else:
                raise FileNotFoundError(f"Directory not found: {data_dir}")

        print(f"Scanning directory: {self.data_dir}")
        image_extensions = ['*.JPEG', '*.jpg', '*.png', '*.PNG', '*.JPG']

        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                for ext in image_extensions:
                    for img_path in class_dir.glob(ext):
                        self.samples.append((img_path, class_dir.name))
                        if limit and len(self.samples) >= limit:
                            break
                    if limit and len(self.samples) >= limit:
                        break
            if limit and len(self.samples) >= limit:
                break

        self.classes = sorted(set([s[1] for s in self.samples]))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        print(f"Total: {len(self.samples)} images from {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[class_name]
        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)


# ==========================================
# 2. METRIC FUNCTIONS
# ==========================================

def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim_batch(img1, img2):
    if img1.dim() == 4:
        vals = []
        for i in range(img1.shape[0]):
            a = img1[i].detach().cpu().numpy().transpose(1, 2, 0)
            b = img2[i].detach().cpu().numpy().transpose(1, 2, 0)
            vals.append(ssim(a, b, multichannel=True, data_range=1.0, channel_axis=2))
        return float(np.mean(vals))
    a = img1.detach().cpu().numpy().transpose(1, 2, 0)
    b = img2.detach().cpu().numpy().transpose(1, 2, 0)
    return float(ssim(a, b, multichannel=True, data_range=1.0, channel_axis=2))


# ==========================================
# 3. CONTRASTIVE COUNTERFACTUAL GENERATION (CCG) PIXEL MASK
# ==========================================
#
# Replaces the old Integrated-Gradients pixel-importance mask.
# CCG selects the top-k most influential pixels using a single-step
# contrastive gradient: grad( logit[target_class] - logit[true_class] ).
# This is much cheaper than IG (no interpolation/integration loop) and is
# directly aligned with the adversarial objective (push toward target_class,
# away from true_class).

def get_ccg_mask(model, image, true_label, k, target_class=None):
    """
    Contrastive Counterfactual Generation (CCG) for pixel selection.

    Selects the k pixels most responsible for the gap between the model's
    logit on a contrastive (next most-likely) class and the true class.
    """
    B, C, H, W = image.shape

    if k == 0:
        return torch.zeros(1, 3, H, W, device=image.device)

    model.eval()
    image = image.clone().detach()

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            sorted_indices = torch.argsort(probs[0], descending=True)
            target_class = None
            for idx in sorted_indices:
                if idx.item() != true_label.item():
                    target_class = idx.item()
                    break
            if target_class is None:
                target_class = (true_label.item() + 1) % output.shape[1]

    image.requires_grad_(True)

    output = model(image)
    target_logit = output[0, target_class]
    true_logit = output[0, true_label.item()]

    contrastive_loss = target_logit - true_logit

    model.zero_grad()
    contrastive_loss.backward()

    grad = image.grad.detach()
    pixel_importance = torch.norm(grad[0], p=2, dim=0)

    flat_importance = pixel_importance.flatten()
    k_safe = min(k, flat_importance.numel())
    _, top_k_indices = torch.topk(flat_importance, k_safe)

    mask_flat = torch.zeros_like(flat_importance)
    mask_flat[top_k_indices] = 1.0

    mask = mask_flat.reshape(1, 1, H, W).repeat(1, 3, 1, 1)
    return mask


# ==========================================
# 4. HIGH-FREQUENCY DOMAIN HELPERS
# ==========================================
#
# Replaces the old pywt/DWT wavelet-domain injection. Instead, the
# perturbation is periodically re-projected into the high-frequency band of
# its 2D FFT spectrum (frequencies outside a low-frequency disc of radius
# `freq_radius` around the DC component are kept, everything inside is
# zeroed out). This concentrates the adversarial signal in high-frequency
# texture, which is both harder to perceive and known to transfer well
# across architectures.

def get_high_pass_filter_mask(shape, radius, device):
    """Creates a binary High-Pass Filter mask in the FFT-shifted domain."""
    rows, cols = shape[-2:]
    crow, ccol = rows // 2, cols // 2

    y, x = torch.meshgrid(
        torch.arange(rows, device=device),
        torch.arange(cols, device=device),
        indexing='ij'
    )

    dist = torch.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

    mask = torch.ones(shape, device=device)
    mask[:, :, dist <= radius] = 0
    return mask


def project_to_high_freq(delta, hp_mask):
    """Projects a perturbation tensor into the high-frequency domain."""
    fft_delta = torch.fft.fft2(delta, dim=(-2, -1))
    fft_delta_shifted = torch.fft.fftshift(fft_delta, dim=(-2, -1))
    fft_filtered = fft_delta_shifted * hp_mask
    fft_unshifted = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
    delta_filtered = torch.fft.ifft2(fft_unshifted, dim=(-2, -1)).real
    return delta_filtered


# ==========================================
# 5. HIGH-FREQUENCY MI-PGD ATTACK  (replaces WaveletMIPGD)
# ==========================================
#
# Same MI-PGD scaffolding as before (momentum, sign-gradient steps,
# periodic diversity noise, CCG/IG pixel-mask gating, eps-ball / [0,1]
# clamping) — only the domain the perturbation is periodically re-projected
# into has changed: high-frequency FFT band instead of wavelet detail bands.

class HighFreqMIPGD:
    def __init__(self, model, eps=8/255, alpha=2/255, steps=40,
                 decay=1.0, freq_radius=20, project_every=3):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.freq_radius = freq_radius
        self.project_every = project_every

    def generate_noise(self, shape, device):
        return torch.zeros(shape, device=device).uniform_(-self.eps, self.eps)

    def attack(self, images, labels, pixel_mask=None):
        device = images.device
        images = images.clone().detach()
        labels = labels.clone().detach()

        hp_mask = get_high_pass_filter_mask(images.shape, self.freq_radius, device)

        delta = self.generate_noise(images.shape, device)
        if pixel_mask is not None:
            delta = delta * pixel_mask
        delta = project_to_high_freq(delta, hp_mask)
        delta = torch.clamp(delta, -self.eps, self.eps)

        momentum = torch.zeros_like(delta)
        delta.requires_grad_(True)

        for step in range(self.steps):
            if step % 5 == 0:
                diversity_noise = torch.randn_like(delta) * 0.01
                adv_images = images + delta + diversity_noise
            else:
                adv_images = images + delta

            adv_images = torch.clamp(adv_images, 0, 1)
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            grad = delta.grad.detach()
            grad_norm = torch.norm(grad, p=1)
            momentum = self.decay * momentum + grad / (grad_norm + 1e-8)

            delta.data = delta.data + self.alpha * momentum.sign()

            if pixel_mask is not None:
                delta.data = delta.data * pixel_mask

            # Periodically re-project the perturbation into the
            # high-frequency band (replaces the old wavelet-noise injection)
            if step % self.project_every == 0:
                delta.data = project_to_high_freq(delta.data, hp_mask)

            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images
            delta.grad.zero_()

        return (images + delta.detach()).clamp(0, 1)


# ==========================================
# 6. MODEL LOADING
# ==========================================

def load_model(model_name, device):
    """Load a model by name. Returns (model, input_size)."""
    print(f"  Loading {model_name}...")
    input_size = 224

    if model_name == 'resnet50':
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.resnet50(pretrained=True)

    elif model_name == 'convnext_base':
        try:
            model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.convnext_base(pretrained=True)

    elif model_name == 'vit_b_16':
        try:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.vit_b_16(pretrained=True)

    elif model_name == 'visionmamba_small':
        # Vision Mamba Small via timm (vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2)
        # Falls back through a priority list of known timm model IDs
        if not HAS_TIMM:
            raise ImportError("timm is required for Vision Mamba. pip install timm")

        mamba_candidates = [
            'vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2',
            'vim_small_patch16_224',
            'vim_small_patch16_stride8_224',
            'vim_tiny_patch16_224',
        ]
        loaded = False
        for candidate in mamba_candidates:
            try:
                model = timm.create_model(candidate, pretrained=True)
                print(f"  Loaded Vision Mamba via timm: {candidate}")
                loaded = True
                break
            except Exception as e:
                print(f"  timm candidate {candidate} failed: {e}")

        if not loaded:
            # Last resort: ViT-Small as structural proxy
            print("  WARNING: Vision Mamba not available in timm. Using ViT-Small as proxy.")
            try:
                model = timm.create_model('vit_small_patch16_224', pretrained=True)
            except Exception:
                model = models.vit_b_16(pretrained=True)

    elif model_name == 'mobilenet_v2':
        try:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.mobilenet_v2(pretrained=True)

    elif model_name == 'shufflenet_v2':
        try:
            model = models.shufflenet_v2_x1_0(
                weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.shufflenet_v2_x1_0(pretrained=True)

    elif model_name == 'densenet121':
        try:
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.densenet121(pretrained=True)

    elif model_name == 'efficientnet_b0':
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    model.eval()
    return model, input_size


# ==========================================
# 7. ATTACK RUNNER
# ==========================================

def run_attack_on_dataset(
        source_model,
        dataset,
        k_pixels,
        steps,
        device,
        eps=8/255,
        alpha=2/255,
        freq_radius=20,
        verbose=False):
    """
    Generate adversarial examples for every image in dataset using source_model,
    using CCG for pixel selection and the High-Frequency MI-PGD attack for
    optimization. Returns lists of (clean_img_tensor, adv_img_tensor, true_label)
    and source white-box accuracy metrics.
    """
    source_model.eval()
    attack_obj = HighFreqMIPGD(
        source_model, eps=eps, alpha=alpha, steps=steps,
        decay=1.0, freq_radius=freq_radius, project_every=3
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False)

    clean_imgs, adv_imgs, labels_list = [], [], []
    correct_clean, correct_adv, total = 0, 0, 0

    ssim_vals, psnr_vals, lpips_vals = [], [], []

    lpips_model = None
    if HAS_LPIPS:
        try:
            lpips_model = lpips_lib.LPIPS(net='alex').to(device)
            lpips_model.eval()
        except Exception as e:
            print(f"  LPIPS load failed: {e}")

    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            out = source_model(images)
            pred = out.argmax(dim=1)
            correct_clean += (pred == labels).sum().item()

        # Build pixel mask via Contrastive Counterfactual Generation (CCG)
        mask = get_ccg_mask(source_model, images, labels, k_pixels).to(device)

        adv = attack_obj.attack(images, labels, mask)

        with torch.no_grad():
            adv_out = source_model(adv)
            adv_pred = adv_out.argmax(dim=1)
            correct_adv += (adv_pred == labels).sum().item()

        ssim_vals.append(calculate_ssim_batch(images, adv))
        psnr_vals.append(calculate_psnr(images, adv))

        if lpips_model is not None:
            with torch.no_grad():
                lv = lpips_model(images * 2 - 1, adv * 2 - 1).item()
                lpips_vals.append(lv)

        clean_imgs.append(images.cpu())
        adv_imgs.append(adv.cpu())
        labels_list.append(labels.cpu())
        total += 1

        if verbose and total % 20 == 0:
            print(f"    [{total}/{len(dataset)}] WB ASR so far: "
                  f"{100*(correct_clean-correct_adv)/max(correct_clean,1):.1f}%")

    clean_acc  = 100 * correct_clean / total if total > 0 else 0.0
    adv_acc    = 100 * correct_adv   / total if total > 0 else 0.0
    wb_asr     = 100 * (correct_clean - correct_adv) / correct_clean if correct_clean > 0 else 0.0
    avg_ssim   = float(np.mean(ssim_vals))  if ssim_vals  else 0.0
    avg_psnr   = float(np.mean(psnr_vals))  if psnr_vals  else 0.0
    avg_lpips  = float(np.mean(lpips_vals)) if lpips_vals else None

    return {
        'clean_imgs':  clean_imgs,
        'adv_imgs':    adv_imgs,
        'labels':      labels_list,
        'clean_acc':   clean_acc,
        'adv_acc':     adv_acc,
        'wb_asr':      wb_asr,
        'ssim':        avg_ssim,
        'psnr':        avg_psnr,
        'lpips':       avg_lpips,
    }


def evaluate_transfer(adv_imgs, labels_list, target_model, device):
    """Evaluate black-box transfer ASR: fraction of correctly-initially-classified
    images that are now misclassified after adversarial perturbation."""
    target_model.eval()
    correct_adv = 0
    total = 0

    with torch.no_grad():
        for adv, lbl in zip(adv_imgs, labels_list):
            adv = adv.to(device)
            lbl = lbl.to(device)
            out = target_model(adv)
            pred = out.argmax(dim=1)
            correct_adv += (pred == lbl).sum().item()
            total += lbl.shape[0]

    if total == 0:
        return 0.0

    adv_acc = 100 * correct_adv / total
    # Transfer ASR = images now misclassified / total
    return 100.0 - adv_acc


# ==========================================
# 8. TABLE 1 – CROSS-ARCHITECTURE TRANSFERABILITY
# ==========================================

def run_table1_transferability(dataset, device, eps, alpha, freq_radius,
                                k_values, source_models_names,
                                cnn_target_names, vit_target_name,
                                steps=40):
    """
    For each source model and each k value:
      - generate adversarial examples (white-box, CCG mask + High-Freq MI-PGD)
      - evaluate on each CNN target
      - evaluate on ViT target
      - record SSIM, LPIPS
    """
    print("\n" + "="*80)
    print("TABLE 1 – CROSS-ARCHITECTURE TRANSFERABILITY (CCG + High-Frequency Attack)")
    print("="*80)

    rows = []

    # Pre-load target models (they stay fixed)
    print("\nPre-loading target models...")
    cnn_targets = {}
    for name in cnn_target_names:
        cnn_targets[name], _ = load_model(name, device)

    vit_target = None
    if vit_target_name:
        vit_target, _ = load_model(vit_target_name, device)

    for src_name in source_models_names:
        print(f"\n{'='*60}")
        print(f"SOURCE MODEL: {src_name}")
        print(f"{'='*60}")

        try:
            src_model, _ = load_model(src_name, device)
        except Exception as e:
            print(f"  Could not load {src_name}: {e}")
            continue

        for k in k_values:
            print(f"\n  k = {k} pixels ...")
            attack_result = run_attack_on_dataset(
                src_model, dataset, k_pixels=k, steps=steps,
                device=device, eps=eps, alpha=alpha, freq_radius=freq_radius,
                verbose=True
            )

            wb_asr  = attack_result['wb_asr']
            avg_ssim = attack_result['ssim']
            avg_lpips = attack_result['lpips'] if attack_result['lpips'] is not None else float('nan')
            print(f"  WB ASR={wb_asr:.1f}%  SSIM={avg_ssim:.4f}  LPIPS={avg_lpips:.4f}")

            # CNN targets
            cnn_asrs = []
            for tgt_name, tgt_model in cnn_targets.items():
                if tgt_name == src_name:
                    continue  # skip self
                tgt_asr = evaluate_transfer(
                    attack_result['adv_imgs'], attack_result['labels'], tgt_model, device
                )
                print(f"    → {tgt_name} ASR: {tgt_asr:.2f}%")
                cnn_asrs.append(tgt_asr)

            avg_cnn_asr = float(np.mean(cnn_asrs)) if cnn_asrs else float('nan')

            # ViT target
            if vit_target is not None and src_name != vit_target_name:
                vit_asr = evaluate_transfer(
                    attack_result['adv_imgs'], attack_result['labels'], vit_target, device
                )
                print(f"    → {vit_target_name} ASR: {vit_asr:.2f}%")
            else:
                vit_asr = float('nan')

            rows.append({
                'Source':              src_name,
                'k':                   k,
                'Source ASR (%)':      round(wb_asr, 2),
                'Avg ASR → CNNs (%)':  round(avg_cnn_asr, 2),
                f'ASR → ViT (%)':      round(vit_asr, 2) if not np.isnan(vit_asr) else '--',
                'SSIM':                round(avg_ssim, 4),
                'LPIPS':               round(avg_lpips, 4) if not np.isnan(avg_lpips) else '--',
            })

        # Free source model memory
        del src_model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    return df


# ==========================================
# 9. TABLE 2 – EFFECT OF STEPS T  (ResNet-50 → MobileNet-V2, ShuffleNet-V2)
# ==========================================

def run_table2_steps_effect(dataset, device, eps, alpha, freq_radius,
                             k_values, steps_list,
                             source_model_name='resnet50',
                             target_names=None):
    """
    For ResNet-50 as source, sweep T (optimization steps) and k values.
    Targets: MobileNet-V2 and ShuffleNet-V2.
    """
    if target_names is None:
        target_names = ['mobilenet_v2', 'shufflenet_v2']

    print("\n" + "="*80)
    print("TABLE 2 – EFFECT OF OPTIMIZATION STEPS T  (Source: ResNet-50, CCG + High-Freq)")
    print("="*80)

    print("\nLoading source model (ResNet-50)...")
    src_model, _ = load_model(source_model_name, device)

    print("Loading target models...")
    target_models = {}
    for name in target_names:
        target_models[name], _ = load_model(name, device)

    rows = []

    for T in steps_list:
        for k in k_values:
            print(f"\n  T={T}, k={k} ...")
            attack_result = run_attack_on_dataset(
                src_model, dataset, k_pixels=k, steps=T,
                device=device, eps=eps, alpha=alpha, freq_radius=freq_radius,
                verbose=False
            )

            wb_asr   = attack_result['wb_asr']
            avg_ssim = attack_result['ssim']
            avg_psnr = attack_result['psnr']
            avg_lpips = attack_result['lpips'] if attack_result['lpips'] is not None else float('nan')

            row = {
                'T':             T,
                'k':             k,
                'Source ASR (%)': round(wb_asr, 2),
                'PSNR ↑':        round(avg_psnr, 2),
                'SSIM ↑':        round(avg_ssim, 4),
                'LPIPS ↓':       round(avg_lpips, 4) if not np.isnan(avg_lpips) else '--',
            }

            for tgt_name, tgt_model in target_models.items():
                tgt_asr = evaluate_transfer(
                    attack_result['adv_imgs'], attack_result['labels'],
                    tgt_model, device
                )
                print(f"    → {tgt_name} ASR: {tgt_asr:.2f}%")
                short = tgt_name.replace('mobilenet_v2', 'Mob-V2').replace('shufflenet_v2', 'Shuffle-V2')
                row[f'{short} (%)'] = round(tgt_asr, 2)

            print(f"  WB ASR={wb_asr:.1f}%  SSIM={avg_ssim:.4f}  "
                  f"PSNR={avg_psnr:.2f}  LPIPS={avg_lpips:.4f}")
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ==========================================
# 10. TRANSFER BAR PLOT (Figure)
# ==========================================

def plot_transfer_bar(df_table1, source_name='visionmamba_small',
                      output_path='transfer_bar.png'):
    """
    Average CNN transfer ASR vs K for one source model.
    Mirrors Figure in the paper (originally EfficientNet-B0, here VisionMamba-Small).
    """
    src_df = df_table1[df_table1['Source'] == source_name].copy()
    if src_df.empty:
        print("No data for source model in Table 1. Skipping bar plot.")
        return

    k_vals  = src_df['k'].tolist()
    asr_cnn = src_df['Avg ASR → CNNs (%)'].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(k) for k in k_vals], asr_cnn,
                  color=['#2196F3', '#4CAF50', '#FF5722'], edgecolor='black', width=0.5)

    for bar, val in zip(bars, asr_cnn):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_title(f'Average Transfer ASR (→ CNNs) vs K\nSource: {source_name} (CCG + High-Freq)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Perturbed Pixels K', fontsize=12)
    ax.set_ylabel('Average Transfer ASR (%)', fontsize=12)
    ax.set_ylim(0, max(asr_cnn) * 1.25 + 2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.close()


# ==========================================
# 11. PRETTY PRINT & SAVE TABLES
# ==========================================

def print_and_save(df, title, csv_path):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")


# ==========================================
# 12. MAIN
# ==========================================

def main():
    overall_start = time.time()

    # ── Configuration ─────────────────────────────────────────────────────────
    DATA_DIR    = '/kaggle/input/imagenetmini-1000/imagenet-mini/val'
    NUM_SAMPLES = 100          # images to evaluate (increase for more reliable estimates)
    EPS         = 8  / 255
    ALPHA       = 2  / 255
    FREQ_RADIUS = 20           # High-pass filter radius (larger = fewer freqs allowed = higher-freq-only)

    # Table 1 – Cross-Architecture Transferability
    TABLE1_K_VALUES       = [100, 1000, 5000]
    TABLE1_SOURCE_MODELS  = ['resnet50', 'convnext_base', 'visionmamba_small', 'vit_b_16']
    TABLE1_CNN_TARGETS    = ['resnet50', 'mobilenet_v2', 'shufflenet_v2', 'densenet121']
    TABLE1_VIT_TARGET     = 'vit_b_16'
    TABLE1_STEPS          = 40

    # Table 2 – Steps sweep  (ResNet-50 → MobileNet-V2, ShuffleNet-V2)
    TABLE2_STEPS_LIST     = [10, 25, 50, 100, 200]
    TABLE2_K_VALUES       = [100, 1000, 5000]
    TABLE2_SOURCE         = 'resnet50'
    TABLE2_TARGETS        = ['mobilenet_v2', 'shufflenet_v2']

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return

    print(f"\nLoading dataset ({NUM_SAMPLES} samples)...")
    dataset = ImageNetMiniDataset(DATA_DIR, transform=transform, limit=NUM_SAMPLES)
    if len(dataset) == 0:
        print("ERROR: No images found. Check DATA_DIR.")
        return

    # ── TABLE 1 ───────────────────────────────────────────────────────────────
    t1_start = time.time()
    df_table1 = run_table1_transferability(
        dataset      = dataset,
        device       = device,
        eps          = EPS,
        alpha        = ALPHA,
        freq_radius  = FREQ_RADIUS,
        k_values     = TABLE1_K_VALUES,
        source_models_names = TABLE1_SOURCE_MODELS,
        cnn_target_names    = TABLE1_CNN_TARGETS,
        vit_target_name     = TABLE1_VIT_TARGET,
        steps        = TABLE1_STEPS,
    )
    t1_elapsed = time.time() - t1_start
    print_and_save(df_table1,
                   "TABLE 1 – Cross-Architecture Transferability (CCG + High-Frequency)",
                   "table1_transferability.csv")

    # ── TABLE 2 ───────────────────────────────────────────────────────────────
    t2_start = time.time()
    df_table2 = run_table2_steps_effect(
        dataset      = dataset,
        device       = device,
        eps          = EPS,
        alpha        = ALPHA,
        freq_radius  = FREQ_RADIUS,
        k_values     = TABLE2_K_VALUES,
        steps_list   = TABLE2_STEPS_LIST,
        source_model_name = TABLE2_SOURCE,
        target_names = TABLE2_TARGETS,
    )
    t2_elapsed = time.time() - t2_start
    print_and_save(df_table2,
                   "TABLE 2 – Effect of Optimization Steps T (Source: ResNet-50, CCG + High-Frequency)",
                   "table2_steps_effect.csv")

    # ── Figure – Transfer Bar Plot ─────────────────────────────────────────────
    print("\nGenerating transfer bar plot (VisionMamba-Small)...")
    plot_transfer_bar(df_table1, source_name='visionmamba_small',
                      output_path='transfer_bar_visionmamba.png')

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - overall_start
    print("\n" + "="*80)
    print("DONE")
    print(f"  Table 1 time: {timedelta(seconds=int(t1_elapsed))}")
    print(f"  Table 2 time: {timedelta(seconds=int(t2_elapsed))}")
    print(f"  Total time:   {timedelta(seconds=int(total_time))}")
    print("="*80)
    print("\nOutput files:")
    print("  table1_transferability.csv")
    print("  table2_steps_effect.csv")
    print("  transfer_bar_visionmamba.png")


if __name__ == "__main__":
    main()