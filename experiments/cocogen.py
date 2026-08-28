import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
import torch.fft
import pandas as pd
import time
from datetime import timedelta
import lpips


# ==========================================
# 1. UTILITY CLASSES (Dataset, Metrics)
# ==========================================

class ImageNetMiniDataset(Dataset):
    """Dataset loader for ImageNet Mini"""

    def __init__(self, data_dir, transform=None, limit=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        if not self.data_dir.exists():
            kaggle_path = Path('/kaggle/input/imagenetmini-1000/imagenet-mini/val')
            if kaggle_path.exists():
                self.data_dir = kaggle_path
            else:
                raise FileNotFoundError(f"Directory not found: {data_dir}")

        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.JPEG'):
                    self.samples.append((img_path, class_dir.name))
                    if limit and len(self.samples) >= limit:
                        break
                if limit and len(self.samples) >= limit:
                    break

        self.classes = sorted(set([s[1] for s in self.samples]))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)


class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 for FID calculation"""

    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True, aux_logits=True)
        inception.eval()

        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if x.shape[2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# ==========================================
# NEW: Additional Metric Functions
# ==========================================

def calculate_linf(img1, img2):
    """Calculate L-infinity distance"""
    return torch.max(torch.abs(img1 - img2)).item()


def calculate_l2(img1, img2):
    """Calculate L2 distance"""
    return torch.norm(img1 - img2, p=2).item()


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1, img2):
    if img1.dim() == 4:
        ssim_values = []
        for i in range(img1.shape[0]):
            img1_np = img1[i].detach().cpu().numpy().transpose(1, 2, 0)
            img2_np = img2[i].detach().cpu().numpy().transpose(1, 2, 0)
            ssim_val = ssim(img1_np, img2_np, multichannel=True, data_range=1.0, channel_axis=2)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        img1_np = img1.detach().cpu().numpy().transpose(1, 2, 0)
        img2_np = img2.detach().cpu().numpy().transpose(1, 2, 0)
        return ssim(img1_np, img2_np, multichannel=True, data_range=1.0, channel_axis=2)


def calculate_activation_statistics(images, model, device, batch_size=32):
    model.eval()
    features_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            features = model(batch)
            features_list.append(features.cpu().numpy())
    features = np.concatenate(features_list, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid


# ==========================================
# 2. CONTRASTIVE COUNTERFACTUAL GENERATION
# ==========================================

def get_ccg_mask(model, image, true_label, k, image_size=(224, 224), target_class=None):
    """
    Contrastive Counterfactual Generation (CCG) for pixel selection.
    """
    if k == 0:
        return torch.zeros(1, 3, *image_size, device=image.device)

    model.eval()
    image = image.clone().detach()

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            sorted_indices = torch.argsort(probs[0], descending=True)
            for idx in sorted_indices:
                if idx.item() != true_label.item():
                    target_class = idx.item()
                    break

    B, C, H, W = image.shape
    importance_map = torch.zeros(H, W, device=image.device)

    eps = 0.1

    image.requires_grad = True

    output = model(image)

    target_logit = output[0, target_class]
    true_logit = output[0, true_label]

    contrastive_loss = target_logit - true_logit

    model.zero_grad()
    contrastive_loss.backward()

    grad = image.grad.detach()

    pixel_importance = torch.norm(grad[0], p=2, dim=0)

    flat_importance = pixel_importance.flatten()
    k_safe = min(k, flat_importance.numel())
    _, top_k_indices = torch.topk(flat_importance, k_safe)

    mask_flat = torch.zeros_like(flat_importance)
    mask_flat[top_k_indices] = 1

    mask = mask_flat.reshape(1, 1, H, W)
    mask = mask.repeat(1, 3, 1, 1)

    return mask


# ==========================================
# 3. HIGH FREQUENCY HELPERS
# ==========================================

def get_high_pass_filter_mask(shape, radius, device):
    """Creates a High Pass Filter mask."""
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


# ==========================================
# 4. HIGH FREQUENCY PGD ATTACK WITH CCG
# ==========================================

class HighFreqSpatialPGD:
    """PGD Attack with spatial and frequency constraints"""

    def __init__(self, model, eps=0.03, alpha=0.01, steps=40, freq_radius=20):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.freq_radius = freq_radius

    def project_to_high_freq(self, delta, hp_mask):
        """Projects the perturbation into high-frequency domain"""
        fft_delta = torch.fft.fft2(delta, dim=(-2, -1))
        fft_delta_shifted = torch.fft.fftshift(fft_delta, dim=(-2, -1))
        fft_filtered = fft_delta_shifted * hp_mask
        fft_unshifted = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        delta_filtered = torch.fft.ifft2(fft_unshifted, dim=(-2, -1)).real
        return delta_filtered

    def attack(self, images, labels, pixel_mask=None):
        images = images.clone().detach().to(images.device)
        labels = labels.clone().detach().to(images.device)

        hp_mask = get_high_pass_filter_mask(images.shape, self.freq_radius, images.device)
        delta = torch.zeros_like(images).uniform_(-self.eps, self.eps)

        if pixel_mask is not None:
            delta = delta * pixel_mask
        delta = self.project_to_high_freq(delta, hp_mask)

        delta.requires_grad = True

        for step in range(self.steps):
            outputs = self.model(images + delta)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            grad = delta.grad.detach()
            delta.data = delta.data + self.alpha * grad.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)

            if pixel_mask is not None:
                delta.data = delta.data * pixel_mask

            delta.data = self.project_to_high_freq(delta.data, hp_mask)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images

            delta.grad.zero_()

        return images + delta.detach()


# ==========================================
# 5. EVALUATION & GRID SEARCH WITH TIMING
# ==========================================

def evaluate_attack(model, dataloader, attack_obj, k_pixels, device,
                    inception_model=None, lpips_model=None):
    """Evaluate attack with per-image timing and comprehensive metrics"""

    model.eval()
    correct_clean = 0
    correct_adv = 0
    total = 0

    ssim_values = []
    linf_values = []
    l2_values = []
    psnr_values = []
    lpips_values = []

    clean_images_list = []
    adv_images_list = []
    image_times = []

    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            correct_clean += (clean_preds == labels).sum().item()

        for i in range(images.size(0)):
            img_start = time.time()

            img = images[i:i + 1]
            label = labels[i:i + 1]

            pixel_mask = get_ccg_mask(
                model,
                img,
                label,
                k_pixels
            ).to(device)

            adv_img = attack_obj.attack(
                img,
                label,
                pixel_mask
            )

            with torch.no_grad():
                adv_output = model(adv_img)
                adv_pred = adv_output.argmax(dim=1)
                correct_adv += (adv_pred == label).sum().item()

            # Metrics
            ssim_val = calculate_ssim(img, adv_img)
            linf_val = calculate_linf(img, adv_img)
            l2_val = calculate_l2(img, adv_img)
            psnr_val = calculate_psnr(img, adv_img)

            ssim_values.append(ssim_val)
            linf_values.append(linf_val)
            l2_values.append(l2_val)
            psnr_values.append(psnr_val)

            # LPIPS
            if lpips_model is not None:
                with torch.no_grad():
                    img_normalized = img * 2 - 1
                    adv_normalized = adv_img * 2 - 1

                    lpips_val = lpips_model(
                        img_normalized,
                        adv_normalized
                    ).item()

                    lpips_values.append(lpips_val)

            clean_images_list.append(img.cpu())
            adv_images_list.append(adv_img.cpu())

            img_time = time.time() - img_start
            image_times.append(img_time)

            total += 1

            if total % 10 == 0:
                print(
                    f"  Processed {total} images | "
                    f"Avg time per image: "
                    f"{np.mean(image_times):.3f}s"
                )

    clean_acc = 100 * correct_clean / total
    adv_acc = 100 * correct_adv / total

    attack_success_rate = (
        100 * (correct_clean - correct_adv) / correct_clean
        if correct_clean > 0 else 0
    )

    avg_ssim = np.mean(ssim_values)
    avg_linf = np.mean(linf_values)
    avg_l2 = np.mean(l2_values)
    avg_psnr = np.mean(psnr_values)

    avg_lpips = (
        np.mean(lpips_values)
        if lpips_values else None
    )

    avg_img_time = np.mean(image_times)

    # FID
    fid_score = None

    if inception_model is not None and len(clean_images_list) > 1:
        clean_images = torch.cat(
            clean_images_list,
            dim=0
        )

        adv_images = torch.cat(
            adv_images_list,
            dim=0
        )

        try:
            mu1, sigma1 = calculate_activation_statistics(
                clean_images,
                inception_model,
                device
            )

            mu2, sigma2 = calculate_activation_statistics(
                adv_images,
                inception_model,
                device
            )

            fid_score = calculate_fid(
                mu1,
                sigma1,
                mu2,
                sigma2
            )

        except Exception as e:
            print(f"Warning: FID calculation failed: {e}")
            fid_score = 0.0

    return (
        clean_acc,
        adv_acc,
        attack_success_rate,
        avg_ssim,
        fid_score,
        avg_linf,
        avg_l2,
        avg_psnr,
        avg_lpips,
        avg_img_time
    )


def adaptive_grid_search_optimal_k(
        model,
        model_name,
        dataloader,
        device,
        eps,
        alpha,
        steps,
        ssim_threshold=0.95,
        asr_threshold=100.0,
        fid_threshold=20.0
):
    model_start_time = time.time()

    attack_obj = HighFreqSpatialPGD(
        model,
        eps=eps,
        alpha=alpha,
        steps=steps,
        freq_radius=25
    )

    print("Loading InceptionV3 for FID calculation...")

    inception_model = InceptionV3FeatureExtractor().to(device)
    inception_model.eval()

    print("Loading LPIPS model...")

    lpips_model = lpips.LPIPS(
        net='alex'
    ).to(device)

    lpips_model.eval()

    first_batch = next(iter(dataloader))
    max_pixels = (
            first_batch[0].shape[2] *
            first_batch[0].shape[3]
    )

    print(f"\nRunning Adaptive Search for {model_name}...")
    print(
        "Method: CCG (Contrastive Counterfactual Generation) "
        "+ High-Frequency Constraint"
    )

    results = []

    k = 0
    step = 50

    best_k = None
    iteration = 0

    while k <= max_pixels:

        iteration += 1
        k_start_time = time.time()

        print(f"\n--- Testing k={k} pixels ---")

        (
            clean_acc,
            adv_acc,
            asr,
            avg_ssim,
            fid,
            avg_linf,
            avg_l2,
            avg_psnr,
            avg_lpips,
            avg_img_time
        ) = evaluate_attack(
            model,
            dataloader,
            attack_obj,
            k,
            device,
            inception_model,
            lpips_model
        )

        k_elapsed = time.time() - k_start_time

        result = {
            'k': k,
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'attack_success_rate': asr,
            'ssim': avg_ssim,
            'fid': fid if fid is not None else 0.0,
            'linf': avg_linf,
            'l2': avg_l2,
            'psnr': avg_psnr,
            'lpips': (
                avg_lpips
                if avg_lpips is not None else 0.0
            ),
            'avg_time_per_image': avg_img_time,
            'total_time_for_k': k_elapsed
        }

        results.append(result)

        print(
            f"Clean Acc: {clean_acc:.2f}% | "
            f"ASR: {asr:.2f}% | "
            f"Adv Acc: {adv_acc:.2f}%"
        )

        print(
            f"SSIM: {avg_ssim:.4f} | "
            f"FID: {fid if fid else 0:.4f} | "
            f"LPIPS: {avg_lpips if avg_lpips else 0:.4f}"
        )

        print(
            f"L∞: {avg_linf:.6f} | "
            f"L2: {avg_l2:.4f} | "
            f"PSNR: {avg_psnr:.2f} dB"
        )

        print(
            f"Time: {k_elapsed:.2f}s total | "
            f"{avg_img_time:.3f}s per image"
        )

        ssim_ok = avg_ssim > ssim_threshold
        asr_ok = asr >= asr_threshold
        fid_ok = (
                fid is not None and
                fid < fid_threshold
        )

        if ssim_ok and asr_ok and fid_ok:
            best_k = k

            print(
                f"\n>>> STOPPING CRITERIA MET at k={k}"
            )

            break

        if k == 0:
            k = 10

        else:
            if len(results) >= 2:

                diff = (
                        asr -
                        results[-2]['attack_success_rate']
                )

                if diff > 10:
                    step = 20

                elif diff < 1:
                    step = min(
                        step * 2,
                        1000
                    )

            k += step

        if iteration > 40:
            break

    model_total_time = time.time() - model_start_time

    return (
        results,
        best_k,
        model_total_time
    )


# ==========================================
# MODEL LOADING
# ==========================================

def load_model(model_name, device):
    print(f"\n{'=' * 70}")
    print(f"Loading {model_name}...")
    print(f"{'=' * 70}")

    if model_name == 'resnet50':
        model = models.resnet50(
            pretrained=True
        )

    elif model_name == 'convnext_base':
        model = models.convnext_base(
            pretrained=True
        )

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(
            pretrained=True
        )

    elif model_name == 'vit_b16':
        model = models.vit_b_16(
            pretrained=True
        )

    else:
        raise ValueError(
            f"Model '{model_name}' is not supported."
        )

    model = model.to(device)
    model.eval()

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"✓ {model_name} loaded successfully")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model

def visualize_results(results, model_name):
    k_vals = [r['k'] for r in results]
    asr = [r['attack_success_rate'] for r in results]
    ssim = [r['ssim'] for r in results]
    lpips_vals = [r['lpips'] for r in results]
    psnr_vals = [r['psnr'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(k_vals, asr, 'r-o')
    axes[0, 0].set_title(f'{model_name}: Attack Success Rate vs K (CCG)')
    axes[0, 0].set_xlabel('K Pixels')
    axes[0, 0].set_ylabel('ASR (%)')
    axes[0, 0].grid(True)

    axes[0, 1].plot(k_vals, ssim, 'b-s')
    axes[0, 1].set_title(f'{model_name}: SSIM vs K (CCG)')
    axes[0, 1].set_xlabel('K Pixels')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].grid(True)

    axes[1, 0].plot(k_vals, lpips_vals, 'g-^')
    axes[1, 0].set_title(f'{model_name}: LPIPS vs K (CCG)')
    axes[1, 0].set_xlabel('K Pixels')
    axes[1, 0].set_ylabel('LPIPS')
    axes[1, 0].grid(True)

    axes[1, 1].plot(k_vals, psnr_vals, 'm-d')
    axes[1, 1].set_title(f'{model_name}: PSNR vs K (CCG)')
    axes[1, 1].set_xlabel('K Pixels')
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f'results_{model_name}_ccg_comprehensive.png')
    print(f"Plot saved to results_{model_name}_ccg_comprehensive.png")


# ==========================================
# 6. MAIN EXECUTION WITH COMPREHENSIVE TIMING
# ==========================================

def main():
    overall_start_time = time.time()

    DATA_DIR = '/kaggle/input/imagenetmini-1000/imagenet-mini/val'

    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found. Please create it or adjust path.")
        return

    BATCH_SIZE = 1
    NUM_SAMPLES = 50

    EPS = 8 / 255
    ALPHA = 2 / 255
    STEPS = 40

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset = ImageNetMiniDataset(DATA_DIR, transform=transform, limit=NUM_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    target_models = [
        'resnet50',
        'convnext_base',
        'efficientnet_b0',
        'vit_b16'
    ]

    all_results = {}
    model_times = {}

    for model_name in target_models:
        try:
            model = load_model(model_name, device)

            results, best_k, model_time = adaptive_grid_search_optimal_k(
                model, model_name, dataloader, device,
                EPS, ALPHA, STEPS
            )

            all_results[model_name] = results
            model_times[model_name] = model_time
            visualize_results(results, model_name)

            print(f"\n{'=' * 60}")
            print(f"TOTAL TIME FOR {model_name.upper()}: {timedelta(seconds=int(model_time))}")
            print(f"{'=' * 60}")

        except Exception as e:
            print(f"Error executing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    overall_time = time.time() - overall_start_time

    print("\n" + "=" * 150)
    print("FINAL COMPARISON: CCG + High-Frequency Attack WITH COMPREHENSIVE METRICS")
    print("=" * 150)

    summary_data = []

    for model_name, res_list in all_results.items():
        for r in res_list:
            summary_data.append({
                "Model": model_name,
                "K": r['k'],
                "Clean%": f"{r['clean_accuracy']:.2f}",
                "Adv%": f"{r['adversarial_accuracy']:.2f}",
                "ASR%": f"{r['attack_success_rate']:.2f}",
                "SSIM": f"{r['ssim']:.4f}",
                "FID": f"{r['fid']:.4f}",
                "LPIPS": f"{r['lpips']:.4f}",
                "L∞": f"{r['linf']:.6f}",
                "L2": f"{r['l2']:.4f}",
                "PSNR": f"{r['psnr']:.2f}",
                "Time/Img": f"{r['avg_time_per_image']:.3f}s"
            })

    if summary_data:
        df = pd.DataFrame(summary_data)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')

        print(df.to_string(index=False))

        print("\n" + "-" * 150)
        print("MODEL TIMING SUMMARY")
        print("-" * 150)

        for model_name, res in all_results.items():
            best_r = max(res, key=lambda x: x['attack_success_rate'])
            model_time = model_times[model_name]
            print(f"\n{model_name.upper()}:")
            print(
                f"  Best Result: k={best_r['k']} | ASR: {best_r['attack_success_rate']:.2f}% | SSIM: {best_r['ssim']:.4f}")
            print(
                f"  Metrics: LPIPS: {best_r['lpips']:.4f} | L∞: {best_r['linf']:.6f} | L2: {best_r['l2']:.4f} | PSNR: {best_r['psnr']:.2f} dB")
            print(f"  Total Model Time: {timedelta(seconds=int(model_time))} ({model_time:.2f}s)")
            print(f"  Average Time per K-value: {model_time / len(res):.2f}s")

        # Export results to CSV
        df.to_csv('attack_results_comprehensive.csv', index=False)
        print(f"\n✓ Results exported to 'attack_results_comprehensive.csv'")

    print("\n" + "=" * 150)
    print(f"TOTAL EXECUTION TIME FOR ALL MODELS: {timedelta(seconds=int(overall_time))} ({overall_time:.2f}s)")
    print("=" * 150)

    # Summary statistics across all models
    if all_results:
        print("\n" + "=" * 150)
        print("AGGREGATE STATISTICS ACROSS ALL MODELS")
        print("=" * 150)

        all_asrs = []
        all_ssims = []
        all_lpips = []
        all_psnrs = []

        for model_name, res_list in all_results.items():
            for r in res_list:
                all_asrs.append(r['attack_success_rate'])
                all_ssims.append(r['ssim'])
                all_lpips.append(r['lpips'])
                all_psnrs.append(r['psnr'])

        print(f"\nAttack Success Rate (ASR):")
        print(f"  Mean: {np.mean(all_asrs):.2f}% | Median: {np.median(all_asrs):.2f}%")
        print(f"  Min: {np.min(all_asrs):.2f}% | Max: {np.max(all_asrs):.2f}%")

        print(f"\nImage Quality Metrics:")
        print(f"  SSIM  - Mean: {np.mean(all_ssims):.4f} | Median: {np.median(all_ssims):.4f}")
        print(f"  LPIPS - Mean: {np.mean(all_lpips):.4f} | Median: {np.median(all_lpips):.4f}")
        print(f"  PSNR  - Mean: {np.mean(all_psnrs):.2f} dB | Median: {np.median(all_psnrs):.2f} dB")

        # Find best configuration across all models
        print(f"\n" + "-" * 150)
        print("BEST CONFIGURATIONS PER MODEL")
        print("-" * 150)

        for model_name, res_list in all_results.items():
            # Best by ASR with quality constraints
            quality_results = [r for r in res_list if r['ssim'] > 0.90]
            if quality_results:
                best_quality = max(quality_results, key=lambda x: x['attack_success_rate'])
                print(f"\n{model_name.upper()} - Best with SSIM>0.90:")
                print(f"  k={best_quality['k']}, ASR={best_quality['attack_success_rate']:.2f}%, "
                      f"SSIM={best_quality['ssim']:.4f}, LPIPS={best_quality['lpips']:.4f}, "
                      f"PSNR={best_quality['psnr']:.2f} dB")

            # Highest quality attack (highest SSIM with ASR>50%)
            effective_attacks = [r for r in res_list if r['attack_success_rate'] > 50]
            if effective_attacks:
                highest_quality = max(effective_attacks, key=lambda x: x['ssim'])
                print(f"\n{model_name.upper()} - Highest Quality with ASR>50%:")
                print(f"  k={highest_quality['k']}, ASR={highest_quality['attack_success_rate']:.2f}%, "
                      f"SSIM={highest_quality['ssim']:.4f}, LPIPS={highest_quality['lpips']:.4f}, "
                      f"PSNR={highest_quality['psnr']:.2f} dB")

    print("\n" + "=" * 150)
    print("EXECUTION COMPLETE!")
    print("=" * 150)
    print(f"\nGenerated files:")
    print(f"  - attack_results_comprehensive.csv")
    for model_name in all_results.keys():
        print(f"  - results_{model_name}_ccg_comprehensive.png")
    print()


if __name__ == "__main__":
    main()