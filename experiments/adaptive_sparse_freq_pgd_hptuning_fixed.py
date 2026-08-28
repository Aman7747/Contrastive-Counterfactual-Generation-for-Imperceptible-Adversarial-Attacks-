"""
Enhanced Adaptive Sparse-Frequency PGD Attack
==============================================
Hyperparameter Tuning: Pixel Selection Score Weights
  S(i,j) = norm(G(i,j)) * w_grad  +  norm(w(i,j)) * w_invis

ROOT CAUSE FIX
--------------
Previously grad_mag (raw scale ~0–10) was combined with invis (in [0,1])
without normalisation, making w_invis effectively irrelevant.
Both maps are now min-max normalised to [0,1] before weighting.

Model: ResNet50 only.
Metrics: ASR, SSIM, PSNR, LPIPS, FID

Install:
    pip install lpips
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
import matplotlib.pyplot as plt
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
import torch.fft
import pandas as pd
import time
from datetime import timedelta

try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("WARNING: pip install lpips")

WEIGHT_COMBINATIONS = [
    (1.0, 0.0),
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),   # original baseline
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
]


# ── Dataset ───────────────────────────────────────────────────────────────────

class ImageNetMiniDataset(Dataset):
    def __init__(self, data_dir, transform=None, limit=None):
        self.data_dir  = Path(data_dir)
        self.transform = transform
        self.samples   = []
        if not self.data_dir.exists():
            kaggle_path = Path('/kaggle/input/imagenetmini-1000/imagenet-mini/val')
            if kaggle_path.exists():
                self.data_dir = kaggle_path
            else:
                raise FileNotFoundError(f"Not found: {data_dir}")
        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.JPEG'):
                    self.samples.append((img_path, class_dir.name))
                    if limit and len(self.samples) >= limit:
                        break
                if limit and len(self.samples) >= limit:
                    break
        self.classes      = sorted(set(s[1] for s in self.samples))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[class_name]
        if self.transform: image = self.transform(image)
        return image, label, str(img_path)


# ── InceptionV3 for FID ───────────────────────────────────────────────────────

class InceptionV3FeatureExtractor(nn.Module):
    _LAYERS = ['Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3','maxpool1',
               'Conv2d_3b_1x1','Conv2d_4a_3x3','maxpool2',
               'Mixed_5b','Mixed_5c','Mixed_5d',
               'Mixed_6a','Mixed_6b','Mixed_6c','Mixed_6d','Mixed_6e',
               'Mixed_7a','Mixed_7b','Mixed_7c']
    def __init__(self):
        super().__init__()
        try:
            inception = models.inception_v3(pretrained=True, aux_logits=True)
        except TypeError:
            inception = models.inception_v3(weights='IMAGENET1K_V1', aux_logits=True)
        inception.eval()
        for name in self._LAYERS:
            setattr(self, name, getattr(inception, name))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        if x.shape[2:] != (299, 299):
            x = F.interpolate(x, size=(299,299), mode='bilinear', align_corners=False)
        for name in self._LAYERS:
            x = getattr(self, name)(x)
        return torch.flatten(self.avgpool(x), 1)


# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_ssim(img1, img2):
    def _one(a, b):
        return ssim(a.detach().cpu().numpy().transpose(1,2,0),
                    b.detach().cpu().numpy().transpose(1,2,0),
                    multichannel=True, data_range=1.0, channel_axis=2)
    if img1.dim() == 4:
        return float(np.mean([_one(img1[i], img2[i]) for i in range(img1.shape[0])]))
    return _one(img1, img2)


def calculate_psnr(img1, img2, max_val=1.0):
    if img1.dim() == 3: img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
    mse_batch = ((img1.detach().cpu().float()-img2.detach().cpu().float())**2).mean(dim=[1,2,3])
    vals = [100.0 if m.item()<1e-10 else 10.0*torch.log10(torch.tensor(max_val**2)/m).item()
            for m in mse_batch]
    return float(np.mean(vals))


class LPIPSMetric:
    def __init__(self, net='alex', device='cpu'):
        if LPIPS_AVAILABLE:
            self.fn = lpips_lib.LPIPS(net=net).to(device); self.fn.eval()
            self.device = device; self.ready = True
        else:
            self.ready = False
    def __call__(self, img1, img2):
        if not self.ready: return 0.0
        with torch.no_grad():
            return float(self.fn((img1.to(self.device)*2-1).clamp(-1,1),
                                  (img2.to(self.device)*2-1).clamp(-1,1)).mean().item())


def calculate_activation_statistics(images, model, device, batch_size=32):
    model.eval(); feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            feats.append(model(images[i:i+batch_size].to(device)).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


def calculate_fid(mu1, s1, mu2, s2, eps=1e-6):
    mu1,mu2 = np.atleast_1d(mu1),np.atleast_1d(mu2)
    s1,s2   = np.atleast_2d(s1), np.atleast_2d(s2)
    diff    = mu1-mu2
    covmean,_ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((s1+np.eye(s1.shape[0])*eps).dot(s2+np.eye(s1.shape[0])*eps))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return diff.dot(diff)+np.trace(s1)+np.trace(s2)-2*np.trace(covmean)


# ── Differentiable losses ─────────────────────────────────────────────────────

def _gaussian_window(size=11, sigma=1.5, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device)-size//2
    g = torch.exp(-(coords**2)/(2*sigma**2)); g = g/g.sum()
    return (g.unsqueeze(0)*g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

def ssim_loss_differentiable(img1, img2, ws=11, sigma=1.5, C1=1e-4, C2=9e-4):
    device=img1.device; win=_gaussian_window(ws,sigma,device).expand(3,1,ws,ws); pad=ws//2
    mu1=F.conv2d(img1,win,padding=pad,groups=3); mu2=F.conv2d(img2,win,padding=pad,groups=3)
    mu1s,mu2s,mu12=mu1**2,mu2**2,mu1*mu2
    s1=F.conv2d(img1*img1,win,padding=pad,groups=3)-mu1s
    s2=F.conv2d(img2*img2,win,padding=pad,groups=3)-mu2s
    s12=F.conv2d(img1*img2,win,padding=pad,groups=3)-mu12
    return 1.0-((2*mu12+C1)*(2*s12+C2)/((mu1s+mu2s+C1)*(s1+s2+C2))).mean()

def lpips_loss_differentiable(img1, img2, lpips_fn):
    if lpips_fn is None or not lpips_fn.ready:
        return torch.tensor(0.0, device=img1.device)
    return lpips_fn.fn((img1*2-1).clamp(-1,1).detach(),(img2*2-1).clamp(-1,1)).mean()

def psnr_loss_differentiable(img1, img2, target_db=40.0, max_val=1.0):
    return F.relu(((img1.detach()-img2)**2).mean()-(max_val**2)*10**(-target_db/10.0))


# ── KEY FIX: normalised pixel selection ──────────────────────────────────────

def _minmax_norm(t, eps=1e-8):
    """Min-max normalise 2-D tensor to [0,1]."""
    return (t-t.min())/(t.max()-t.min()+eps)


def get_distortion_aware_mask(model, image, true_label, k,
                              image_size=(224,224), target_class=None,
                              w_grad=0.7, w_invis=0.3, return_maps=False):
    """
    Top-K pixel selection with FIXED score formula:
        grad_norm  = minmax( ||grad||_2 )        ∈ [0,1]
        invis_norm = minmax( 1 - local_var_norm ) ∈ [0,1]
        S(i,j)     = grad_norm * w_grad + invis_norm * w_invis

    Previously grad_mag was raw (unbounded), drowning out invis completely.
    After normalisation both terms have equal dynamic range so weights
    actually control the trade-off.
    """
    if k == 0:
        mask = torch.zeros(1,3,*image_size,device=image.device)
        return (mask,None,None) if return_maps else mask
    model.eval()
    img = image.clone().detach()
    with torch.no_grad():
        probs = F.softmax(model(img),dim=1)[0]
        if target_class is None:
            for idx in torch.argsort(probs,descending=True):
                if idx.item()!=true_label.item(): target_class=idx.item(); break
    img.requires_grad_(True)
    out = model(img)
    (out[0,target_class]-out[0,true_label]).backward()
    grad_raw = torch.norm(img.grad.detach()[0],p=2,dim=0)
    gray      = img[0].detach().mean(dim=0,keepdim=True).unsqueeze(0)
    lm        = F.avg_pool2d(gray,5,stride=1,padding=2)
    lm2       = F.avg_pool2d(gray**2,5,stride=1,padding=2)
    variance  = (lm2-lm**2).clamp(min=0).squeeze()
    invis_raw = 1.0 - variance/(variance.max()+1e-8)
    # *** NORMALISE BOTH MAPS ***
    grad_norm  = _minmax_norm(grad_raw)
    invis_norm = _minmax_norm(invis_raw)
    score = grad_norm*w_grad + invis_norm*w_invis
    flat  = score.flatten(); k_safe=min(k,flat.numel())
    _,topk = torch.topk(flat,k_safe)
    mask_flat=torch.zeros_like(flat); mask_flat[topk]=1.0
    B,C,H,W=image.shape; mask=mask_flat.reshape(1,1,H,W).repeat(1,C,1,1)
    if return_maps: return mask,grad_norm.detach().cpu(),invis_norm.detach().cpu()
    return mask


# ── Jaccard overlap diagnostic ────────────────────────────────────────────────

def jaccard_overlap(mask_a, mask_b):
    a=(mask_a[0,0]>0.5).flatten(); b=(mask_b[0,0]>0.5).flatten()
    inter=(a&b).sum().item(); union=(a|b).sum().item()
    return inter/union if union>0 else 1.0


def log_pixel_overlap(model, dataloader, device, k, weight_combos, n_samples=5):
    print(f"\n{'─'*60}")
    print(f"[Pixel-overlap diagnostic]  k={k}, n_samples={n_samples}")
    print("Expected: off-diagonal Jaccard < 0.90 after normalisation fix")
    imgs_c,lbls_c=[],[]
    for imgs,lbls,_ in dataloader:
        for i in range(imgs.size(0)):
            imgs_c.append(imgs[i:i+1].to(device)); lbls_c.append(lbls[i:i+1].to(device))
            if len(imgs_c)>=n_samples: break
        if len(imgs_c)>=n_samples: break
    n=len(weight_combos); labels=[f"({wg:.1f},{wi:.1f})" for wg,wi in weight_combos]
    masks_per_combo=[]
    for wg,wi in weight_combos:
        combo=[]
        for img,lbl in zip(imgs_c,lbls_c):
            combo.append(get_distortion_aware_mask(model,img,lbl,k,w_grad=wg,w_invis=wi).detach().cpu())
        masks_per_combo.append(combo)
    mat=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            mat[i,j]=np.mean([jaccard_overlap(masks_per_combo[i][s],masks_per_combo[j][s])
                               for s in range(n_samples)])
    print("\nJaccard Overlap Matrix (1.0=identical, 0.0=no overlap)")
    print(" "*14+"  ".join(f"{l:>12}" for l in labels))
    for i,rl in enumerate(labels):
        print(f"{rl:>14}  "+"  ".join(f"{mat[i,j]:.4f}      " for j in range(n)))
    off=mat[~np.eye(n,dtype=bool)]
    if off.mean()>0.95:
        print(f"\n⚠  Off-diagonal mean={off.mean():.4f} > 0.95 — still too similar!")
    else:
        print(f"\n✓  Off-diagonal mean={off.mean():.4f} — combos differ meaningfully.")
    return mat


# ── High-pass filter ──────────────────────────────────────────────────────────

def get_adaptive_high_pass_mask(shape, radius, device, blend_width=10):
    rows,cols=shape[-2:]; crow,ccol=rows//2,cols//2
    y,x=torch.meshgrid(torch.arange(rows,device=device),
                        torch.arange(cols,device=device),indexing='ij')
    dist=torch.sqrt((x.float()-ccol)**2+(y.float()-crow)**2)
    mask2d=torch.sigmoid((dist-radius)/(blend_width/6.0))
    return mask2d.unsqueeze(0).unsqueeze(0).expand(shape[0],shape[1],-1,-1)


# ── PGD Attack ────────────────────────────────────────────────────────────────

class AdaptiveSparseFreqPGD:
    def __init__(self, model, eps=0.03, alpha=0.01, steps=40,
                 freq_radius=20, momentum=0.9,
                 ssim_target=0.99, lambda_ssim=2.0,
                 lpips_target=0.05, lambda_lpips=1.0,
                 psnr_target_db=40.0, lambda_psnr=0.5,
                 lpips_fn=None, w_grad=0.7, w_invis=0.3):
        self.model=model; self.eps=eps; self.alpha=alpha; self.steps=steps
        self.freq_radius=freq_radius; self.momentum=momentum
        self.ssim_target=ssim_target; self.lambda_ssim=lambda_ssim
        self.lpips_target=lpips_target; self.lambda_lpips=lambda_lpips
        self.psnr_target_db=psnr_target_db; self.lambda_psnr=lambda_psnr
        self.lpips_fn=lpips_fn; self.w_grad=w_grad; self.w_invis=w_invis

    def _hp_project(self, delta, hp_mask):
        fft=torch.fft.fftshift(torch.fft.fft2(delta,dim=(-2,-1)),dim=(-2,-1))
        return torch.fft.ifft2(torch.fft.ifftshift(fft*hp_mask,dim=(-2,-1)),dim=(-2,-1)).real

    def attack(self, images, labels, pixel_mask=None, warm_start_delta=None):
        images=images.clone().detach(); labels=labels.clone().detach()
        hp_mask=get_adaptive_high_pass_mask(images.shape,self.freq_radius,images.device)
        delta=(warm_start_delta.clone().detach() if warm_start_delta is not None
               else torch.zeros_like(images).uniform_(-self.eps,self.eps))
        if pixel_mask is not None: delta=delta*pixel_mask
        delta=self._hp_project(delta,hp_mask).requires_grad_(True)
        velocity=torch.zeros_like(delta)
        for _ in range(self.steps):
            adv=images+delta
            loss=(F.cross_entropy(self.model(adv),labels)
                  +self.lambda_ssim*F.relu(ssim_loss_differentiable(images,adv)-(1-self.ssim_target))
                  +self.lambda_lpips*F.relu(lpips_loss_differentiable(images,adv,self.lpips_fn)-self.lpips_target)
                  +self.lambda_psnr*psnr_loss_differentiable(images,adv,self.psnr_target_db))
            loss.backward()
            grad=delta.grad.detach()
            g_norm=torch.norm(grad.reshape(grad.shape[0],-1),dim=1).reshape(-1,1,1,1).clamp(1e-12)
            velocity=self.momentum*velocity+grad/g_norm
            delta.data=delta.data+self.alpha*velocity.sign()
            delta.data=torch.clamp(delta.data,-self.eps,self.eps)
            if pixel_mask is not None: delta.data=delta.data*pixel_mask
            delta.data=self._hp_project(delta.data,hp_mask)
            delta.data=torch.clamp(images+delta.data,0,1)-images
            delta.grad.zero_()
            with torch.no_grad():
                if (self.model(images+delta).argmax(1)!=labels).all(): break
        return (images+delta.detach()).clamp(0,1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_attack(model, dataloader, attack_obj, k_pixels, device,
                    inception_model=None, lpips_metric=None, prev_deltas=None):
    model.eval()
    correct_clean=correct_adv=total=0
    ssim_vals,psnr_vals,lpips_vals=[],[],[]
    clean_list,adv_list,img_times,new_deltas=[],[],[],[]
    for images,labels,_ in dataloader:
        images,labels=images.to(device),labels.to(device)
        with torch.no_grad():
            correct_clean+=(model(images).argmax(1)==labels).sum().item()
        for i in range(images.size(0)):
            t0=time.time(); img=images[i:i+1]; lbl=labels[i:i+1]
            pixel_mask=get_distortion_aware_mask(model,img,lbl,k_pixels,
                                                  w_grad=attack_obj.w_grad,
                                                  w_invis=attack_obj.w_invis).to(device)
            warm=(prev_deltas[total].to(device) if (prev_deltas and total<len(prev_deltas)) else None)
            adv_img=attack_obj.attack(img,lbl,pixel_mask,warm_start_delta=warm)
            new_deltas.append((adv_img-img).detach().cpu())
            with torch.no_grad():
                correct_adv+=(model(adv_img).argmax(1)==lbl).sum().item()
            ssim_vals.append(calculate_ssim(img,adv_img))
            psnr_vals.append(calculate_psnr(img,adv_img))
            if lpips_metric: lpips_vals.append(lpips_metric(img,adv_img))
            clean_list.append(img.cpu()); adv_list.append(adv_img.cpu())
            img_times.append(time.time()-t0); total+=1
            if total%10==0:
                lp=np.mean(lpips_vals) if lpips_vals else 0.0
                print(f"  [{total:>3}]  SSIM={np.mean(ssim_vals):.4f}  "
                      f"PSNR={np.mean(psnr_vals):.2f}dB  LPIPS={lp:.4f}  "
                      f"t/img={np.mean(img_times):.3f}s")
    clean_acc=100*correct_clean/total; adv_acc=100*correct_adv/total
    asr=100*(correct_clean-correct_adv)/correct_clean if correct_clean else 0.0
    fid_score=None
    if inception_model and len(clean_list)>1:
        try:
            ct=torch.cat(clean_list,dim=0); at=torch.cat(adv_list,dim=0)
            m1,s1=calculate_activation_statistics(ct,inception_model,device)
            m2,s2=calculate_activation_statistics(at,inception_model,device)
            fid_score=calculate_fid(m1,s1,m2,s2)
        except Exception as e:
            print(f"FID failed: {e}"); fid_score=0.0
    return (clean_acc,adv_acc,asr,
            float(np.mean(ssim_vals)),float(np.mean(psnr_vals)),
            float(np.mean(lpips_vals)) if lpips_vals else 0.0,
            fid_score,float(np.mean(img_times)),new_deltas)


# ── Binary search for optimal k ──────────────────────────────────────────────

def binary_search_optimal_k(model, dataloader, device, eps, alpha, steps,
                             ssim_threshold=0.99, psnr_threshold=40.0,
                             lpips_threshold=0.05, asr_threshold=100.0,
                             fid_threshold=5.0, k_min=10, k_max=None,
                             max_binary_rounds=8, w_grad=0.7, w_invis=0.3,
                             inception_model=None, lpips_metric=None):
    atk=AdaptiveSparseFreqPGD(model,eps=eps,alpha=alpha,steps=steps,
                               freq_radius=25,momentum=0.9,
                               ssim_target=ssim_threshold,lambda_ssim=2.0,
                               lpips_target=lpips_threshold,lambda_lpips=1.0,
                               psnr_target_db=psnr_threshold,lambda_psnr=0.5,
                               lpips_fn=lpips_metric,w_grad=w_grad,w_invis=w_invis)
    first_batch=next(iter(dataloader))
    if k_max is None: k_max=first_batch[0].shape[2]*first_batch[0].shape[3]
    results,prev_deltas=[],None
    def _run(k,phase):
        nonlocal prev_deltas
        out=evaluate_attack(model,dataloader,atk,k,device,inception_model,lpips_metric,prev_deltas)
        ca,aa,asr,s,p,l,fid,t,prev_deltas=out
        fid_v=fid if fid is not None else 0.0
        r=dict(k=k,clean_accuracy=ca,adversarial_accuracy=aa,attack_success_rate=asr,
               ssim=s,psnr=p,lpips=l,fid=fid_v,avg_time_per_image=t,phase=phase,
               w_grad=w_grad,w_invis=w_invis)
        results.append(r)
        ok=(asr>=asr_threshold and s>=ssim_threshold and p>=psnr_threshold
            and l<=lpips_threshold and fid_v<fid_threshold)
        print(f"  k={k:>5}  ASR={asr:.2f}%  SSIM={s:.4f}  PSNR={p:.2f}dB  "
              f"LPIPS={l:.4f}  FID={fid_v:.4f}  {'✓' if ok else '✗'}")
        return r,ok
    k,high_k,low_k=k_min,None,k_min
    while k<=k_max:
        r,ok=_run(k,'ramp-up')
        if r['attack_success_rate']>=asr_threshold: high_k=k; break
        low_k=k; k=max(k+10,int(k*2))
    if high_k is None: high_k=k_max
    best_k=high_k
    for _ in range(max_binary_rounds):
        mid_k=(low_k+high_k)//2
        if mid_k in(low_k,high_k): break
        r,ok=_run(mid_k,'binary')
        if ok: best_k=mid_k; high_k=mid_k
        else: low_k=mid_k
    return results,best_k


# ── Composite score ───────────────────────────────────────────────────────────

def compute_composite_score(r, asr_t=100.0, ssim_t=0.99,
                             psnr_t=40.0, lpips_t=0.05, fid_t=5.0):
    """Weighted composite ∈ [0,1]. ASR 40% | SSIM 20% | PSNR 15% | LPIPS 15% | FID 10%"""
    w={'asr':0.40,'ssim':0.20,'psnr':0.15,'lpips':0.15,'fid':0.10}
    sc={'asr':min(r['best_asr']/asr_t,1.0),'ssim':min(r['best_ssim']/ssim_t,1.0),
        'psnr':min(r['best_psnr']/psnr_t,1.0),
        'lpips':min(lpips_t/(r['best_lpips']+1e-9),1.0),
        'fid':min(fid_t/(r['best_fid']+1e-9),1.0)}
    return max(0.0, sum(w[m]*sc[m] for m in w)-(r['best_k']/1000)*0.01)


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_hp_comparison(hp_summary, prefix='hp_tuning_resnet50_fixed'):
    labels=[f"({r['w_grad']:.1f},{r['w_invis']:.1f})" for r in hp_summary]
    x=np.arange(len(labels))
    panels=[('best_asr','ASR (%)',100.0,'red','≥100%',False),
            ('best_ssim','SSIM',0.99,'green','≥0.99',False),
            ('best_psnr','PSNR (dB)',40.0,'teal','≥40dB',False),
            ('best_lpips','LPIPS',0.05,'purple','≤0.05',True),
            ('best_fid','FID',5.0,'brown','<5',True),
            ('best_k','Best K',None,'navy','lower=efficient',True)]
    fig,axes=plt.subplots(2,3,figsize=(22,11))
    fig.suptitle('ResNet50 — Pixel Selection Weight Tuning  [FIXED: both maps normalised]\n'
                 'S(i,j) = norm(G)·w_grad + norm(w)·w_invis',fontsize=12)
    for ax,(key,ylabel,thresh,col,lbl,lb) in zip(axes.flatten(),panels):
        vals=[r[key] for r in hp_summary]
        bars=ax.bar(x,vals,color='steelblue',alpha=0.75,edgecolor='k',lw=0.6)
        if thresh: ax.axhline(thresh,color=col,ls='--',lw=1.5,label=f'Target {lbl}')
        ax.set_xticks(x); ax.set_xticklabels(labels,rotation=40,ha='right',fontsize=8)
        ax.set_xlabel('(w_grad, w_invis)'); ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel}{"  ↓ better" if lb else "  ↑ better"}')
        if thresh: ax.legend(fontsize=8)
        ax.grid(axis='y',alpha=0.4)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),
                    f'{v:.3f}' if isinstance(v,float) else str(v),
                    ha='center',va='bottom',fontsize=7)
    plt.tight_layout(); out=f'{prefix}_comparison.png'
    plt.savefig(out,bbox_inches='tight',dpi=120); print(f"Plot → {out}"); plt.close()


def visualize_k_search_per_combo(all_results, prefix='hp_tuning_resnet50_fixed'):
    n=len(all_results); fig,axes=plt.subplots(n,5,figsize=(30,5*n))
    if n==1: axes=axes[None,:]
    panels=[('attack_success_rate','ASR (%)',100.0,'red','≥100%'),
            ('ssim','SSIM',0.99,'green','≥0.99'),
            ('psnr','PSNR dB',40.0,'teal','≥40dB'),
            ('lpips','LPIPS',0.05,'purple','≤0.05'),
            ('fid','FID',5.0,'brown','<5')]
    for row,(clabel,results) in enumerate(all_results.items()):
        k_v=[r['k'] for r in results]
        colors=['tab:blue' if r.get('phase')=='ramp-up' else 'tab:orange' for r in results]
        for col,(key,ylabel,thresh,c,lbl) in enumerate(panels):
            ax=axes[row,col]; vals=[r[key] for r in results]
            ax.scatter(k_v,vals,c=colors,zorder=3,s=50)
            ax.plot(k_v,vals,'k--',alpha=0.35,lw=1)
            ax.axhline(thresh,color=c,ls=':',lw=1.5,label=f'Target {lbl}')
            ax.set_xlabel('K Pixels'); ax.set_ylabel(ylabel)
            ax.set_title(f'[{clabel}] {ylabel} vs K')
            ax.grid(True,alpha=0.4); ax.legend(fontsize=7)
    plt.tight_layout(); out=f'{prefix}_k_search.png'
    plt.savefig(out,bbox_inches='tight',dpi=100); print(f"K-search plot → {out}"); plt.close()


# ── Hyperparameter tuning loop ────────────────────────────────────────────────

def run_hyperparameter_tuning(model, dataloader, device, eps, alpha, steps,
                               ssim_threshold, psnr_threshold, lpips_threshold,
                               asr_threshold, fid_threshold,
                               weight_combinations=WEIGHT_COMBINATIONS):
    print("\n"+"="*70)
    print("Loading InceptionV3 (shared across all HP runs) …")
    inception_model=InceptionV3FeatureExtractor().to(device)
    print("Loading LPIPS/AlexNet (shared) …")
    lpips_metric=LPIPSMetric(net='alex',device=str(device))
    first_batch=next(iter(dataloader))
    H,W=first_batch[0].shape[2],first_batch[0].shape[3]
    log_pixel_overlap(model,dataloader,device,k=(H*W)//40,
                      weight_combos=weight_combinations,n_samples=5)
    all_results,hp_summary={},[]
    for w_grad,w_invis in weight_combinations:
        label=f"wG={w_grad:.1f}_wI={w_invis:.1f}"
        print(f"\n{'#'*70}\n# w_grad={w_grad:.1f}  w_invis={w_invis:.1f}\n{'#'*70}")
        t0=time.time()
        results,best_k=binary_search_optimal_k(
            model,dataloader,device,eps,alpha,steps,
            ssim_threshold=ssim_threshold,psnr_threshold=psnr_threshold,
            lpips_threshold=lpips_threshold,asr_threshold=asr_threshold,
            fid_threshold=fid_threshold,w_grad=w_grad,w_invis=w_invis,
            inception_model=inception_model,lpips_metric=lpips_metric)
        elapsed=time.time()-t0; all_results[label]=results
        best_row=max(results,key=lambda r:(r['attack_success_rate'],r['ssim'],
                                            r['psnr'],-r['lpips'],-r['fid']))
        hp_summary.append(dict(w_grad=w_grad,w_invis=w_invis,best_k=best_k,
                               best_asr=best_row['attack_success_rate'],
                               best_ssim=best_row['ssim'],best_psnr=best_row['psnr'],
                               best_lpips=best_row['lpips'],best_fid=best_row['fid'],
                               elapsed_s=elapsed))
        print(f"\n  [{label}]  best_k={best_k}  "
              f"ASR={best_row['attack_success_rate']:.2f}%  SSIM={best_row['ssim']:.4f}  "
              f"PSNR={best_row['psnr']:.2f}dB  LPIPS={best_row['lpips']:.4f}  "
              f"FID={best_row['fid']:.4f}  time={timedelta(seconds=int(elapsed))}")
    return all_results,hp_summary


# ── Main ──────────────────────────────────────────────────────────────────────

def load_model(name, device):
    print(f"\nLoading {name} …")
    try: m=models.resnet50(pretrained=True)
    except TypeError: m=models.resnet50(weights='IMAGENET1K_V1')
    return m.to(device).eval()


def main():
    t_start=time.time()
    DATA_DIR='/kaggle/input/imagenetmini-1000/imagenet-mini/val'
    NUM_SAMPLES=50; EPS,ALPHA,STEPS=8/255,2/255,40
    SSIM_T,PSNR_T,LPIPS_T,ASR_T,FID_T=0.99,40.0,0.05,100.0,5.0
    if not os.path.exists(DATA_DIR): print(f"Data not found: {DATA_DIR}"); return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
    dataset=ImageNetMiniDataset(DATA_DIR,transform=transform,limit=NUM_SAMPLES)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
    model=load_model('resnet50',device)
    all_results,hp_summary=run_hyperparameter_tuning(
        model,dataloader,device,EPS,ALPHA,STEPS,
        ssim_threshold=SSIM_T,psnr_threshold=PSNR_T,
        lpips_threshold=LPIPS_T,asr_threshold=ASR_T,fid_threshold=FID_T)
    visualize_hp_comparison(hp_summary); visualize_k_search_per_combo(all_results)
    for r in hp_summary:
        r['composite_score']=compute_composite_score(r,asr_t=ASR_T,ssim_t=SSIM_T,
                                                      psnr_t=PSNR_T,lpips_t=LPIPS_T,fid_t=FID_T)
    ranked=sorted(hp_summary,key=lambda r:r['composite_score'],reverse=True)
    total_time=time.time()-t_start
    print("\n"+"="*150)
    print("HYPERPARAMETER TUNING RESULTS — ResNet50  [normalised pixel selection]")
    print("="*150)
    rows=[{"w_grad":f"{r['w_grad']:.1f}","w_invis":f"{r['w_invis']:.1f}",
           "Best K":r['best_k'],"ASR (%) ↑":f"{r['best_asr']:.2f}",
           "SSIM ↑":f"{r['best_ssim']:.4f}","PSNR (dB) ↑":f"{r['best_psnr']:.2f}",
           "LPIPS ↓":f"{r['best_lpips']:.4f}","FID ↓":f"{r['best_fid']:.4f}",
           "Composite ↑":f"{r['composite_score']:.4f}",
           "Time":str(timedelta(seconds=int(r['elapsed_s'])))} for r in ranked]
    pd.set_option('display.max_columns',None); pd.set_option('display.width',1600)
    print(pd.DataFrame(rows).to_string(index=False))
    best=ranked[0]
    print(f"\n{'='*150}")
    print(f"★  BEST: w_grad={best['w_grad']:.1f}  w_invis={best['w_invis']:.1f}")
    for k,v in [('Composite',f"{best['composite_score']:.4f}"),('Best k',best['best_k']),
                ('ASR',f"{best['best_asr']:.2f}%"),('SSIM',f"{best['best_ssim']:.4f}"),
                ('PSNR',f"{best['best_psnr']:.2f} dB"),('LPIPS',f"{best['best_lpips']:.4f}"),
                ('FID',f"{best['best_fid']:.4f}")]:
        print(f"   {k:<12}: {v}")
    print(f"{'='*150}\nTOTAL TIME: {timedelta(seconds=int(total_time))}")


if __name__=="__main__":
    main()
