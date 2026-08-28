# CoCoGen: Contrastive Counterfactual Generation for Imperceptible Adversarial Attacks

Official research implementation accompanying:

**Contrastive Counterfactual Generation for Imperceptible Adversarial Attacks**

Aman Desai, Hitika Tiwari, Tushar Shinde  
Department of Data Science & AI, School of Engineering and Science  
Indian Institute of Technology Madras (Zanzibar Campus)

**Transactions on Machine Learning Research (TMLR), 2026**

[Paper](paper/cocogen_tmlr.pdf) | [OpenReview](https://openreview.net/forum?id=dnme31GvOd)

---

## Abstract

CoCoGen is an adversarial attack framework for generating highly effective and visually imperceptible adversarial examples under a composite threat model. The method combines an \(L_\infty\) magnitude budget, an adaptively selected spatial sparsity budget, and a high-frequency spectral constraint.

CoCoGen formulates perturbation generation using a contrastive counterfactual margin that explicitly targets the most competitive incorrect class. The perturbation is localized through gradient-based Top-\(k\) spatial projection and constrained to the high-frequency Fourier subspace. Masked momentum-based optimization is then used to cross the decision boundary while preserving perceptual fidelity. An adaptive sparsity/spectral search identifies feasible configurations.

The camera-ready paper reports 100% attack success across the four primary architectures while maintaining high perceptual quality, together with cross-dataset, transferability, ablation, qualitative, and human-perceptual evaluations.

---

## Method Overview

CoCoGen combines four main components:

1. **Contrastive Counterfactual Guidance (CCG)**  
   The attack minimizes the margin between the true class and the most competitive incorrect class.

2. **Top-\(k\) Spatial Projection**  
   Gradient-based attribution identifies the most decision-relevant pixels and restricts the perturbation support to those locations.

3. **High-Frequency Fourier Projection**  
   The perturbation is transformed into the Fourier domain and low-frequency components are suppressed using a radial frequency threshold.

4. **Masked Momentum Optimization and Adaptive Search**  
   Iterative momentum updates are combined with the spatial and spectral constraints. The sparsity and spectral configuration can be selected using the adaptive search procedure.

The composite feasible set is:

\[
\mathcal{F}
=
\left\{
\delta:
\|\delta\|_\infty \leq \epsilon,\;
\|\delta\|_0 \leq k,\;
\delta \in \operatorname{Im}(P_f)
\right\}.
\]

---

## Repository Structure

```text
CoCoGen/
├── README.md
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── .gitignore
│
├── experiments/
│   ├── cocogen.py
│   ├── ablation_cocogen.py
│   └── transferability_cocogen.py
│
├── paper/
│   ├── cocogen_tmlr.pdf
│   └── README.md
│
├── figures/
│   └── README.md
│
├── results/
│   ├── main/
│   ├── ablations/
│   ├── transferability/
│   └── qualitative/
│
├── checkpoints/
│   └── README.md
│
└── data/
    └── README.md
```

The three Python files in `experiments/` are the research implementations supplied with this release. They are intentionally kept close to the experimental code used for the reported studies.

---

# 1. Requirements

## Hardware

The experiments are designed for CUDA-capable GPUs.

Recommended:

- NVIDIA GPU
- CUDA-compatible PyTorch installation
- At least 12 GB GPU memory for standard experiments
- 16 GB or more recommended for larger transferability/metric runs

CPU execution is possible when CUDA is unavailable, but the experiments will be substantially slower.

## Software

Recommended:

- Python 3.10
- PyTorch
- torchvision
- NumPy
- SciPy
- scikit-image
- pandas
- Pillow
- matplotlib
- tqdm
- LPIPS
- timm

The exact package versions used for a final archival reproduction should be pinned to the original experimental environment before publication of a tagged release.

---

# 2. Installation

## Conda

```bash
conda create -n cocogen python=3.10 -y
conda activate cocogen
```

Install a PyTorch build appropriate for the CUDA version installed on your machine.

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate cocogen
```

## Verify PyTorch/CUDA

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

# 3. Dataset

## ImageNet-Mini

The supplied implementation expects an ImageNet-Mini validation directory containing class subdirectories.

Example:

```text
imagenet-mini/
└── val/
    ├── class_001/
    │   ├── image1.JPEG
    │   ├── image2.JPEG
    │   └── ...
    ├── class_002/
    │   └── ...
    └── ...
```

The implementation also recognizes the Kaggle path:

```text
/kaggle/input/imagenetmini-1000/imagenet-mini/val
```

If running locally, change the `DATA_DIR` variable in the corresponding experiment script to the location of your dataset.

The standard preprocessing used by the supplied implementation is:

```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
```

Images are represented in the `[0, 1]` range.

See `data/README.md` for the expected directory layout.

---

# 4. Pretrained Models

The main experiments use:

- ResNet-50
- EfficientNet-B0
- ConvNeXt-Base
- ViT-Base

The transferability experiments additionally use:

- MobileNet-V2
- ShuffleNet-V2
- DenseNet-121
- Vision Mamba Small
- ViT-Base

The supplied code loads standard pretrained models through `torchvision` and, where required, `timm`.

Pretrained model weights are not redistributed in this repository. They are downloaded/cached by the corresponding model libraries when the scripts are executed, subject to the respective model and dataset terms.

---

# 5. Main CoCoGen Experiment

The main implementation is:

```text
experiments/cocogen.py
```

Run:

```bash
python experiments/cocogen.py
```

The supplied implementation uses the following standard settings:

| Parameter | Value |
|---|---:|
| \(L_\infty\) budget \(\epsilon\) | \(8/255\) |
| Iterations \(T\) | 40 |
| Step size \(\alpha\) | \(2/255\) |
| Momentum coefficient \(\mu\) | 1.0 |
| Frequency threshold \(\tau_{freq}\) | 25 |
| SSIM threshold | 0.95 |
| FID threshold | 20 |
| Sparsity \(k\) | Adaptive |

The main implementation evaluates the four primary architectures and reports clean accuracy, adversarial accuracy, ASR, SSIM, FID, LPIPS, \(L_\infty\), \(L_2\), PSNR, and runtime.

The supplied experimental script currently uses a limited number of evaluation images for development/experimental execution. Check the script before a full reproduction and set the evaluation count to the intended experimental protocol.

---

# 6. Main Experiment Output

The main experiment exports:

```text
attack_results_comprehensive.csv
```

and generates model-specific visualization files.

The CSV contains fields including:

```text
Model
K
Clean%
Adv%
ASR%
SSIM
FID
LPIPS
L∞
L2
PSNR
Time/Img
```

For a clean repository, move generated CSVs and plots into:

```text
results/main/
```

after execution.

---

# 7. Ablation Experiments

The ablation implementation is:

```text
experiments/ablation_cocogen.py
```

Run:

```bash
python experiments/ablation_cocogen.py
```

The ablation evaluates the principal components of CoCoGen, including:

- Full CCG + high-frequency configuration
- Without contrastive counterfactual guidance
- Without high-frequency Fourier projection
- Without adaptive search
- Adaptive versus fixed-step search

The ablation implementation reports:

- Attack Success Rate
- SSIM
- PSNR
- LPIPS
- FID
- \(L_\infty\)
- \(L_2\)
- Average time per image

The supplied code exports ablation and search-strategy CSV files. Keep these under:

```text
results/ablations/
```

---

# 8. Adaptive Search Ablation

The search-strategy experiment compares:

```text
Adaptive Grid Search
```

against:

```text
Fixed-Step Grid Search
```

The fixed-step implementation uses a constant search increment, while the adaptive implementation changes its step size according to the observed attack-success progression.

The stopping criteria used by the implementation are:

```text
ASR >= 100%
SSIM > 0.95
FID < 20
```

The fixed-step comparison is implemented in:

```text
experiments/ablation_cocogen.py
```

---

# 9. Transferability Experiments

The transferability implementation is:

```text
experiments/transferability_cocogen.py
```

Run:

```bash
python experiments/transferability_cocogen.py
```

The script evaluates cross-architecture transferability.

Source architectures include:

```text
ResNet-50
ConvNeXt-Base
Vision Mamba Small
ViT-Base
```

Target architectures include CNNs such as:

```text
ResNet-50
MobileNet-V2
ShuffleNet-V2
DenseNet-121
```

and a ViT target.

The supplied transferability experiment evaluates sparsity levels:

```text
k = 100
k = 1000
k = 5000
```

and reports:

- White-box ASR
- Transfer ASR
- PSNR
- SSIM
- LPIPS

The optimization-step experiment evaluates multiple values of \(T\), including:

```text
T ∈ {10, 25, 50, 100, 200}
```

Generated tables should be stored under:

```text
results/transferability/
```

---

# 10. CIFAR-100

The paper includes an additional CIFAR-100 evaluation to examine generalization beyond the primary ImageNet benchmark.

The camera-ready evaluation compares CoCoGen against:

```text
PGD
DeepFool
NCF
ACA
DiffPGD
PerC-AL
AdvDrop
SSAH
```

across:

```text
ResNet-50
EfficientNet-B0
ConvNeXt-Base
ViT
```

The reported evaluation shows that CoCoGen uses substantially fewer modified pixels than the full-image baselines while retaining high attack success and strong perceptual quality.

The supplied three scripts are primarily organized around the ImageNet-Mini experiments. If a separate CIFAR-100 driver is required for one-command reproduction, it should be added as a dedicated experiment script rather than silently reusing the ImageNet loader.

---

# 11. Evaluation Metrics

## Attack Success Rate

ASR is computed over samples that are correctly classified before the attack.

Higher is better.

## PSNR

Peak Signal-to-Noise Ratio measures pixel-level distortion.

Higher is better.

## SSIM

Structural Similarity Index measures structural similarity between clean and adversarial images.

Higher is better.

## LPIPS

Learned Perceptual Image Patch Similarity measures deep perceptual distance.

Lower is better.

## FID

Fréchet Inception Distance measures distributional divergence between clean and adversarial image sets.

Lower is better.

## \(L_\infty\)

The maximum absolute perturbation value.

The main threat model enforces:

\[
\|\delta\|_\infty \leq 8/255.
\]

## \(L_2\)

The Euclidean magnitude of the perturbation.

Lower values indicate smaller overall perturbation energy.

## MUSIQ

MUSIQ is used in the paper as an additional no-reference perceptual image-quality measure. Higher scores indicate better perceived image quality.

---

# 12. Reported Main-Result Configuration

The camera-ready paper specifies:

```text
epsilon       = 8/255
T             = 40
alpha         = 2/255
mu            = 1.0
tau_freq      = 25
tau_s         = 0.95
tau_f         = 20
k             = adaptive
```

The paper selects the frequency threshold \(\tau_{freq}=25\) as the operating point for the main experiments.

For example, the camera-ready frequency sweep reports 100% ASR for \(\tau_{freq}\in\{10,20,25\}\), with \(\tau_{freq}=25\) providing the selected trade-off among attack success and perceptual/distributional metrics.

---

# 13. Reproducibility Workflow

For a fresh environment:

```bash
git clone https://github.com/<USERNAME>/CoCoGen.git
cd CoCoGen

conda create -n cocogen python=3.10 -y
conda activate cocogen

pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch, torchvision, numpy, scipy, skimage, pandas, PIL, matplotlib, tqdm, lpips; print('Core dependencies imported successfully.')"
```

Then configure the ImageNet-Mini path in:

```text
experiments/cocogen.py
```

Run:

```bash
python experiments/cocogen.py
```

For ablations:

```bash
python experiments/ablation_cocogen.py
```

For transferability:

```bash
python experiments/transferability_cocogen.py
```

---

# 14. Kaggle Execution

The supplied implementation supports the Kaggle ImageNet-Mini path:

```text
/kaggle/input/imagenetmini-1000/imagenet-mini/val
```

A typical Kaggle workflow is:

```bash
git clone https://github.com/<USERNAME>/CoCoGen.git
cd CoCoGen
pip install -r requirements.txt
python experiments/cocogen.py
```

Enable GPU acceleration in the Kaggle notebook/runtime before executing the experiments.

---

# 15. Qualitative Results

The paper includes qualitative comparisons between clean images, adversarial outputs, and amplified perturbation maps.

The perturbation maps use method-specific amplification factors solely for visualization. They should not be interpreted as the actual perturbation magnitude.

For the paper's qualitative comparison, the reported amplification factors are:

```text
PGD       ×8
NCF       ×8
ACA       ×8
DiffPGD   ×8
AdvDrop   ×80
PerC-AL   ×80
SSAH      ×80
CoCoGen   ×100
```

The larger amplification factor used for CoCoGen reflects the substantially smaller raw perturbation magnitude.

---

# 16. Important Reproducibility Notes

### Dataset order

The dataset loader scans class directories in sorted order. The exact image subset therefore depends on the contents and directory naming of the dataset supplied to the script.

### Pretrained weights

Different versions of model libraries can expose different model names or pretrained checkpoints. For exact archival reproduction, record the model-library versions and checkpoint identifiers.

### Vision Mamba

The transferability script uses `timm` for Vision Mamba. Model availability can vary across `timm` versions. Record the exact model identifier and package version used for the reported experiment.

### Metric implementations

LPIPS and FID depend on their feature networks and preprocessing. Exact reproduction requires keeping these components fixed.

### Randomness

For strict reproduction, additionally record random seeds and deterministic CUDA settings if those are introduced in future refactoring.

### Runtime

Runtime depends strongly on GPU model, CUDA version, PyTorch version, metric computation, and model loading/caching. Reported wall-clock times should therefore be interpreted relative to the experimental hardware.

---

# 17. Threat Model

CoCoGen operates under the composite feasible set:

\[
\mathcal{F}
=
\left\{
\delta:
\|\delta\|_\infty \leq \epsilon,\;
\|\delta\|_0 \leq k,\;
\delta\in\operatorname{Im}(P_f)
\right\}.
\]

This is more restrictive than conventional attacks that optimize only an \(L_\infty\) constraint.

Accordingly, comparisons with unrestricted or less-constrained attacks should be interpreted as comparisons under different feasible sets rather than as claims of universal superiority.

The most direct comparisons are with methods imposing related spatial, frequency, or perceptual constraints.

---

# 18. Responsible Use

This repository is released for research into adversarial robustness, model security, perceptual robustness, and adversarial machine learning.

Users should only evaluate systems for which they have appropriate authorization.

Potential research applications include:

- adversarial robustness benchmarking
- security evaluation
- studying model decision boundaries
- evaluating adversarial defenses
- perceptual robustness research
- benchmarking sparse and frequency-constrained attacks

---

# 19. Citation

If you use CoCoGen or this implementation in your research, please cite:

```bibtex
@article{desai2026cocogen,
  title   = {Contrastive Counterfactual Generation for Imperceptible Adversarial Attacks},
  author  = {Desai, Aman and Tiwari, Hitika and Shinde, Tushar},
  journal = {Transactions on Machine Learning Research},
  year    = {2026}
}
```

See also:

```text
CITATION.cff
```

for GitHub citation metadata.

---

# 20. License and Third-Party Components

A repository license should be selected by the authors before the public GitHub release.

Third-party packages, pretrained models, and datasets remain subject to their respective licenses and terms.

This repository does not redistribute ImageNet data or pretrained model weights.

---

# 21. Contact

For questions regarding the implementation or paper, please use the GitHub issue tracker or the contact information associated with the TMLR/OpenReview submission.

