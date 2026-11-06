# PiDSED: Physics-Informed Differentiable SED Fitting for Self-Supervised Photometric Redshift Estimation

Official Python implementation of PiDSED (Physics-Informed Differentiable SED Fitting), a self-supervised photometric redshift (photo-$z$) estimation framework presented in the paper:

> **Physics-Informed Differentiable SED Fitting for Self-Supervised Photometric Redshift Estimation**  
>  Zhijian Luo\*, Junhao Lu\*, Jianzhen Chen\*, Zhu Chen, Hubing Xiao, Shaohua Zhang, and Chenggang Shu  
 


---

##  Overview

PiDSED bridges physical interpretability with deep-learning efficiency to address the acute scarcity of spectroscopic labels and domain shift in wide-field photometric surveys (e.g., Euclid, LSST, CSST, Roman). 

### Key Innovations:
- **Differentiable Physics Engine:** Integrates a precomputed four-dimensional theoretical photometric flux tensor $\mathbf{G} \in \mathbb{R}^{N_z \times N_E \times N_{\text{tpl}} \times N_b}$ derived from Flexible Stellar Population Synthesis (`FSPS`) templates.
- **End-to-End Differentiable Grid Interpolation Layer:** Computes continuous forward predictions from estimated redshift ($z$), dust extinction ($E(B-V)$), and template coefficients ($c_{\text{tpl}}$) directly to synthetic multi-band fluxes.
- **Label-Free Self-Supervised Training:** Optimized via a composite loss function comprising photometric flux reconstruction, inter-band color consistency, template $\ell_1$ sparsity, and extinction-posterior variance constraints—without requiring spectroscopic redshift targets during training.

---

##  Repository Structure

```text
.
├── grid/                        # Scripts for precomputing the 4D flux tensor G
├── Zenodo/                      # Data downloading instructions & Zenodo links
│   └── ...                      # Tensor construction utilities & filter responses
├── PiDSED_photoz.py             # Main training pipeline for PiDSED
├── PiDSED_photoz_infer.py       # Inference & evaluation script for trained PiDSED models
├── pure_mlp_photoz.py           # Training script for the purely supervised MLP baseline
├── pure_mlp_photoz_infer.py     # Inference script for the supervised MLP baseline
├── random_PiDSED_photoz.py      # Seed-reproducibility experiments for PiDSED
├── random_pure_mlp.py           # Seed-reproducibility experiments for supervised MLP
└── README.md                    # Project documentation
```

---

##  Data Availability & Zenodo Assets

The dataset, filter transmission curves, and precomputed physical flux tensor $\mathbf{G}$ are archived on **Zenodo**:
- **Zenodo DOI:** [`10.5281/zenodo.xxxxxxx`](https://doi.org/10.5281/zenodo.xxxxxxx)

### Included Files in Zenodo Archive:
1. `G_tensor.npy`: Precomputed 4D static flux tensor ($\mathbb{R}^{401 \times 50 \times 12 \times 11}$).
2. Filter response curve data files for 11 photometric bands (`GALEX_NUV`, `CFHT_u*`, `HSC_g/r/i/z/y`, `UltraVISTA_J/Ks`, `IRAC_CH1/CH2`).
3. Compiled COSMOS2020 training, validation, and test subsets for both spectroscopic and photometric samples.

---

##  Environment & Installation

### Requirements:
- Python $\ge$ 3.8
- Tensorflow $\ge$ 2.6 (with CUDA acceleration support)
- numpy>=1.24
- scipy>=1.10          # dependency of eazy
- astropy>=5.3         # Quantity input for dust_extinction
- matplotlib>=3.7      # diagnostics figure only
- dust_extinction>=1.2 # G23 attenuation curve
- eazy-py>=0.7         # Inoue14 IGM model (import eazy)
- threadpoolctl>=3.1   # optional: cap BLAS threads


---

## Quick Start & Usage

### 1. Precomputing the 4D Physical Flux Tensor
To regenerate or inspect the 4D SPS flux grid:
```bash
python grid/build_tensor.py
```

### 2. Training PiDSED
Train PiDSED in a self-supervised manner on unlabeled photometric fluxes:
```bash
python PiDSED_photoz.py
```

### 3. Inference & Benchmarking
Run batch evaluation on the held-out test sets to obtain $\sigma_{\text{NMAD}}$, outlier fraction ($\eta$), and median bias:
```bash
python PiDSED_photoz_infer.py --checkpoint path/to/model.pt
```

### 4. Running Baselines
To execute the supervised MLP baseline experiments:
```bash
python pure_mlp_photoz.py
python pure_mlp_photoz_infer.py
```

---

##  Performance Benchmark (COSMOS2020 11-Band Evaluation)

| Method | Target Test Set | Training Labels | $\sigma_{\text{NMAD}}$ | $\eta$ (%) | $\operatorname{bias}$ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **EAZY** | Photometric Test Set | None | $0.0282 \pm 0.0003$ | $9.44 \pm 0.22$ | $-0.0007 \pm 0.0002$ |
| **MLP** | Photometric Test Set | $z_{\text{spec}}$ (Cross-Domain) | $0.0295 \pm 0.0020$ | $5.89 \pm 0.68$ | $+0.0026 \pm 0.0052$ |
| **MLP-PL** | Photometric Test Set | Pseudo-$z$ (EAZY) | $0.0323 \pm 0.0015$ | $8.54 \pm 0.40$ | $-0.0019 \pm 0.0048$ |
| **PiDSED** | Photometric Test Set | **None (In-Domain Self-Supervised)** | **$0.0234 \pm 0.0004$** | **$3.90 \pm 0.57$** | **$+0.0028 \pm 0.0005$** |

---

##  Citation

If you use PiDSED in your research, please cite our paper:

```bibtex
@article{Luo2026PiDSED,
  author    = {Luo, Zhijian and Lu, Junhao and Chen, Jianzhen and Chen, Zhu and Xiao, Hubing and Zhang, Shaohua and Shu, Chenggang},
  title     = {Physics-Informed Differentiable SED Fitting for Self-Supervised Photometric Redshift Estimation},
  journal   = {xxxxxxxxxxxxxxxxxxxxxxxxx},
  volume    = {xx},
  number    = {xx},
  pages     = {xxxxxxx},
  year      = {xxxx},
  doi       = {xx.xxxx/xxxxxx-xxx-xxxx-x}
}
```

---

##  Contact & Support

For questions, issues, or collaborations regarding PiDSED, please open an issue in this repository or reach out directly to **Zhijian Luo** or **Junhao Lu** (Shanghai Key Lab for Astrophysics, Shanghai Normal University).
