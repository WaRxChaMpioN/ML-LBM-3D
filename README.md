# ML-LBM-3D: CNN Surrogate for 3D Porous Media Flow

**Predicting steady-state velocity fields in 3D porous media using convolutional neural networks as a surrogate for Lattice Boltzmann simulations.**

---

## Overview

ML-LBM-3D is a deep learning pipeline that replaces expensive 3D Lattice Boltzmann (D3Q19 MRT-LBM) simulations with a fast CNN surrogate model:

- **3D CNN surrogate** replacing expensive D3Q19 MRT-LBM simulations
- **Predicts 3D velocity fields** (ux, uy, uz) directly from pore geometry (EDT-normalised input)
- **Supports plain porous media AND fractured geometries**
- Trained on **Wendian HPC** (Colorado School of Mines) using NVIDIA V100 GPUs
- **Domain:** 64³ voxels | **Architecture:** K4N8 encoder-decoder with residual blocks | ~18M parameters

---

## Scientific Background

This work builds on the following publications:

1. **Wang et al. (2021)** — ML-LBM methodology (CNN surrogate for LBM):
   > Wang, Y. et al. *ML-LBM: Predicting and Accelerating Steady State Flow Simulation in Porous Media with Convolutional Neural Networks.* Transport in Porous Media (2021).
   > https://doi.org/10.1007/s11242-021-01590-6

2. **Liu & Mostaghimi (2017)** — Correlated porous geometry generation:
   > Liu, M. & Mostaghimi, P. *Characterisation of reactive transport in pore-scale correlated porous media.* Chemical Engineering Science (2017).
   > https://doi.org/10.1016/j.ces.2017.06.044

3. **Cirpka & Attinger (2003)** — Spectral Fourier method for geometry generation:
   > Cirpka, O.A. & Attinger, S. *Effective dispersion in heterogeneous media under random transient flow conditions.* Water Resources Research (2003).
   > https://doi.org/10.1029/2002WR001931

4. **Dwinanda & Dharmawan (2025)** — Fractured geometry generation and analysis:
   > Dwinanda, R. & Dharmawan, I.A. *Quantifying effective fluid transport in fractured sandstone using image-based modeling and lattice Boltzmann simulation.* AIP Advances (2025).
   > https://doi.org/10.1063/5.0291596

---

## Pipeline

### Plain Porous Media

```
Step 1 → generate_geometry_3d.py        (Cirpka & Attinger spectral method)
Step 2 → run_lbm_3d.py                  (D3Q19 MRT-LBM CUDA solver)
Step 3 → poreScaleVelMain.py            (3D CNN training/inference)
Step 4 → paper_plots_3d.py             (analysis and visualisation)
```

### Fractured Media

```
Step 1 → generate_fractured_geometry_3d.py   (fBm fractures, Dwinanda 2025)
Step 2 → run_lbm_3d.py
Step 3 → poreScaleVelMain.py
Step 4 → analyse_fracture_lbm.py
```

---

## File Descriptions

| File | Description |
|------|-------------|
| `generate_geometry_3d.py` | Generates 3D correlated porous media using spectral Fourier method. Implements Cirpka & Attinger (2003) via Liu & Mostaghimi (2017). Outputs EDT-normalised geometry arrays (64³, float32). |
| `generate_fractured_geometry_3d.py` | Extends geometry generation with synthetic fractures using fractional Brownian motion (fBm). Hurst exponent H controls roughness (0.4–0.8); aperture A controls fracture width (2–8 voxels for 64³). Follows Dwinanda & Dharmawan (2025). |
| `run_lbm_3d.py` | D3Q19 MRT-LBM solver implemented in CUDA via PyCUDA. Applies Zou-He pressure boundary conditions. Computes Darcy permeability from velocity field. Outputs (64,64,64,3) float32 velocity arrays. |
| `poreScaleVelMain.py` | 3D CNN training and inference. Architecture: encoder-decoder with residual blocks (K4N8: kernel=4, filters=8, ~18M params). Supports L2 loss + continuity (mass conservation) physics loss (alpha). Includes LR schedule, gradient clipping, checkpoint management. |
| `paper_plots_3d.py` | Generates publication-quality figures. CNN vs LBM accuracy curves, 3D velocity visualisation, permeability scatter, fracture physics analysis. |
| `analyse_fracture_lbm.py` | Pure LBM analysis of fractured geometries. Computes porosity, effective porosity, tortuosity (BFS/Dijkstra), SSA, throat size, coordination number. Reproduces figures from Dwinanda & Dharmawan (2025). |

---

## Quick Start

### 1. Generate Geometries (1000 samples, 64³)

```bash
python3 generate_geometry_3d.py --dims 3 --n_total 1000 --output_dir ./dataset_64
```

### 2. Run LBM (requires CUDA GPU)

```bash
python3 run_lbm_3d.py --dataset_dir ./dataset_64 --subset train --start_id 1 --end_id 800
```

### 3. Train CNN

```bash
python3 poreScaleVelMain.py --train True --nDims 3 --width 64 --height 64 --depth 64 \
    --numFilters 8 --baseKernelSize 4 --learnRate 1e-4 --batch-size 4 \
    --dataset ./dataset_64
```

### 4. Fractured Geometries

```bash
python3 generate_fractured_geometry_3d.py --size 64 --n_total 1000 \
    --H_min 0.4 --H_max 0.8 --A_min 2 --A_max 6 --frac_prob 0.7
```

---

## HPC Usage (SLURM — Wendian)

```bash
cd hpc/
bash submit_geom_64.sh     # geometry generation on compute nodes
bash submit_lbm_64.sh      # LBM on GPU nodes
bash submit_cnn_64_v2.sh   # CNN training on GPU
```

See [docs/pipeline.md](docs/pipeline.md) for full HPC workflow details.

---

## Results

| Metric | Value |
|--------|-------|
| Best validation loss | 0.0076 (epoch 510) |
| R² on 64³ test set | ~0.85 |
| Epoch time | ~42 s (NVIDIA V100 32 GB) |

---

## Requirements

- Python 3.11+
- TensorFlow 2.x
- PyCUDA (CUDA-capable GPU required for LBM)
- NumPy, SciPy, scikit-image, matplotlib, tqdm, h5py

```bash
pip install -r requirements.txt
```

---

## Acknowledgements

All code in this repository is original and written by Kaushal Jha. The implementations are based on methods described in the papers listed under [Scientific Background](#scientific-background):

- The CNN surrogate methodology follows Wang et al. (2021)
- The geometry generation follows Cirpka & Attinger (2003) as described in Liu & Mostaghimi (2017)
- The fractured geometry generation follows the approach of Dwinanda & Dharmawan (2025)

The extension of the CNN surrogate to fractured porous media — combining fBm fracture generation with the ML-LBM training pipeline — is an original contribution by Kaushal Jha.

If you use this code, please cite the relevant methodology papers above.

---

## License

MIT License — see [LICENSE](LICENSE).
