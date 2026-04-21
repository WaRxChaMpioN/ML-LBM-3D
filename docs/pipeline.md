# ML-LBM-3D Pipeline Documentation

## Overview

This document describes the full computational pipeline for training and evaluating the 3D CNN surrogate model for porous media flow simulation.

---

## Stage 1: Geometry Generation

**Script:** `generate_geometry_3d.py`

Generates 3D correlated porous media geometries using the spectral Fourier method of Cirpka & Attinger (2003), as described by Liu & Mostaghimi (2017).

### Algorithm

1. Generate a 3D Gaussian random field in Fourier space with power spectrum:
   `S(k) = exp(-2π² |k|² λ²)`
   where λ is the correlation length.
2. Threshold the field at quantile `(1 - φ)` to achieve target porosity φ.
3. Compute the Euclidean Distance Transform (EDT) of the pore space and normalise to [0, 1].

### Output

```
dataset_64/
  train/  geom_00001.npy  edt_00001.npy  ...  (800 samples)
  val/    geom_00001.npy  edt_00001.npy  ...  (100 samples)
  test/   geom_00001.npy  edt_00001.npy  ...  (100 samples)
```

- `geom_*.npy`: Binary mask, shape (64,64,64), dtype uint8. 1=pore, 0=solid.
- `edt_*.npy`: Normalised EDT, shape (64,64,64), dtype float32.

### HPC

```bash
sbatch hpc/submit_geom_64.sh
```

---

## Stage 2: LBM Simulation

**Script:** `run_lbm_3d.py`

Runs the D3Q19 MRT-LBM solver for each geometry to produce ground-truth velocity fields.

### Physics

- **Velocity set:** D3Q19 (19-velocity 3D model)
- **Collision:** Multiple Relaxation Time (MRT) for improved stability
- **Boundary conditions:** Zou-He pressure BCs at inlet (x=0) and outlet (x=N-1)
- **Wall BC:** Full-way bounce-back (no-slip)
- **Convergence:** Stops when max velocity change < tol (default 1e-6) or max_iters reached
- **Permeability:** Darcy's law: `K = ν × ū_x × L / ΔP`

### Output

```
dataset_64/train/  vel_00001.npy  perm_00001.npy  ...
```

- `vel_*.npy`: Velocity field, shape (64,64,64,3), dtype float32.
- `perm_*.npy`: Scalar Darcy permeability (lattice units).

### HPC (SLURM array for train/val/test in parallel)

```bash
sbatch hpc/submit_lbm_64.sh
```

---

## Stage 3: CNN Training

**Script:** `poreScaleVelMain.py`

Trains the K4N8 encoder-decoder 3D CNN to map EDT geometry → velocity field.

### Architecture

```
Input: (64,64,64,1) EDT geometry
  ↓  ResBlock(8) → Conv3D stride=2 → (32,32,32,16)
  ↓  ResBlock(16) → Conv3D stride=2 → (16,16,16,32)
  ↓  ResBlock(32) → Conv3D stride=2 → (8,8,8,64)
  ↓  ResBlock(64) [bottleneck]
  ↑  ConvTranspose3D + skip → ResBlock(32)
  ↑  ConvTranspose3D + skip → ResBlock(16)
  ↑  ConvTranspose3D + skip → ResBlock(8)
  ↓  Conv3D(3) → (64,64,64,3) velocity
```

### Loss Function

```
L = L_L2 + α × L_continuity
```

Where:
- `L_L2 = mean((u_pred - u_LBM)²)` — supervised regression loss
- `L_continuity = mean((∇·u_pred)²)` — physics-informed mass conservation loss
- `α = 0.1` — physics loss weight (v2 model)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Initial LR | 1e-4 |
| LR decay | Exponential, 0.95 per 1000 steps |
| Gradient clipping | L2 norm ≤ 1.0 |
| Batch size | 4 |
| Max epochs | 1000 |
| Early stopping | patience=50 |

### Results

| Metric | V1 (L2 only) | V2 (L2 + physics) |
|--------|-------------|-------------------|
| Best val loss | ~0.012 | **0.0076** |
| Best epoch | ~480 | **510** |
| R² (test) | ~0.80 | **~0.85** |
| Epoch time (V100) | ~42 s | ~44 s |

### HPC

```bash
sbatch hpc/submit_cnn_64_v2.sh   # recommended (physics loss)
```

---

## Stage 4: Analysis and Figures

**Scripts:** `paper_plots_3d.py`, `analyse_fracture_lbm.py`

### Plain Porous Media (paper_plots_3d.py)

```bash
python3 paper_plots_3d.py \
    --dataset ./dataset_64 \
    --predictions ./dataset_64/predictions \
    --checkpoint_history ./checkpoints/model_k4n8_v2_history.csv \
    --output_dir ./figures
```

Generates:
- `figures/loss_curve.pdf` — training convergence
- `figures/velocity_scatter.pdf` — CNN vs LBM component scatter
- `figures/velocity_3d_vis.pdf` — midplane velocity visualisation
- `figures/permeability_scatter.pdf` — K scatter plot

### Fractured Geometries (analyse_fracture_lbm.py)

```bash
python3 analyse_fracture_lbm.py \
    --dataset_dir ./dataset_frac_64 \
    --subset test \
    --start_id 1 --end_id 100 \
    --output_dir ./analysis_results \
    --plot
```

Computes: porosity, effective porosity, tortuosity, SSA, throat size, aperture distribution.

---

## Fractured Media Pipeline

For fractured sandstone geometries (following Dwinanda & Dharmawan 2025):

```bash
# 1. Generate fractured geometries
python3 generate_fractured_geometry_3d.py \
    --size 64 --n_total 1000 \
    --H_min 0.4 --H_max 0.8 \
    --A_min 2 --A_max 6 \
    --frac_prob 0.7 \
    --output_dir ./dataset_frac_64

# 2. Run LBM (same script, point to fractured dataset)
python3 run_lbm_3d.py \
    --dataset_dir ./dataset_frac_64 \
    --subset train --start_id 1 --end_id 800

# 3. Train CNN (same script)
python3 poreScaleVelMain.py --train True --dataset ./dataset_frac_64 ...

# 4. Analyse
python3 analyse_fracture_lbm.py --dataset_dir ./dataset_frac_64 --plot
```

---

## References

- Wang et al. (2021) https://doi.org/10.1007/s11242-021-01590-6
- Liu & Mostaghimi (2017) https://doi.org/10.1016/j.ces.2017.06.044
- Cirpka & Attinger (2003) https://doi.org/10.1029/2002WR001931
- Dwinanda & Dharmawan (2025) https://doi.org/10.1063/5.0291596
