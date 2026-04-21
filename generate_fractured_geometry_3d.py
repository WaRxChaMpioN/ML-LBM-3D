#!/usr/bin/env python3
"""
generate_fractured_geometry_3d.py
===================================
Generates 3D fractured porous media geometries for ML-LBM training.
Matches methodology of Dwinanda & Dharmawan (2025), Physics of Fluids 37, 093129.

Pipeline per sample:
  Step 1: Generate porous matrix using Cirpka & Attinger (2003) spectral method
          (same as existing generate_geometry_3d.py)
  Step 2: Generate synthetic fracture using fractional Brownian motion (fBm)
          Parameters: Hurst exponent H, mean aperture A
  Step 3: Embed fracture into porous matrix
          fracture voids override solid voxels → new combined geometry
  Step 4: Quality check (porosity, percolation, throat size)
  Step 5: Save EDT-normalised geometry

Dataset split (matches existing pipeline):
  train:      samples 1     to train_end  (default 800)
  validation: samples train_end+1 to val_end    (default 900)
  test:       samples val_end+1   to n_total    (default 1000)

Output shape: float32 (64, 64, 64, 1) — EDT normalised [0,1]

Usage:
  # Full dataset with fractures
  python3 generate_fractured_geometry_3d.py \\
      --dims 3 --size 64 --n_total 1000 \\
      --output_dir /beegfs/scratch/kaushal_jha/velML3DfracturedDataset_64 \\
      --H_min 0.4 --H_max 0.8 \\
      --A_min 5   --A_max 15 \\
      --frac_prob 0.7 \\
      --n_workers 16 --seed 42

  # Quick test
  python3 generate_fractured_geometry_3d.py \\
      --n_total 10 --n_workers 2 --size 64 \\
      --output_dir ./test_fractured
"""

import argparse
import os
import warnings
import numpy as np
from scipy.ndimage import label, distance_transform_edt, gaussian_filter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SIZE_3D         = 64
LAMBDA_MIN_3D   = 5      # min correlation length (~8% of 64)
LAMBDA_MAX_3D   = 32     # max correlation length (~50% of 64)
EPSILON         = 0.03   # threshold increment above percolation
MU_R            = 1.0
SIGMA_R         = 1.0

POROSITY_MIN    = 0.20   # reject below (too tight)
POROSITY_MAX    = 0.75   # reject above (too open)
MIN_THROAT      = 1.5    # min EDT max value (voxels)

# Fracture parameters (paper: H=0.4-0.8, A=10-30 lattice units)
# Scaled for 64³: A=5-15 (paper used 128³ domain)
H_MIN_DEFAULT   = 0.4
H_MAX_DEFAULT   = 0.8
A_MIN_DEFAULT   = 5      # min aperture (lattice units, scaled for 64³)
A_MAX_DEFAULT   = 15     # max aperture

FRAC_PROB       = 0.7    # probability of fracture in a sample (70% fractured)
MAX_ATTEMPTS    = 60

# ─────────────────────────────────────────────────────────────────────────────
# POROUS MATRIX GENERATION (Cirpka & Attinger 2003)
# ─────────────────────────────────────────────────────────────────────────────
def generate_correlated_field(shape, lam, mu_R=MU_R, sigma_R=SIGMA_R, rng=None):
    """
    Spectral (Fourier) method for correlated Gaussian random field.
    Isotropic Gaussian variogram: gamma(h) = sig_Y^2 * exp(-h^2/lam^2)
    """
    if rng is None:
        rng = np.random.default_rng()

    ndim  = len(shape)
    mu_Y  = np.log(mu_R**2 / np.sqrt(sigma_R**2 + mu_R**2))
    sig_Y = np.sqrt(np.log(sigma_R**2 / mu_R**2 + 1))

    # Build frequency grid
    freq_axes = [np.fft.fftfreq(n) for n in shape]
    grids     = np.meshgrid(*freq_axes, indexing='ij')
    k_sq      = sum(g**2 for g in grids)

    # Gaussian spectral density
    S         = sig_Y**2 * (np.sqrt(np.pi) * lam)**ndim * np.exp(-np.pi**2 * k_sq * lam**2)
    amplitude = np.sqrt(S / np.prod(shape))

    # Complex random field
    noise = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    Y_hat = amplitude * noise
    Y     = np.real(np.fft.ifftn(Y_hat)) * np.prod(shape)
    Y     = mu_Y + sig_Y * (Y - Y.mean()) / (Y.std() + 1e-10)
    return Y


def is_percolating(binary, axis=0):
    """Check if pore space percolates from inlet to outlet along axis."""
    labeled, _ = label(binary)
    inlet_labels  = set(np.unique(labeled.take(0,  axis=axis))) - {0}
    outlet_labels = set(np.unique(labeled.take(-1, axis=axis))) - {0}
    return bool(inlet_labels & outlet_labels)


def find_percolation_threshold(Y, axis=0, n_steps=200):
    """Binary search for minimum threshold allowing percolation."""
    f_min, f_max = Y.min(), Y.max()
    lo, hi = f_min, f_max
    for _ in range(n_steps):
        mid = (lo + hi) / 2
        if is_percolating((Y <= mid).astype(np.uint8), axis):
            hi = mid
        else:
            lo = mid
        if (hi - lo) < (f_max - f_min) / n_steps:
            break
    return hi if hi < f_max else None


# ─────────────────────────────────────────────────────────────────────────────
# FRACTURE GENERATION (fractional Brownian motion, paper method)
# ─────────────────────────────────────────────────────────────────────────────
def generate_fbm_surface(N, H, rng):
    """
    Generate 2D self-affine surface using spectral synthesis (fBm).
    
    Based on Madadi & Sahimi (2003) method used in paper.
    Height difference: <[B_H(r) - B_H(r0)]^2> ~ |r-r0|^{2H}
    
    Args:
        N: surface size (N x N grid)
        H: Hurst exponent (0 < H < 1)
           H > 0.5 → smooth, correlated (paper: 0.4-0.8)
           H < 0.5 → rough, anti-correlated
        rng: numpy random generator
    
    Returns:
        surface: (N, N) float array, zero-mean height field
    """
    # Frequency grid
    fx = np.fft.fftfreq(N)
    fy = np.fft.fftfreq(N)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    k = np.sqrt(FX**2 + FY**2)
    k[0, 0] = 1.0  # avoid division by zero at DC

    # Power spectrum: S(k) ~ k^{-(2H+2)} for 2D fBm surface
    power = k**(-(H + 1.0))
    power[0, 0] = 0.0  # zero DC component (zero mean)

    # Random complex amplitudes
    phase = rng.uniform(0, 2 * np.pi, (N, N))
    Z     = power * np.exp(1j * phase)

    # Ensure Hermitian symmetry for real output
    surface = np.real(np.fft.ifft2(Z))

    # Normalise to zero mean, unit std
    surface = surface - surface.mean()
    surface = surface / (surface.std() + 1e-10)
    return surface


def generate_fracture_mask(size, H, aperture, orientation, rng):
    """
    Generate 3D fracture void mask using fBm surfaces.
    
    FIXED METHOD:
    1. Generate 2D fBm roughness field (zero mean, small amplitude)
    2. Fracture = thin slab centered at domain mid-point
       Each voxel (i,j): fracture occupies [center-A/2 + rough(i,j),
                                             center+A/2 + rough(i,j)]
    3. Roughness amplitude << aperture to keep fracture thin
    
    Args:
        size:        domain size (cubic)
        H:           Hurst exponent (0.4-0.8)
        aperture:    mean aperture in lattice units (5-15 for 64³)
        orientation: 'x', 'y', or 'z' — fracture plane normal direction
        rng:         numpy random generator
    
    Returns:
        mask: (size, size, size) bool array, True = fracture void
    """
    N      = size
    center = N // 2
    half_A = aperture / 2.0

    # Generate fBm roughness field on fracture plane
    # Amplitude = small fraction of aperture (paper: roughness << aperture)
    roughness = generate_fbm_surface(N, H, rng)
    # Scale roughness to ±(aperture * 0.2) max — keeps fracture thin
    roughness_amp = aperture * 0.2
    roughness = roughness * roughness_amp  # zero-mean displacement

    # Build fracture mask
    mask = np.zeros((N, N, N), dtype=bool)

    if orientation == 'z':
        # Fracture plane is XY, fracture normal is Z
        # For each (x,y) position, fracture occupies a band in Z
        for i in range(N):
            for j in range(N):
                disp  = roughness[i, j]
                z_lo  = int(np.clip(np.round(center - half_A + disp), 0, N-1))
                z_hi  = int(np.clip(np.round(center + half_A + disp), 0, N-1))
                z_lo, z_hi = min(z_lo, z_hi), max(z_lo, z_hi)
                # Ensure at least 1 voxel thick
                if z_hi == z_lo: z_hi = z_lo + 1
                mask[i, j, z_lo:z_hi+1] = True

    elif orientation == 'y':
        # Fracture plane is XZ, fracture normal is Y
        for i in range(N):
            for k in range(N):
                disp  = roughness[i, k]
                y_lo  = int(np.clip(np.round(center - half_A + disp), 0, N-1))
                y_hi  = int(np.clip(np.round(center + half_A + disp), 0, N-1))
                y_lo, y_hi = min(y_lo, y_hi), max(y_lo, y_hi)
                if y_hi == y_lo: y_hi = y_lo + 1
                mask[i, y_lo:y_hi+1, k] = True

    else:  # 'x' — fracture plane is YZ, fracture normal is X
        for j in range(N):
            for k in range(N):
                disp  = roughness[j, k]
                x_lo  = int(np.clip(np.round(center - half_A + disp), 0, N-1))
                x_hi  = int(np.clip(np.round(center + half_A + disp), 0, N-1))
                x_lo, x_hi = min(x_lo, x_hi), max(x_lo, x_hi)
                if x_hi == x_lo: x_hi = x_lo + 1
                mask[x_lo:x_hi+1, j, k] = True

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# COMBINE: porous matrix + fracture
# ─────────────────────────────────────────────────────────────────────────────
def integrate_fracture(pore_binary, fracture_mask):
    """
    Embed fracture into porous matrix.
    Fracture voids override solid voxels.
    
    Args:
        pore_binary:   (N,N,N) float32, 1=pore, 0=solid (binary)
        fracture_mask: (N,N,N) bool, True=fracture void
    
    Returns:
        combined: (N,N,N) float32, 1=pore+fracture, 0=solid
    """
    combined = pore_binary.copy()
    combined[fracture_mask] = 1.0
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SAMPLE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_single_sample(args_tuple):
    """
    Generate one fractured porous geometry sample.
    Returns (sample_id, success, porosity, has_fracture, H, aperture)
    """
    (sample_id, size, lam_min, lam_max,
     H_min, H_max, A_min, A_max,
     frac_prob, output_dir, subset, rng_seed) = args_tuple

    rng = np.random.default_rng(rng_seed)

    for attempt in range(MAX_ATTEMPTS):

        # ── Step 1: Generate porous matrix ───────────────────────────────────
        lam   = float(rng.uniform(lam_min, lam_max))
        shape = (size, size, size)
        Y     = generate_correlated_field(shape, lam, rng=rng)

        T_perc = find_percolation_threshold(Y, axis=0)
        if T_perc is None:
            continue

        T_h    = T_perc + EPSILON * (Y.max() - Y.min())
        binary = (Y <= T_h).astype(np.float32)

        # Porosity check
        porosity = float(binary.mean())
        if porosity < POROSITY_MIN or porosity > POROSITY_MAX:
            continue

        # Percolation check
        if not is_percolating(binary.astype(np.uint8), axis=0):
            continue

        # Throat size check
        edt_matrix = distance_transform_edt(binary)
        if float(edt_matrix.max()) < MIN_THROAT:
            continue

        # ── Step 2: Decide whether to add fracture ────────────────────────────
        has_fracture = rng.random() < frac_prob
        H_used       = None
        A_used       = None

        if has_fracture:
            # Sample fracture parameters from paper range
            H_used = float(rng.uniform(H_min, H_max))
            A_used = int(rng.integers(A_min, A_max + 1))

            # Random orientation (x, y, or z) — paper uses single fracture
            orientation = rng.choice(['x', 'y', 'z'])

            fracture_mask = generate_fracture_mask(
                size, H_used, A_used, orientation, rng)

            # Integrate fracture into matrix
            combined = integrate_fracture(binary, fracture_mask)

            # Post-fracture porosity check
            porosity_frac = float(combined.mean())
            if porosity_frac > 0.90:
                # Fracture made it too open
                continue

            # Verify percolation still holds (fracture helps connectivity)
            if not is_percolating(combined.astype(np.uint8), axis=0):
                continue

            final_geom = combined
        else:
            final_geom = binary

        # ── Step 3: Compute EDT and normalise ─────────────────────────────────
        edt = distance_transform_edt(final_geom)
        edt_max = float(edt.max())
        if edt_max < 1e-6:
            continue
        edt_norm = (edt / edt_max).astype(np.float32)

        # Add channel dimension: (N, N, N, 1)
        edt_norm = edt_norm[..., np.newaxis]

        # ── Step 4: Save ──────────────────────────────────────────────────────
        fname = os.path.join(output_dir, f'{subset}_inputs',
                             f'{sample_id:04d}-geom.npy')
        np.save(fname, edt_norm)

        return (sample_id, True,
                float(final_geom.mean()), has_fracture, H_used, A_used)

    return (sample_id, False, 0.0, False, None, None)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D fractured porous media geometries')
    parser.add_argument('--size',      type=int,   default=64)
    parser.add_argument('--n_total',   type=int,   default=1000)
    parser.add_argument('--train_end', type=int,   default=None,
                        help='Last training sample ID (default: 80%%)')
    parser.add_argument('--val_end',   type=int,   default=None,
                        help='Last validation sample ID (default: 90%%)')
    parser.add_argument('--output_dir', type=str,
                        default='./velML3DfracturedDataset_64')
    parser.add_argument('--lam_min',   type=float, default=LAMBDA_MIN_3D)
    parser.add_argument('--lam_max',   type=float, default=LAMBDA_MAX_3D)
    parser.add_argument('--H_min',     type=float, default=H_MIN_DEFAULT,
                        help='Min Hurst exponent (paper: 0.4)')
    parser.add_argument('--H_max',     type=float, default=H_MAX_DEFAULT,
                        help='Max Hurst exponent (paper: 0.8)')
    parser.add_argument('--A_min',     type=int,   default=A_MIN_DEFAULT,
                        help='Min aperture lattice units (scaled for 64³)')
    parser.add_argument('--A_max',     type=int,   default=A_MAX_DEFAULT,
                        help='Max aperture lattice units')
    parser.add_argument('--frac_prob', type=float, default=FRAC_PROB,
                        help='Fraction of samples with fracture (0-1)')
    parser.add_argument('--n_workers', type=int,
                        default=max(1, cpu_count() - 1))
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    # Splits
    train_end = args.train_end or int(args.n_total * 0.8)
    val_end   = args.val_end   or int(args.n_total * 0.9)

    print(f'Generating {args.n_total} samples at {args.size}³')
    print(f'  Train: 1-{train_end} | Val: {train_end+1}-{val_end} '
          f'| Test: {val_end+1}-{args.n_total}')
    print(f'  Fracture prob: {args.frac_prob*100:.0f}% of samples')
    print(f'  H range: [{args.H_min}, {args.H_max}]')
    print(f'  Aperture range: [{args.A_min}, {args.A_max}] lattice units')
    print(f'  Workers: {args.n_workers}')

    # Create output directories
    out = Path(args.output_dir)
    for subset in ['train', 'validation', 'test']:
        (out / f'{subset}_inputs').mkdir(parents=True, exist_ok=True)
        (out / f'{subset}_outputs').mkdir(parents=True, exist_ok=True)

    # Build task list
    rng_master = np.random.default_rng(args.seed)
    seeds      = rng_master.integers(0, 2**31, size=args.n_total)

    tasks = []
    for sid in range(1, args.n_total + 1):
        if   sid <= train_end: subset = 'train'
        elif sid <= val_end:   subset = 'validation'
        else:                  subset = 'test'

        tasks.append((
            sid, args.size,
            args.lam_min, args.lam_max,
            args.H_min, args.H_max,
            args.A_min, args.A_max,
            args.frac_prob,
            str(out), subset,
            int(seeds[sid - 1])
        ))

    # Run parallel
    success       = 0
    failed        = 0
    n_fractured   = 0
    H_values      = []
    A_values      = []

    with Pool(processes=args.n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(generate_single_sample, tasks),
            total=args.n_total, desc='Generating'
        ):
            sid, ok, por, has_frac, H, A = result
            if ok:
                success += 1
                if has_frac:
                    n_fractured += 1
                    if H is not None: H_values.append(H)
                    if A is not None: A_values.append(A)
            else:
                failed += 1
                print(f'  WARNING: sample {sid:04d} failed after '
                      f'{MAX_ATTEMPTS} attempts')

    # Summary
    print(f'\n=== Generation Complete ===')
    print(f'Success:   {success}/{args.n_total}')
    print(f'Failed:    {failed}')
    print(f'Fractured: {n_fractured}/{success} '
          f'({100*n_fractured/max(success,1):.1f}%)')
    if H_values:
        print(f'H range:   [{min(H_values):.2f}, {max(H_values):.2f}] '
              f'mean={np.mean(H_values):.2f}')
    if A_values:
        print(f'A range:   [{min(A_values)}, {max(A_values)}] '
              f'mean={np.mean(A_values):.1f}')

    # Check file counts
    for subset in ['train', 'validation', 'test']:
        n = len(list((out / f'{subset}_inputs').glob('*.npy')))
        print(f'{subset}_inputs: {n} files')


if __name__ == '__main__':
    main()
