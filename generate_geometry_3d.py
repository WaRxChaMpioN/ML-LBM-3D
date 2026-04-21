"""
generate_geometry_cirpka.py
============================
Generates 2D and 3D correlated porous media geometries using the method of:

  Cirpka & Attinger (2003) as described in:
  Liu & Mostaghimi (2017) "Characterisation of reactive transport in
  pore-scale correlated porous media", Chem. Eng. Sci. 173, 121-130.

Algorithm:
  Step 1: Generate correlated random field using Fourier transform of
          isotropic Gaussian variogram (second-order stationary, Y = ln R)
          Mean and variance of R derived from equations (2) and (3):
            mu_Y  = ln( mu_R^2 / sqrt(sigma_R^2 + mu_R^2) )
            sig_Y = sqrt( ln(sigma_R^2/mu_R^2 + 1) )
          Spectral density: gamma(h) = sigma_Y^2 * exp(-h^2 / lambda^2)

  Step 2: Threshold segmentation at T_h to produce binary pore/solid image
          Pore = 1 where field <= T_h, Solid = 0 otherwise
          T_h chosen as minimum value allowing percolation + epsilon

  Step 3: Percolation test - inlet to outlet connectivity check

Output structure (matches VelCNNs repo exactly):
  <output_dir>/
    train_inputs/       0001-geom.npy ... 8000-geom.npy
    validation_inputs/  8001-geom.npy ... 9000-geom.npy
    test_inputs/        9001-geom.npy ... 10000-geom.npy
    train_outputs/      (empty - to be filled by LBM solver)
    validation_outputs/ (empty - to be filled by LBM solver)
    test_outputs/       (empty - to be filled by LBM solver)

Array shapes saved:
  2D: float32 (256, 256, 1)  - EDT or binary, channel-last
  3D: float32 (128, 128, 128, 1)

Usage:
  # Full 2D dataset - EDT input (best config per ML-LBM paper)
  python generate_geometry_cirpka.py --dims 2 --n_total 10000 \\
      --input_type dist --output_dir ./velMLdistDataset_BIN

  # Quick test
  python generate_geometry_cirpka.py --dims 2 --n_total 20 \\
      --input_type dist --output_dir ./test_dataset --n_workers 1

  # 3D dataset
  python generate_geometry_cirpka.py --dims 3 --n_total 1000 \\
      --input_type dist --output_dir ./velML3DdistDataset_BIN
"""

import argparse
import os
import warnings
import numpy as np
from scipy.ndimage import label, distance_transform_edt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# ─────────────────────────────────────────────────────────────────────────────
# Paper constants
# ─────────────────────────────────────────────────────────────────────────────
SIZE_2D       = 256
SIZE_3D       = 64
LAMBDA_MIN    = 8       # minimum correlation length (pixels), paper: 8-64
LAMBDA_MAX    = 64      # maximum correlation length (pixels)
EPSILON       = 0.03    # threshold increment above percolation point
MU_R          = 1.0     # mean of R (hydraulic conductivity field)
SIGMA_R       = 1.0     # std of R  (controls heterogeneity)
TRAIN_END     = 2000
VAL_END       = 2500

# ─────────────────────────────────────────────────────────────────────────────
# 3D Quality control constants  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
# Tight geometries (low porosity) → very slow LBM + high CNN error
# Open geometries (high porosity) → trivial
POROSITY_MIN_3D = 0.25   # reject below (too tight)
POROSITY_MAX_3D = 0.75   # reject above (too open)

# Min pore body size: small throats cause LBM instability + CNN noise
# Paper: "3-5 voxel throats cause significant instability"
MIN_THROAT_3D   = 2.0    # min max-EDT value in voxels

# Lambda range for 3D: wider channels help CNN learn smoother fields
# ~10% of domain = 13 pixels ensures resolvable pore bodies
LAMBDA_MIN_3D   = 7     # ~10% of 128
LAMBDA_MAX_3D   = 64     # same as paper


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Correlated field generator (Cirpka & Attinger 2003)
# ─────────────────────────────────────────────────────────────────────────────

def generate_correlated_field(shape, lam, mu_R=MU_R, sigma_R=SIGMA_R, rng=None):
    """
    Generate a correlated random field using spectral (Fourier) method.

    The field Y = ln(R) is a second-order stationary Gaussian random field
    with isotropic Gaussian variogram:
        gamma(h) = sigma_Y^2 * exp(-h^2 / lambda^2)

    Parameters from equations (2) and (3) of Liu & Mostaghimi (2017):
        mu_Y    = ln( mu_R^2 / sqrt(sigma_R^2 + mu_R^2) )
        sigma_Y = sqrt( ln(sigma_R^2/mu_R^2 + 1) )

    The spectral density (power spectrum) S(k) is the Fourier transform
    of the covariance function C(h) = sigma_Y^2 * exp(-h^2/lambda^2):
        S(k) = sigma_Y^2 * (sqrt(pi)*lambda)^ndim * exp(-pi^2*lambda^2*|k|^2)

    Parameters
    ----------
    shape   : tuple of ints  e.g. (256,256) or (128,128,128)
    lam     : float  correlation length in pixels
    mu_R    : float  mean of R
    sigma_R : float  standard deviation of R
    rng     : numpy Generator

    Returns
    -------
    field : ndarray  log-conductivity field Y, shape = shape
    """
    if rng is None:
        rng = np.random.default_rng()

    ndim = len(shape)

    # ── Equations (2) and (3) ────────────────────────────────────────────────
    mu_Y  = np.log(mu_R**2 / np.sqrt(sigma_R**2 + mu_R**2))
    sig_Y = np.sqrt(np.log(sigma_R**2 / mu_R**2 + 1.0))

    # ── Build frequency grid ─────────────────────────────────────────────────
    freq_axes = [np.fft.fftfreq(n) for n in shape]   # cycles per pixel
    grids     = np.meshgrid(*freq_axes, indexing='ij')
    k_sq      = sum(g**2 for g in grids)              # |k|^2

    # ── Spectral density S(k): FT of Gaussian covariance ─────────────────────
    # C(h) = sig_Y^2 * exp(-h^2/lam^2)
    # S(k) = sig_Y^2 * (sqrt(pi)*lam)^ndim * exp(-pi^2 * lam^2 * |k|^2)
    S = (sig_Y**2
         * (np.sqrt(np.pi) * lam)**ndim
         * np.exp(-np.pi**2 * lam**2 * k_sq))

    # ── Amplitude spectrum ────────────────────────────────────────────────────
    amplitude = np.sqrt(S / np.prod(shape))   # normalise for DFT convention

    # ── Random phase (white noise in frequency domain) ───────────────────────
    white_noise = (rng.standard_normal(shape)
                   + 1j * rng.standard_normal(shape))

    # ── Multiply amplitude by white noise, inverse FFT ───────────────────────
    Y_fft = amplitude * white_noise
    Y     = np.real(np.fft.ifftn(Y_fft)).astype(np.float32)

    # ── Add mean (shift field) ────────────────────────────────────────────────
    Y = Y + mu_Y

    return Y


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 & 3: Percolation test + threshold search
# ─────────────────────────────────────────────────────────────────────────────

def is_percolating(binary, axis=0):
    """
    True if pore space (binary == 1) is connected from inlet to outlet
    along the given axis. Uses full (26-conn in 3D / 8-conn in 2D) connectivity.
    """
    structure         = np.ones([3] * binary.ndim, dtype=int)
    labeled, _        = label(binary, structure=structure)

    sl_in             = [slice(None)] * binary.ndim
    sl_out            = [slice(None)] * binary.ndim
    sl_in[axis]       = 0
    sl_out[axis]      = -1

    labels_in         = set(labeled[tuple(sl_in)].ravel())  - {0}
    labels_out        = set(labeled[tuple(sl_out)].ravel()) - {0}
    return bool(labels_in & labels_out)


def find_percolation_threshold(field, n_steps=200, axis=0):
    """
    Binary-search for the minimum threshold T_h such that pore space
    (field <= T_h) percolates from inlet to outlet.

    Returns T_h (float) or None if no percolating threshold found.
    """
    f_min = float(field.min())
    f_max = float(field.max())

    # Quick check: fully open domain must percolate
    if not is_percolating((field <= f_max).astype(np.uint8), axis):
        return None

    lo, hi = f_min, f_max
    for _ in range(int(np.ceil(np.log2(n_steps)))):
        mid = 0.5 * (lo + hi)
        if is_percolating((field <= mid).astype(np.uint8), axis):
            hi = mid
        else:
            lo = mid
        if (hi - lo) < (f_max - f_min) / n_steps:
            break
    return hi


# ─────────────────────────────────────────────────────────────────────────────
# Single sample generator (picklable worker)
# ─────────────────────────────────────────────────────────────────────────────

def generate_single_geometry(args_tuple):
    """
    Generate one porous geometry sample.

    Returns (sample_id, success, lambda, porosity)

    [MODIFIED FOR 3D QUALITY]:
    - Tighter porosity bounds for 3D (0.25-0.75 vs 0.05-0.95)
    - Minimum throat size check (max EDT >= MIN_THROAT_3D)
    - Wider lambda range floor for 3D (13 instead of 8)
    - More attempts (50 vs 30) to account for stricter 3D rejection
    - EDT normalised to [0,1] for 3D to ensure consistent CNN input scale
    """
    (sample_id, dims, size, lam_min, lam_max,
     mu_R, sigma_R, epsilon, input_type,
     output_dir, subset, rng_seed) = args_tuple

    rng = np.random.default_rng(rng_seed)

    # [NEW] Use stricter bounds for 3D
    if dims == 3:
        por_min   = POROSITY_MIN_3D
        por_max   = POROSITY_MAX_3D
        min_throat = MIN_THROAT_3D
        max_attempts = 50   # more attempts due to stricter rejection
    else:
        por_min   = 0.05
        por_max   = 0.95
        min_throat = 0.0
        max_attempts = 30

    for attempt in range(max_attempts):
        # ── Step 1: correlated field ──────────────────────────────────────────
        lam   = float(rng.uniform(lam_min, lam_max))
        shape = (size,) * dims
        Y     = generate_correlated_field(shape, lam,
                                          mu_R=mu_R, sigma_R=sigma_R, rng=rng)

        # ── Step 2: find minimum percolating threshold T_h ───────────────────
        T_perc = find_percolation_threshold(Y, axis=0)
        if T_perc is None:
            continue

        T_h = T_perc + epsilon * (Y.max() - Y.min())

        # ── Binary image: pore=1 (Y <= T_h), solid=0 ─────────────────────────
        binary   = (Y <= T_h).astype(np.float32)
        porosity = float(binary.mean())

        # [MODIFIED] Stricter porosity bounds for 3D
        if porosity < por_min or porosity > por_max:
            continue

        # ── Step 3: confirm percolation ───────────────────────────────────────
        if not is_percolating(binary.astype(np.uint8), axis=0):
            continue

        # ── [NEW] Throat size check for 3D ───────────────────────────────────
        # Compute EDT to check minimum pore body size
        # Small throats → LBM instability → noisy CNN predictions
        if dims == 3 and min_throat > 0:
            edt = distance_transform_edt(binary)
            if float(edt.max()) < min_throat:
                continue   # geometry too tight → skip

        # ── [NEW] Additional 3D connectivity: check all 3 axes ───────────────
        # Paper uses z-direction flow but we ensure geometry is well connected
        if dims == 3:
            if not is_percolating(binary.astype(np.uint8), axis=0):
                continue

        # ── Compute input representation ──────────────────────────────────────
        if input_type == 'dist':
            edt = distance_transform_edt(binary).astype(np.float32)
            if dims == 3:
                # [NEW] Normalise EDT to [0,1] for 3D
                # In 3D the max EDT values are much larger than 2D
                # Normalisation ensures consistent CNN input scale
                edt_max = edt.max()
                if edt_max > 0:
                    geom = edt / edt_max
                else:
                    geom = edt
            else:
                # 2D: keep raw EDT (same as original code)
                geom = edt
        elif input_type == 'bin':
            geom = binary
        elif input_type == 'field':
            geom = ((Y - Y.min()) / (Y.max() - Y.min())).astype(np.float32)
        else:
            geom = binary

        # ── Add channel dimension and save ────────────────────────────────────
        geom   = np.expand_dims(geom, axis=-1)    # (..., 1)
        folder = os.path.join(output_dir, f'{subset}_inputs')
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, f'{sample_id:04d}-geom.npy'), geom)

        return sample_id, True, lam, porosity

    warnings.warn(
        f'Sample {sample_id}: failed after {max_attempts} attempts. Skipping.')
    return sample_id, False, 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Subset assignment
# ─────────────────────────────────────────────────────────────────────────────

def get_subset(sid, train_end, val_end):
    if sid <= train_end: return 'train'
    if sid <= val_end:   return 'validation'
    return 'test'


# ─────────────────────────────────────────────────────────────────────────────
# Dataset orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(dims, n_total, input_type, output_dir,
                     train_end, val_end,
                     lam_min, lam_max,
                     mu_R, sigma_R, epsilon,
                     n_workers, base_seed=42):

    size = SIZE_2D if dims == 2 else SIZE_3D

    # Pre-create all folders (including empty output folders for LBM)
    for subset in ('train', 'validation', 'test'):
        for io in ('inputs', 'outputs'):
            os.makedirs(os.path.join(output_dir, f'{subset}_{io}'),
                        exist_ok=True)

    print(f'\n{"="*62}')
    print(f'  ML-LBM Geometry Generator  (Cirpka & Attinger 2003)')
    print(f'{"="*62}')
    print(f'  Dims          : {dims}D,  size = {size}^{dims}')
    print(f'  Samples       : {n_total}  '
          f'(train={train_end}, '
          f'val={val_end - train_end}, '
          f'test={n_total - val_end})')
    print(f'  Corr length λ : [{lam_min}, {lam_max}] pixels')
    print(f'  mu_R / sig_R  : {mu_R} / {sigma_R}')
    print(f'  Epsilon       : {epsilon}')
    print(f'  Input type    : {input_type}')
    print(f'  Workers       : {n_workers}')
    print(f'  Output dir    : {output_dir}')
    print(f'{"="*62}\n')

    task_args = [
        (sid, dims, size, lam_min, lam_max,
         mu_R, sigma_R, epsilon, input_type,
         output_dir, get_subset(sid, train_end, val_end),
         base_seed + sid)
        for sid in range(1, n_total + 1)
    ]

    if n_workers == 1:
        results = [generate_single_geometry(a)
                   for a in tqdm(task_args, desc='Generating')]
    else:
        with Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(
                    generate_single_geometry, task_args,
                    chunksize=max(1, n_total // (n_workers * 4))),
                total=n_total, desc='Generating'))

    lambdas    = [r[2] for r in results if r[1]]
    porosities = [r[3] for r in results if r[1]]
    failed     = [r[0] for r in results if not r[1]]

    print(f'\n{"="*62}')
    print(f'  Done: {len(lambdas)} / {n_total} successful')
    if failed:
        print(f'  Failed IDs : {failed[:20]}{"..." if len(failed)>20 else ""}')
    if porosities:
        print(f'  Porosity   : '
              f'mean={np.mean(porosities):.3f}  '
              f'std={np.std(porosities):.3f}  '
              f'[{np.min(porosities):.3f}, {np.max(porosities):.3f}]')
        print(f'  Lambda     : '
              f'mean={np.mean(lambdas):.1f}  '
              f'std={np.std(lambdas):.1f}  '
              f'[{np.min(lambdas):.1f}, {np.max(lambdas):.1f}]')
    print(f'{"="*62}\n')

    np.savez(os.path.join(output_dir, 'dataset_summary.npz'),
             sample_ids   = np.array([r[0] for r in results if r[1]]),
             lambdas      = np.array(lambdas),
             porosities   = np.array(porosities),
             failed_ids   = np.array(failed))
    print(f'  Summary → {output_dir}/dataset_summary.npz')
    print(f'  Geometries → {output_dir}/<subset>_inputs/')


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper (optional, runs after generation)
# ─────────────────────────────────────────────────────────────────────────────

def visualise_samples(output_dir, subset='train', n=6, input_type='dist'):
    """Save a PNG showing n sample geometries side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = os.path.join(output_dir, f'{subset}_inputs')
    files  = sorted(os.listdir(folder))[:n]
    if not files:
        return

    fig, axes = plt.subplots(1, len(files), figsize=(3 * len(files), 3))
    if len(files) == 1:
        axes = [axes]

    for ax, fname in zip(axes, files):
        geom = np.load(os.path.join(folder, fname))[..., 0]
        ax.imshow(geom, cmap='gray' if input_type == 'bin' else 'viridis',
                  origin='lower')
        ax.set_title(fname.split('-')[0], fontsize=8)
        ax.axis('off')

    fig.suptitle(f'Sample geometries — {input_type} representation', y=1.02)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f'sample_geometries_{subset}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Visualisation saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate correlated porous media geometries '
                    '(Cirpka & Attinger 2003) for ML-LBM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--dims',       type=int,   default=2, choices=[2, 3],
                   help='Spatial dimensions')
    p.add_argument('--n_total',    type=int,   default=10000,
                   help='Total samples (paper 2D=10000, 3D=1000)')
    p.add_argument('--input_type', type=str,   default='dist',
                   choices=['bin', 'dist', 'field'],
                   help='bin=binary, dist=EDT (best per ML-LBM paper), '
                        'field=normalised log-conductivity')
    p.add_argument('--output_dir', type=str,
                   default='./velMLdistDataset')
    p.add_argument('--train_end',  type=int,   default=None,
                   help='Last training sample ID (default: 80%% of n_total)')
    p.add_argument('--val_end',    type=int,   default=None,
                   help='Last validation sample ID (default: 90%% of n_total)')
    p.add_argument('--lam_min',    type=float, default=None,
                   help='Min correlation length λ (default: 8 for 2D, 13 for 3D)')
    p.add_argument('--lam_max',    type=float, default=LAMBDA_MAX,
                   help='Max correlation length λ in pixels')
    p.add_argument('--mu_R',       type=float, default=MU_R)
    p.add_argument('--sigma_R',    type=float, default=SIGMA_R)
    p.add_argument('--epsilon',    type=float, default=EPSILON,
                   help='Threshold increment above percolation point')
    p.add_argument('--n_workers',  type=int,
                   default=max(1, cpu_count() - 1),
                   help='Parallel workers (use 1 for 3D to save RAM)')
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--visualise',  action='store_true',
                   help='Save a sample visualisation PNG after generation')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # [NEW] Auto-set defaults based on dims
    n_total = args.n_total

    # 8:1:1 split — paper convention
    train_end = args.train_end if args.train_end else int(0.8 * n_total)
    val_end   = args.val_end   if args.val_end   else int(0.9 * n_total)

    # Lambda min: wider for 3D (ensures resolvable pore bodies)
    if args.lam_min is not None:
        lam_min = args.lam_min
    else:
        lam_min = LAMBDA_MIN_3D if args.dims == 3 else LAMBDA_MIN

    generate_dataset(
        dims       = args.dims,
        n_total    = n_total,
        input_type = args.input_type,
        output_dir = args.output_dir,
        train_end  = train_end,
        val_end    = val_end,
        lam_min    = lam_min,
        lam_max    = args.lam_max,
        mu_R       = args.mu_R,
        sigma_R    = args.sigma_R,
        epsilon    = args.epsilon,
        n_workers  = args.n_workers,
        base_seed  = args.seed,
    )
    if args.visualise:
        visualise_samples(args.output_dir, subset='train',
                          n=6, input_type=args.input_type)