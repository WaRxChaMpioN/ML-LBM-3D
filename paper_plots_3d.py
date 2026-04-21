"""
paper_plots_3d.py — 3D analysis with fracture physics (Dwinanda & Dharmawan 2025)
==================================================================================
Plots:
  Fig 12: Sorted accuracy curves + permeability scatter
  Fig 13: 3D paper-style velocity visualisation
  Fig 14: Fracture physics analysis (NEW)
          - K vs porosity (fractured vs unfractured)
          - K vs tortuosity
          - K vs SSA
          - Permeability distribution by fracture status
          - Aperture vs K
          - Correlation heatmap (paper Fig 11 style)

Usage:
  python3 paper_plots_3d.py \\
      --dataset_dir /beegfs/scratch/kaushal_jha/velML3DfracturedDataset_64 \\
      --cnn_dir .../test_inputs/CNNOutputs-ckpt-23-... \\
      --subset test --start_id 901 --end_id 1000 \\
      --output_dir ./analysis_frac64_ckpt23
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize, to_rgba
import matplotlib.cm as mcm
from scipy import io as sio
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.signal import savgol_filter
from tqdm import tqdm
from collections import deque

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ── Physics ───────────────────────────────────────────────────────────────────
TAU = 0.7
NU  = (TAU - 0.5) / 3.0
FRAC_VAR_THRESHOLD = 0.015  # slice porosity variance threshold for fracture detection

# ── Path helpers ──────────────────────────────────────────────────────────────
def geom_path(d, subset, sid):
    return os.path.join(d, f'{subset}_inputs', f'{sid:04d}-geom.npy')

def lbm_path(d, subset, sid):
    return os.path.join(d, f'{subset}_outputs', f'{sid:04d}-vels.npy')

def find_cnn_file(cnn_dir, sid):
    for pat in [f'{sid:04d}-geom-pred.mat', f'{sid:04d}-pred.mat']:
        c = os.path.join(cnn_dir, pat)
        if os.path.exists(c): return c
    return None

def load_cnn_3d(path):
    mat = sio.loadmat(path)
    if 'velX' in mat:
        return (mat['velX'].astype(np.float32),
                mat['velY'].astype(np.float32),
                mat['velZ'].astype(np.float32))
    raise IOError(f'Cannot load {path}')

def enforce_solid_3d(ux, uy, uz, pore):
    ux=ux.copy(); uy=uy.copy(); uz=uz.copy()
    ux[~pore]=0; uy[~pore]=0; uz[~pore]=0
    return ux, uy, uz

def smooth_3d(ux, uy, uz, pore, sigma=0.8):
    if sigma <= 0: return ux, uy, uz
    ux_s=gaussian_filter(ux,sigma); ux_s[~pore]=0
    uy_s=gaussian_filter(uy,sigma); uy_s[~pore]=0
    uz_s=gaussian_filter(uz,sigma); uz_s[~pore]=0
    return ux_s, uy_s, uz_s

def smooth_profile(q, window=15, poly=3):
    if len(q) < window: return q
    return savgol_filter(q, window_length=window, polyorder=poly)

def mag3(ux, uy, uz):
    return np.sqrt(ux**2 + uy**2 + uz**2)

def permeability_3d(ux, pore, Nx, dp=1e-5):
    u = float(ux[pore].mean()) if pore.any() else 0.0
    return NU * u * Nx / (dp + 1e-30)

def stafe_3d(ux_l, uy_l, uz_l, ux_c, uy_c, uz_c):
    qx_l=np.sum(ux_l,axis=(1,2)); qx_c=np.sum(ux_c,axis=(1,2))
    qy_l=np.sum(uy_l,axis=(0,2)); qy_c=np.sum(uy_c,axis=(0,2))
    qz_l=np.sum(uz_l,axis=(0,1)); qz_c=np.sum(uz_c,axis=(0,1))
    sx=np.sum(np.abs(qx_l-qx_c))/(np.sum(np.abs(qx_l))+1e-30)
    sy=np.sum(np.abs(qy_l-qy_c))/(np.sum(np.abs(qy_l))+1e-30)
    sz=np.sum(np.abs(qz_l-qz_c))/(np.sum(np.abs(qz_l))+1e-30)
    return float((sx+sy+sz)/3)

# ── Fracture Physics Calculations ─────────────────────────────────────────────
def detect_fracture(pore, threshold=FRAC_VAR_THRESHOLD):
    """
    Detect if sample has a fracture using slice-wise porosity variance.
    Fracture = large planar void → high variance in one slice direction.
    Returns (has_fracture, orientation, max_variance)
    """
    var_x = float(pore.mean(axis=(1,2)).var())
    var_y = float(pore.mean(axis=(0,2)).var())
    var_z = float(pore.mean(axis=(0,1)).var())
    max_var = max(var_x, var_y, var_z)
    if max_var >= threshold:
        orient = ['x','y','z'][[var_x,var_y,var_z].index(max_var)]
        return True, orient, max_var
    return False, None, max_var


def compute_tortuosity(pore, axis=0):
    """
    Geometric tortuosity using BFS shortest path (Dijkstra approximation).
    tau = actual_path_length / straight_line_distance
    """
    N = pore.shape[axis]
    # BFS from inlet slice to outlet slice
    # Use 3D grid BFS with 6-connectivity
    shape = pore.shape

    # inlet = first slice along axis
    if axis == 0:
        inlet  = [(0, j, k) for j in range(shape[1])
                             for k in range(shape[2]) if pore[0, j, k]]
        outlet_check = lambda x,y,z: x == shape[0]-1
    elif axis == 1:
        inlet  = [(i, 0, k) for i in range(shape[0])
                             for k in range(shape[2]) if pore[i, 0, k]]
        outlet_check = lambda x,y,z: y == shape[1]-1
    else:
        inlet  = [(i, j, 0) for i in range(shape[0])
                             for j in range(shape[1]) if pore[i, j, 0]]
        outlet_check = lambda x,y,z: z == shape[2]-1

    if not inlet:
        return float('nan')

    visited  = np.zeros(shape, dtype=bool)
    dist     = np.full(shape, np.inf)
    queue    = deque()

    for node in inlet:
        if not visited[node]:
            visited[node] = True
            dist[node]    = 0
            queue.append(node)

    dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    min_outlet_dist = np.inf

    while queue:
        x, y, z = queue.popleft()
        d = dist[x, y, z]
        if outlet_check(x, y, z):
            min_outlet_dist = min(min_outlet_dist, d)
            continue
        for dx, dy, dz in dirs:
            nx, ny, nz = x+dx, y+dy, z+dz
            if (0 <= nx < shape[0] and 0 <= ny < shape[1] and
                0 <= nz < shape[2] and
                pore[nx, ny, nz] and not visited[nx, ny, nz]):
                visited[nx, ny, nz] = True
                dist[nx, ny, nz]    = d + 1
                queue.append((nx, ny, nz))

    if np.isinf(min_outlet_dist):
        return float('nan')

    straight = float(N - 1)
    return float(min_outlet_dist) / (straight + 1e-10)


def compute_ssa(pore):
    """
    Specific Surface Area using edge detection.
    SSA = number of pore-solid interface voxel faces / total volume
    """
    N = np.prod(pore.shape)
    # Count pore-solid interface faces along each direction
    faces = 0
    faces += np.sum(pore[1:,:,:] != pore[:-1,:,:])
    faces += np.sum(pore[:,1:,:] != pore[:,:-1,:])
    faces += np.sum(pore[:,:,1:] != pore[:,:,:-1])
    return float(faces) / float(N)


def compute_effective_porosity(ux, uy, uz, pore, threshold_pct=10):
    """
    Effective porosity = fraction of pore voxels with non-zero velocity.
    Uses 10th percentile of velocity magnitude as threshold (paper method).
    """
    vel_mag = mag3(ux, uy, uz)
    pore_vels = vel_mag[pore]
    if len(pore_vels) == 0:
        return 0.0
    thresh = np.percentile(pore_vels, threshold_pct)
    effective = (pore_vels > thresh).sum()
    return float(effective) / float(np.prod(pore.shape))


def estimate_aperture(pore, has_fracture, orientation):
    """
    Estimate mean fracture aperture from geometry.
    For fractured samples: find the peak in slice-wise porosity profile.
    """
    if not has_fracture:
        return 0.0
    if orientation == 'x':
        profile = pore.mean(axis=(1,2))
    elif orientation == 'y':
        profile = pore.mean(axis=(0,2))
    else:
        profile = pore.mean(axis=(0,1))
    # Fracture aperture ~ width of peak above background
    baseline = np.percentile(profile, 20)
    above    = profile > baseline * 1.5
    return float(above.sum())


# ── Load all samples ──────────────────────────────────────────────────────────
def load_all(args):
    results = []
    for sid in tqdm(range(args.start_id, args.end_id+1), desc='Loading'):
        gf = geom_path(args.dataset_dir, args.subset, sid)
        lf = lbm_path(args.dataset_dir, args.subset, sid)
        cf = find_cnn_file(args.cnn_dir, sid)
        if not all(os.path.exists(f) for f in [gf, lf]) or cf is None:
            continue

        geom = np.load(gf).astype(np.float32)
        if geom.ndim == 4: geom = geom[..., 0]
        pore = geom > 0.01

        lbm  = np.load(lf).astype(np.float32)
        ux_l, uy_l, uz_l = lbm[...,0], lbm[...,1], lbm[...,2]
        ux_c, uy_c, uz_c = load_cnn_3d(cf)
        ux_c, uy_c, uz_c = enforce_solid_3d(ux_c, uy_c, uz_c, pore)
        ux_c, uy_c, uz_c = smooth_3d(ux_c, uy_c, uz_c, pore, sigma=args.sigma)

        Nx = geom.shape[0]
        Kl = permeability_3d(ux_l, pore, Nx)
        Kc = permeability_3d(ux_c, pore, Nx)
        Kerr = abs(Kl-Kc)/(abs(Kl)+1e-30)
        ml = mag3(ux_l,uy_l,uz_l); mc = mag3(ux_c,uy_c,uz_c)
        mse = float(np.mean((ml-mc)**2))
        rl2 = float(np.sqrt(np.sum((ml[pore]-mc[pore])**2)) /
                    (np.sqrt(np.sum(ml[pore]**2))+1e-30))
        sf  = stafe_3d(ux_l,uy_l,uz_l,ux_c,uy_c,uz_c)

        # ── Fracture physics ──────────────────────────────────────────────────
        has_frac, orient, frac_var = detect_fracture(pore)
        tortuosity = compute_tortuosity(pore, axis=0)
        ssa        = compute_ssa(pore)
        eff_por    = compute_effective_porosity(ux_l, uy_l, uz_l, pore)
        aperture   = estimate_aperture(pore, has_frac, orient)

        results.append(dict(
            sid=sid, geom=geom, pore=pore,
            ux_l=ux_l, uy_l=uy_l, uz_l=uz_l,
            ux_c=ux_c, uy_c=uy_c, uz_c=uz_c,
            Kl=Kl, Kc=Kc, Kerr=Kerr,
            mse=mse, rl2=rl2, stafe=sf,
            porosity=float(pore.mean()),
            eff_porosity=eff_por,
            has_fracture=has_frac,
            frac_orientation=orient,
            frac_variance=frac_var,
            tortuosity=tortuosity,
            ssa=ssa,
            aperture=aperture,
        ))
    return results


# ── Fig 12: Accuracy curves ───────────────────────────────────────────────────
def plot_fig12(results, out):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle('3D CNN Accuracy — K4N8 Fractured Dataset', fontsize=13,
                 fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    configs = [
        ('Kerr',  'Permeability Error',                    '#e63946', gs[0,0]),
        ('stafe', 'Scaled Total Absolute Flow Rate Error', '#457b9d', gs[0,1]),
        ('mse',   'Mean Squared Error',                    '#2a9d8f', gs[1,0]),
    ]
    for key, title, col, pos in configs:
        ax   = fig.add_subplot(pos)
        vals = np.sort([r[key] for r in results])
        ax.semilogy(vals, color=col, lw=2.0)
        ax.fill_between(np.arange(len(vals)), vals, alpha=0.12, color=col)
        ax.set_xlabel('Examples (individually sorted)', fontsize=10)
        ax.set_ylabel('Error', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax   = fig.add_subplot(gs[1,1])
    Kl_f = np.array([r['Kl'] for r in results if r['has_fracture'] and r['Kl']>0])
    Kc_f = np.array([r['Kc'] for r in results if r['has_fracture'] and r['Kl']>0])
    Kl_n = np.array([r['Kl'] for r in results if not r['has_fracture'] and r['Kl']>0])
    Kc_n = np.array([r['Kc'] for r in results if not r['has_fracture'] and r['Kl']>0])
    Kl_a = np.array([r['Kl'] for r in results if r['Kl']>0])
    Kc_a = np.array([r['Kc'] for r in results if r['Kl']>0])

    if len(Kl_f) > 0:
        ax.scatter(Kl_f, Kc_f, c='#e63946', s=20, alpha=0.7,
                   edgecolors='none', label=f'Fractured (n={len(Kl_f)})')
    if len(Kl_n) > 0:
        ax.scatter(Kl_n, Kc_n, c='#457b9d', s=20, alpha=0.7,
                   edgecolors='none', label=f'No fracture (n={len(Kl_n)})')
    lims = [min(Kl_a.min(),Kc_a.min())*0.5, max(Kl_a.max(),Kc_a.max())*2]
    ax.plot(lims, lims, 'k--', lw=1.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2 = np.corrcoef(np.log10(Kl_a+1e-30),
                     np.log10(Kc_a+1e-30))[0,1]**2
    ax.set_xlabel('LBM Permeability (D)', fontsize=10)
    ax.set_ylabel('CNN Permeability (D)', fontsize=10)
    ax.set_title('Permeability: Fractured vs Unfractured', fontsize=9)
    ax.legend(fontsize=8)
    ax.text(0.05, 0.92, f'$R^2$ = {r2:.5f}',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat',
                      edgecolor='gray', alpha=0.8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, ls='--', which='both')

    plt.savefig(os.path.join(out, 'fig12_accuracy.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  -> fig12_accuracy.png')


# ── Fig 14: Fracture physics analysis ────────────────────────────────────────
def plot_fracture_physics(results, out):
    """
    Paper-style fracture physics plots matching Dwinanda & Dharmawan (2025).
    """
    frac = [r for r in results if r['has_fracture']]
    nofr = [r for r in results if not r['has_fracture']]

    print(f'  Fractured samples: {len(frac)}, Unfractured: {len(nofr)}')

    # ── Fig 14a: K vs Absolute Porosity (paper Fig 9a style) ─────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('white')
    fig.suptitle('Fracture Physics Analysis — Dwinanda & Dharmawan (2025) Style',
                 fontsize=13, fontweight='bold')

    col_frac = '#e63946'
    col_nofr = '#457b9d'

    def scatter_two(ax, xkey, ykey, xlabel, ylabel, title,
                    xscale='linear', yscale='log'):
        xf = np.array([r[xkey] for r in frac if r[ykey]>0 and not np.isnan(r[xkey])])
        yf = np.array([r[ykey] for r in frac if r[ykey]>0 and not np.isnan(r[xkey])])
        xn = np.array([r[xkey] for r in nofr if r[ykey]>0 and not np.isnan(r[xkey])])
        yn = np.array([r[ykey] for r in nofr if r[ykey]>0 and not np.isnan(r[xkey])])
        if len(xf) > 0:
            ax.scatter(xf, yf, c=col_frac, s=20, alpha=0.7,
                       edgecolors='none', label=f'Fractured (n={len(xf)})')
            if len(xf) > 2:
                corr_f = np.corrcoef(xf, np.log10(yf+1e-30))[0,1]
                ax.annotate(f'r={corr_f:.2f}', xy=(0.05,0.92),
                            xycoords='axes fraction', color=col_frac, fontsize=9)
        if len(xn) > 0:
            ax.scatter(xn, yn, c=col_nofr, s=20, alpha=0.7,
                       edgecolors='none', label=f'No fracture (n={len(xn)})')
            if len(xn) > 2:
                corr_n = np.corrcoef(xn, np.log10(yn+1e-30))[0,1]
                ax.annotate(f'r={corr_n:.2f}', xy=(0.05,0.80),
                            xycoords='axes fraction', color=col_nofr, fontsize=9)
        ax.set_xscale(xscale); ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, framealpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, ls='--')

    scatter_two(axes[0,0], 'porosity',     'Kl',
                'Absolute Porosity $\\phi$',
                'Permeability K (D)',
                'K vs Absolute Porosity')

    scatter_two(axes[0,1], 'eff_porosity', 'Kl',
                'Effective Porosity $\\phi_{eff}$',
                'Permeability K (D)',
                'K vs Effective Porosity')

    scatter_two(axes[0,2], 'tortuosity',   'Kl',
                'Tortuosity $\\tau$',
                'Permeability K (D)',
                'K vs Tortuosity')

    scatter_two(axes[1,0], 'ssa',          'Kl',
                'Specific Surface Area (SSA)',
                'Permeability K (D)',
                'K vs SSA')

    # K distribution — fractured vs unfractured
    ax = axes[1,1]
    Kl_f = np.array([r['Kl'] for r in frac if r['Kl']>0])
    Kl_n = np.array([r['Kl'] for r in nofr if r['Kl']>0])
    bins = np.logspace(np.log10(1e-6), np.log10(1e-1), 30)
    if len(Kl_f) > 0:
        ax.hist(Kl_f, bins=bins, color=col_frac, alpha=0.6,
                label=f'Fractured (n={len(Kl_f)})', edgecolor='white')
    if len(Kl_n) > 0:
        ax.hist(Kl_n, bins=bins, color=col_nofr, alpha=0.6,
                label=f'No fracture (n={len(Kl_n)})', edgecolor='white')
    ax.set_xscale('log')
    ax.set_xlabel('Permeability K (D)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title('Permeability Distribution', fontsize=9)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, ls='--')

    # Aperture vs K (fractured only)
    ax = axes[1,2]
    if len(frac) > 0:
        ap  = np.array([r['aperture'] for r in frac])
        Kf  = np.array([r['Kl']      for r in frac])
        sc  = ax.scatter(ap, Kf, c=ap, cmap='hot', s=20,
                         alpha=0.75, edgecolors='none')
        ax.set_yscale('log')
        plt.colorbar(sc, ax=ax, label='Aperture (voxels)', fraction=0.04)
        if len(ap) > 2:
            corr = np.corrcoef(ap, np.log10(Kf+1e-30))[0,1]
            ax.text(0.05, 0.92, f'r = {corr:.2f}',
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Estimated Aperture (voxels)', fontsize=9)
    ax.set_ylabel('Permeability K (D)', fontsize=9)
    ax.set_title('Aperture vs K (Fractured only)', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, ls='--')

    plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig14_fracture_physics.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  -> fig14_fracture_physics.png')

    # ── Fig 15: Correlation heatmap (paper Fig 11 style) ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle('Parameter Correlation Heatmaps — Fractured vs Unfractured',
                 fontsize=12, fontweight='bold')

    params     = ['porosity', 'eff_porosity', 'ssa', 'tortuosity', 'Kl']
    param_lbls = ['Abs.\nPorosity', 'Eff.\nPorosity', 'SSA',
                  'Tortuosity', 'Permeability']

    for ax, group, title in zip(axes,
                                [nofr, frac],
                                ['No Fracture', 'Fractured']):
        if len(group) < 3:
            ax.set_title(f'{title} (insufficient data)')
            continue

        data = np.array([[r[p] for p in params] for r in group
                         if not any(np.isnan(r[p]) for p in params)])
        if len(data) < 3:
            continue

        # Log-transform permeability for correlation
        data[:, -1] = np.log10(data[:, -1] + 1e-30)

        corr = np.corrcoef(data.T)
        im   = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1,
                         aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(params)))
        ax.set_yticks(range(len(params)))
        ax.set_xticklabels(param_lbls, fontsize=8)
        ax.set_yticklabels(param_lbls, fontsize=8)
        ax.set_title(f'{title} (n={len(data)})', fontsize=10)

        # Annotate with values
        for i in range(len(params)):
            for j in range(len(params)):
                ax.text(j, i, f'{corr[i,j]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if abs(corr[i,j]) > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig15_correlation_heatmap.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  -> fig15_correlation_heatmap.png')


# ── Fig 13: 3D paper-style velocity visualisation ────────────────────────────
def render_paper_style(ax, vel_mag, pore, geom_edt, title,
                       dot_color='#1a5fa0',
                       surface_color='#5bbcbf',
                       elev=25, azim=45,
                       n_dots=5000,
                       smooth_sigma=2.5,
                       max_tris=8000):
    Nx, Ny, Nz = pore.shape
    if HAS_SKIMAGE:
        try:
            geom_smooth = gaussian_filter(geom_edt.astype(float), sigma=smooth_sigma)
            level = geom_smooth.max() * 0.08
            verts, faces, _, _ = marching_cubes(geom_smooth, level=level, step_size=2)
            if len(faces) > max_tris:
                idx = np.random.choice(len(faces), max_tris, replace=False)
                faces = faces[idx]
            centroids   = verts[faces].mean(axis=1)
            azim_r = np.radians(azim); elev_r = np.radians(elev)
            vx = np.cos(elev_r)*np.cos(azim_r)
            vy = np.cos(elev_r)*np.sin(azim_r)
            vz = np.sin(elev_r)
            depth = (centroids[:,0]/Nx*vx + centroids[:,1]/Ny*vy +
                     centroids[:,2]/Nz*vz)
            depth_n = (depth-depth.min())/(depth.max()-depth.min()+1e-10)
            alphas  = 0.08 + 0.27 * depth_n
            r,g,b,_ = to_rgba(surface_color)
            face_colors = np.array([[r,g,b,a] for a in alphas])
            poly = Poly3DCollection(verts[faces], zsort='average')
            poly.set_facecolor(face_colors)
            poly.set_edgecolor('none')
            ax.add_collection3d(poly)
        except Exception as e:
            pass

    px, py, pz = np.where(pore)
    pvals = vel_mag[pore]
    vmax  = pvals.max() if pvals.max() > 0 else 1.0
    weights = pvals / (pvals.sum() + 1e-30)
    n_sample = min(n_dots, len(px))
    idx = np.random.choice(len(px), n_sample, replace=False, p=weights)
    xs, ys, zs, vs = px[idx].astype(float), py[idx].astype(float), pz[idx].astype(float), pvals[idx]
    vn = vs / (vmax + 1e-30)

    azim_r = np.radians(azim); elev_r = np.radians(elev)
    vx = np.cos(elev_r)*np.cos(azim_r); vy = np.cos(elev_r)*np.sin(azim_r); vz_d = np.sin(elev_r)
    dd = xs/Nx*vx + ys/Ny*vy + zs/Nz*vz_d
    dd_n = (dd-dd.min())/(dd.max()-dd.min()+1e-10)
    dot_alpha = np.clip(0.4 + 0.5 * vn * (0.5 + 0.5*dd_n), 0.15, 0.98)
    sizes = 3.0 + 15.0 * (vn**0.5)
    r,g,b,_ = to_rgba(dot_color)
    colors = np.array([[r,g,b,a] for a in dot_alpha])
    ax.scatter(xs, ys, zs, c=colors, s=sizes, depthshade=False, edgecolors='none')

    ax.set_xlim(0, Nx); ax.set_ylim(0, Ny); ax.set_zlim(0, Nz)
    ax.set_xlabel('X-axis', fontsize=8, labelpad=-4)
    ax.set_ylabel('Y-axis', fontsize=8, labelpad=-4)
    ax.set_zlabel('Z-axis', fontsize=8, labelpad=-4)
    ax.tick_params(labelsize=6, pad=-2)
    ax.set_title(title, fontsize=10, pad=5, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1,1,1])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#cccccc'); ax.yaxis.pane.set_edgecolor('#cccccc')
    ax.zaxis.pane.set_edgecolor('#cccccc')
    ax.grid(False); ax.set_facecolor('white')


def plot_fig13(results, out):
    rl2s = np.array([r['rl2'] for r in results])
    idx  = np.argsort(rl2s)
    N    = len(idx)
    picks = [idx[N//4], idx[N//2], idx[3*N//4], idx[-1]]

    Kl_all = np.array([r['Kl'] for r in results])
    Kc_all = np.array([r['Kc'] for r in results])
    mask_k = (Kl_all > 0) & (Kc_all > 0)
    r2_all = np.corrcoef(np.log10(Kl_all[mask_k]+1e-30),
                          np.log10(Kc_all[mask_k]+1e-30))[0,1]**2

    for pick in picks:
        s    = results[pick]
        pore = s['pore']
        ml   = mag3(s['ux_l'], s['uy_l'], s['uz_l'])
        mc   = mag3(s['ux_c'], s['uy_c'], s['uz_c'])
        frac_str = f"Fractured ({s['frac_orientation']})" if s['has_fracture'] else "No fracture"

        fig = plt.figure(figsize=(22, 14))
        fig.patch.set_facecolor('white')
        fig.suptitle(
            f'Sample {s["sid"]}  |  {frac_str}  |  '
            f'$\\phi$={s["porosity"]:.3f}  '
            f'$\\phi_{{eff}}$={s["eff_porosity"]:.3f}  '
            f'$\\tau$={s["tortuosity"]:.2f}  '
            f'SSA={s["ssa"]:.4f}\n'
            f'$K_{{LBM}}$={s["Kl"]:.2e}  '
            f'$K_{{CNN}}$={s["Kc"]:.2e}  '
            f'$\\Delta K$={s["Kerr"]*100:.1f}%  '
            f'RelL2={s["rl2"]:.3f}',
            fontsize=10, fontweight='bold', y=0.98
        )

        gs = gridspec.GridSpec(2, 4, figure=fig,
                               hspace=0.2, wspace=0.05,
                               top=0.91, bottom=0.07,
                               left=0.02, right=0.98)

        ax_lbm = fig.add_subplot(gs[0, 0:2], projection='3d')
        render_paper_style(ax_lbm, ml, pore, s['geom'],
                           f'Real Velocity Field\n$K$={s["Kl"]:.2e}',
                           elev=25, azim=45, n_dots=5000)

        ax_cnn = fig.add_subplot(gs[0, 2:4], projection='3d')
        render_paper_style(ax_cnn, mc, pore, s['geom'],
                           f'CNN Velocity Field\n$K$={s["Kc"]:.2e}',
                           elev=25, azim=45, n_dots=5000)

        flow_data = [
            ('Flow X', np.sum(s['ux_l'],axis=(1,2)), np.sum(s['ux_c'],axis=(1,2)), 'x'),
            ('Flow Y', np.sum(s['uy_l'],axis=(0,2)), np.sum(s['uy_c'],axis=(0,2)), 'y'),
            ('Flow Z', np.sum(s['uz_l'],axis=(0,1)), np.sum(s['uz_c'],axis=(0,1)), 'z'),
        ]
        for col, (flabel, ql, qc, xlabel) in enumerate(flow_data):
            ax = fig.add_subplot(gs[1, col])
            ql_s = smooth_profile(ql); qc_s = smooth_profile(qc)
            x = np.arange(len(ql))
            ax.plot(x, ql, color='#1d6fa4', lw=0.5, alpha=0.2)
            ax.plot(x, ql_s, color='#1d6fa4', lw=2.5, label='LBM')
            ax.plot(x, qc, color='#e05a4c', lw=0.5, alpha=0.2, ls='--')
            ax.plot(x, qc_s, color='#e05a4c', lw=2.5, label='CNN', ls='--')
            ax.axhline(0, color='k', lw=0.7, ls=':', alpha=0.4)
            ax.set_title(f'Flow {xlabel.upper()}  MeanQ={np.mean(np.abs(ql)):.5f}', fontsize=9)
            ax.set_xlabel(f'{xlabel} (voxels)', fontsize=9)
            ax.set_ylabel('Flow Rate', fontsize=9)
            ax.legend(fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, ls='--')

        ax = fig.add_subplot(gs[1, 3])
        fc = ['#e63946' if r['has_fracture'] else '#457b9d' for r in results if mask_k[results.index(r)]]
        ax.scatter(Kl_all[mask_k], Kc_all[mask_k], c='#5e81ac',
                   s=10, alpha=0.5, edgecolors='none')
        ax.scatter([s['Kl']], [s['Kc']], c='#bf616a', s=120, zorder=5, marker='*')
        lims = [min(Kl_all[mask_k].min(),Kc_all[mask_k].min())*0.5,
                max(Kl_all[mask_k].max(),Kc_all[mask_k].max())*2]
        ax.plot(lims, lims, 'k--', lw=1.5, alpha=0.7)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('LBM $K$', fontsize=9)
        ax.set_ylabel('CNN $K$', fontsize=9)
        ax.set_title(f'$R^2$={r2_all:.4f}', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, ls='--', which='both')

        p = os.path.join(out, f'fig13_sample_{s["sid"]}.png')
        plt.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'  -> {p}')


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--cnn_dir',     required=True)
    parser.add_argument('--subset',      default='test')
    parser.add_argument('--start_id',    type=int, default=901)
    parser.add_argument('--end_id',      type=int, default=1000)
    parser.add_argument('--sigma',       type=float, default=0.8)
    parser.add_argument('--output_dir',  default='./analysis_3d')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading results (sigma={args.sigma})...')
    results = load_all(args)
    print(f'Loaded {len(results)} samples')
    n_frac = sum(1 for r in results if r['has_fracture'])
    print(f'Fractured: {n_frac}/{len(results)} ({100*n_frac/max(len(results),1):.1f}%)')

    if len(results) == 0:
        print('No samples found!'); exit(1)

    print('Plotting Fig 12 (accuracy)...'); plot_fig12(results, args.output_dir)
    print('Plotting Fig 13 (3D velocity)...'); plot_fig13(results, args.output_dir)
    print('Plotting Fig 14 (fracture physics)...'); plot_fracture_physics(results, args.output_dir)

    Kerrs = np.array([r['Kerr'] for r in results]) * 100
    rl2s  = np.array([r['rl2']  for r in results])
    Kl    = np.array([r['Kl']   for r in results])
    Kc    = np.array([r['Kc']   for r in results])
    mask  = (Kl > 0) & (Kc > 0)
    r2    = np.corrcoef(np.log10(Kl[mask]+1e-30),
                        np.log10(Kc[mask]+1e-30))[0,1]**2

    Kl_f  = np.array([r['Kl'] for r in results if r['has_fracture'] and r['Kl']>0])
    Kl_n  = np.array([r['Kl'] for r in results if not r['has_fracture'] and r['Kl']>0])

    print(f'\n=== Summary ===')
    print(f'Samples:          {len(results)}')
    print(f'Fractured:        {n_frac} ({100*n_frac/len(results):.1f}%)')
    print(f'R² log K:         {r2:.4f}')
    print(f'Mean K Error:     {np.mean(Kerrs):.2f}%')
    print(f'K < 10%:          {(Kerrs<10).mean()*100:.1f}%')
    print(f'Mean Rel L2:      {np.mean(rl2s):.4f}')
    if len(Kl_f) > 0:
        print(f'Mean K (frac):    {np.mean(Kl_f):.2e} D')
    if len(Kl_n) > 0:
        print(f'Mean K (no frac): {np.mean(Kl_n):.2e} D')
