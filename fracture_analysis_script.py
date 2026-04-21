"""
analyse_fracture_lbm.py — Pure LBM Fracture Physics Analysis
=============================================================
Reproduces analysis from Dwinanda & Dharmawan (2025), Phys. Fluids 37, 093129
Using YOUR LBM simulation results on fractured geometries.

Computes per sample:
  - Absolute porosity
  - Effective porosity (velocity-based, paper method)
  - Permeability (Darcy-LBM)
  - Tortuosity (BFS shortest path, Dijkstra method)
  - Specific Surface Area (SSA, edge detection)
  - Coordination number (watershed-based pore count)
  - Average throat size (EDT-based)
  - Fracture detection + orientation + aperture

Plots (matching paper figures):
  Fig 5:  3D velocity field visualisation (paper style)
  Fig 6:  Parameter distributions by fracture status
  Fig 8:  Effective vs absolute porosity scatter
  Fig 9:  K vs microstructural parameters
  Fig 11: Correlation heatmaps

Usage:
  python3 analyse_fracture_lbm.py \\
      --dataset_dir /beegfs/scratch/kaushal_jha/velML3DfracturedDataset_64 \\
      --subset test --start_id 901 --end_id 1000 \\
      --output_dir ./fracture_analysis_lbm
"""

import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize, to_rgba
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           label as nd_label)
from scipy.signal import savgol_filter
from collections import deque
from tqdm import tqdm

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ── Physics ───────────────────────────────────────────────────────────────────
TAU = 0.7
NU  = (TAU - 0.5) / 3.0
FRAC_VAR_THRESHOLD = 0.015

# ── Paths ─────────────────────────────────────────────────────────────────────
def geom_path(d, subset, sid):
    return os.path.join(d, f'{subset}_inputs',  f'{sid:04d}-geom.npy')

def lbm_path(d, subset, sid):
    return os.path.join(d, f'{subset}_outputs', f'{sid:04d}-vels.npy')

# ── Basic physics ─────────────────────────────────────────────────────────────
def mag3(ux, uy, uz):
    return np.sqrt(ux**2 + uy**2 + uz**2)

def permeability_darcy(ux, pore, Nx, dp=1e-5):
    u = float(ux[pore].mean()) if pore.any() else 0.0
    return NU * u * Nx / (dp + 1e-30)

# ── Fracture detection ────────────────────────────────────────────────────────
def detect_fracture(pore, threshold=FRAC_VAR_THRESHOLD):
    var_x = float(pore.mean(axis=(1,2)).var())
    var_y = float(pore.mean(axis=(0,2)).var())
    var_z = float(pore.mean(axis=(0,1)).var())
    max_var = max(var_x, var_y, var_z)
    if max_var >= threshold:
        orient = ['x','y','z'][[var_x,var_y,var_z].index(max_var)]
        return True, orient, max_var
    return False, None, max_var

def estimate_aperture(pore, has_fracture, orientation):
    if not has_fracture: return 0.0
    if   orientation == 'x': profile = pore.mean(axis=(1,2))
    elif orientation == 'y': profile = pore.mean(axis=(0,2))
    else:                    profile = pore.mean(axis=(0,1))
    baseline = np.percentile(profile, 20)
    return float((profile > baseline * 1.5).sum())

# ── Microstructural parameters ────────────────────────────────────────────────
def compute_absolute_porosity(pore):
    return float(pore.mean())

def compute_effective_porosity(ux, uy, uz, pore, threshold_pct=10):
    """Paper method: 10th percentile of pore velocity as threshold."""
    vel_mag   = mag3(ux, uy, uz)
    pore_vels = vel_mag[pore]
    if len(pore_vels) == 0: return 0.0
    thresh    = np.percentile(pore_vels, threshold_pct)
    return float((pore_vels > thresh).sum()) / float(np.prod(pore.shape))

def compute_tortuosity_bfs(pore, axis=0):
    """
    Geometric tortuosity using BFS (Dijkstra shortest path approximation).
    tau = L_min / L_straight  (paper Eq. 5)
    """
    shape = pore.shape
    N     = shape[axis]

    # Build inlet nodes
    slc = [slice(None)] * 3
    slc[axis] = 0
    inlet_mask = pore[tuple(slc)]
    inlet = []
    for idx in np.argwhere(inlet_mask):
        node = [0, 0, 0]
        axes_other = [i for i in range(3) if i != axis]
        node[axis] = 0
        node[axes_other[0]] = idx[0]
        node[axes_other[1]] = idx[1]
        inlet.append(tuple(node))

    if not inlet: return np.nan

    visited = np.zeros(shape, dtype=bool)
    dist    = np.full(shape, np.inf)
    queue   = deque()

    for node in inlet:
        if not visited[node]:
            visited[node] = True
            dist[node]    = 0
            queue.append(node)

    dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    min_outlet = np.inf

    while queue:
        x, y, z = queue.popleft()
        d = dist[x, y, z]
        # Check outlet
        node_arr = [x, y, z]
        if node_arr[axis] == shape[axis] - 1:
            min_outlet = min(min_outlet, d)
            continue
        for dx, dy, dz in dirs:
            nx, ny, nz = x+dx, y+dy, z+dz
            if (0 <= nx < shape[0] and 0 <= ny < shape[1] and
                0 <= nz < shape[2] and
                pore[nx,ny,nz] and not visited[nx,ny,nz]):
                visited[nx,ny,nz] = True
                dist[nx,ny,nz]    = d + 1
                queue.append((nx, ny, nz))

    if np.isinf(min_outlet): return np.nan
    return float(min_outlet) / float(N - 1)

def compute_ssa(pore):
    """SSA = pore-solid interface faces / total volume (paper Eq. Canny method)."""
    N     = float(np.prod(pore.shape))
    faces = 0
    faces += int(np.sum(pore[1:,:,:] != pore[:-1,:,:]))
    faces += int(np.sum(pore[:,1:,:] != pore[:,:-1,:]))
    faces += int(np.sum(pore[:,:,1:] != pore[:,:,:-1]))
    return faces / N

def compute_throat_size_edt(pore):
    """
    Average throat size using EDT.
    Throat = local minimum in EDT (constriction between pore bodies).
    Returns mean throat radius in voxels.
    """
    edt = distance_transform_edt(pore)
    # Throat size = mean of lower quartile of non-zero EDT values
    pore_edt = edt[pore]
    if len(pore_edt) == 0: return 0.0
    return float(np.percentile(pore_edt, 25))

def compute_coordination_number(pore):
    """
    Coordination number = mean number of connections per pore cluster.
    Estimated from connected component labeling.
    """
    labeled, n_components = nd_label(pore)
    if n_components == 0: return 0.0
    # Count neighbours between different labelled regions
    # Simplified: use pore connectivity as proxy
    # coordination ~ 2 * bonds / components
    bonds = 0
    bonds += int(np.sum((labeled[1:,:,:] > 0) & (labeled[:-1,:,:] > 0) &
                        (labeled[1:,:,:] != labeled[:-1,:,:])))
    bonds += int(np.sum((labeled[:,1:,:] > 0) & (labeled[:,:-1,:] > 0) &
                        (labeled[:,1:,:] != labeled[:,:-1,:])))
    bonds += int(np.sum((labeled[:,:,1:] > 0) & (labeled[:,:,:-1] > 0) &
                        (labeled[:,:,1:] != labeled[:,:,:-1])))
    if n_components == 0: return 0.0
    return float(2 * bonds) / float(n_components)

# ── Load all samples ──────────────────────────────────────────────────────────
def load_all(args):
    results = []
    for sid in tqdm(range(args.start_id, args.end_id+1), desc='Loading'):
        gf = geom_path(args.dataset_dir, args.subset, sid)
        lf = lbm_path(args.dataset_dir, args.subset, sid)
        if not os.path.exists(gf) or not os.path.exists(lf):
            continue

        geom = np.load(gf).astype(np.float32)
        if geom.ndim == 4: geom = geom[..., 0]
        pore = geom > 0.01

        lbm  = np.load(lf).astype(np.float32)
        ux, uy, uz = lbm[...,0], lbm[...,1], lbm[...,2]

        Nx = geom.shape[0]
        K  = permeability_darcy(ux, pore, Nx)

        # Fracture detection
        has_frac, orient, frac_var = detect_fracture(pore)
        aperture = estimate_aperture(pore, has_frac, orient)

        # Microstructural parameters (paper method)
        abs_por  = compute_absolute_porosity(pore)
        eff_por  = compute_effective_porosity(ux, uy, uz, pore)
        tort     = compute_tortuosity_bfs(pore, axis=0)
        ssa      = compute_ssa(pore)
        throat   = compute_throat_size_edt(pore)
        coord    = compute_coordination_number(pore)

        results.append(dict(
            sid=sid, geom=geom, pore=pore,
            ux=ux, uy=uy, uz=uz,
            K=K,
            abs_porosity=abs_por,
            eff_porosity=eff_por,
            tortuosity=tort,
            ssa=ssa,
            throat_size=throat,
            coord_number=coord,
            has_fracture=has_frac,
            frac_orientation=orient,
            frac_variance=frac_var,
            aperture=aperture,
        ))
    return results

# ── 3D paper-style visualisation ──────────────────────────────────────────────
def render_lbm_volume(ax, vel_mag, pore, geom, title,
                      dot_color='#1a5fa0', surface_color='#5bbcbf',
                      elev=25, azim=45, n_dots=5000,
                      smooth_sigma=2.5, max_tris=8000):
    Nx, Ny, Nz = pore.shape
    if HAS_SKIMAGE:
        try:
            gs = gaussian_filter(geom.astype(float), sigma=smooth_sigma)
            verts, faces, _, _ = marching_cubes(
                gs, level=gs.max()*0.08, step_size=2)
            if len(faces) > max_tris:
                idx = np.random.choice(len(faces), max_tris, replace=False)
                faces = faces[idx]
            cen = verts[faces].mean(axis=1)
            ar  = np.radians(azim); er = np.radians(elev)
            vx  = np.cos(er)*np.cos(ar); vy = np.cos(er)*np.sin(ar)
            vz  = np.sin(er)
            dep = cen[:,0]/Nx*vx + cen[:,1]/Ny*vy + cen[:,2]/Nz*vz
            dn  = (dep-dep.min())/(dep.max()-dep.min()+1e-10)
            alp = 0.08 + 0.27*dn
            r,g,b,_ = to_rgba(surface_color)
            fc  = np.array([[r,g,b,a] for a in alp])
            poly = Poly3DCollection(verts[faces], zsort='average')
            poly.set_facecolor(fc); poly.set_edgecolor('none')
            ax.add_collection3d(poly)
        except: pass

    px, py, pz = np.where(pore)
    pv = vel_mag[pore]
    vm = pv.max() if pv.max() > 0 else 1.0
    w  = pv / (pv.sum()+1e-30)
    ns = min(n_dots, len(px))
    i  = np.random.choice(len(px), ns, replace=False, p=w)
    xs,ys,zs,vs = px[i].astype(float),py[i].astype(float),pz[i].astype(float),pv[i]
    vn = vs/(vm+1e-30)
    ar = np.radians(azim); er = np.radians(elev)
    vx = np.cos(er)*np.cos(ar); vy = np.cos(er)*np.sin(ar); vz2 = np.sin(er)
    dd = xs/Nx*vx+ys/Ny*vy+zs/Nz*vz2
    dn = (dd-dd.min())/(dd.max()-dd.min()+1e-10)
    da = np.clip(0.4+0.5*vn*(0.5+0.5*dn), 0.15, 0.98)
    sz = 3.0+15.0*(vn**0.5)
    r,g,b,_ = to_rgba(dot_color)
    col = np.array([[r,g,b,a] for a in da])
    ax.scatter(xs,ys,zs,c=col,s=sz,depthshade=False,edgecolors='none')

    ax.set_xlim(0,Nx); ax.set_ylim(0,Ny); ax.set_zlim(0,Nz)
    ax.set_xlabel('X',fontsize=8,labelpad=-4)
    ax.set_ylabel('Y',fontsize=8,labelpad=-4)
    ax.set_zlabel('Z',fontsize=8,labelpad=-4)
    ax.tick_params(labelsize=6,pad=-2)
    ax.set_title(title,fontsize=10,pad=5,fontweight='bold')
    ax.view_init(elev=elev,azim=azim)
    ax.set_box_aspect([1,1,1])
    for p in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        p.fill=False; p.set_edgecolor('#cccccc')
    ax.grid(False); ax.set_facecolor('white')

# ── Fig 5: Velocity fields (paper style) ─────────────────────────────────────
def plot_fig5(results, out):
    """
    Paper Fig 5 style: show velocity fields for
    no-fracture, fractured (low H), fractured (high H)
    """
    nofrac = [r for r in results if not r['has_fracture']]
    frac   = sorted([r for r in results if r['has_fracture']],
                    key=lambda r: r['frac_variance'])

    # Pick representative samples
    picks = []
    if nofrac: picks.append(('No Fracture', nofrac[len(nofrac)//2]))
    if len(frac) >= 2:
        picks.append(('Fractured (low var)', frac[0]))
        picks.append(('Fractured (high var)', frac[-1]))

    if not picks:
        print('  Not enough samples for Fig 5')
        return

    fig = plt.figure(figsize=(7*len(picks), 7))
    fig.patch.set_facecolor('white')
    fig.suptitle('LBM Velocity Fields — Fractured vs Unfractured Porous Media',
                 fontsize=12, fontweight='bold')

    for col, (label, r) in enumerate(picks):
        vel = mag3(r['ux'], r['uy'], r['uz'])
        ax  = fig.add_subplot(1, len(picks), col+1, projection='3d')
        frac_str = (f"H-proxy={r['frac_variance']:.3f} "
                    f"A≈{r['aperture']:.0f}vox"
                    if r['has_fracture'] else '')
        title = (f'{label}\n'
                 f'$\\phi$={r["abs_porosity"]:.3f}  '
                 f'K={r["K"]:.2e} D\n{frac_str}')
        render_lbm_volume(ax, vel, r['pore'], r['geom'], title,
                          n_dots=4000, smooth_sigma=2.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig5_velocity_fields.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  -> fig5_velocity_fields.png')

# ── Fig 6: Parameter distributions ───────────────────────────────────────────
def plot_fig6(results, out):
    """
    Paper Fig 6 style: distributions of 6 parameters
    split by fracture status
    """
    frac  = [r for r in results if r['has_fracture']]
    nofrac= [r for r in results if not r['has_fracture']]

    params = [
        ('K',            'Permeability K (D)',          True),
        ('abs_porosity', 'Absolute Porosity $\\phi$',   False),
        ('eff_porosity', 'Effective Porosity $\\phi_{eff}$', False),
        ('tortuosity',   'Tortuosity $\\tau$',           False),
        ('ssa',          'Specific Surface Area',        False),
        ('throat_size',  'Throat Size (voxels)',         False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle('Parameter Distributions — Fractured vs Unfractured\n'
                 '(Dwinanda & Dharmawan 2025 Fig 6 Style)',
                 fontsize=12, fontweight='bold')

    col_f = '#e63946'
    col_n = '#457b9d'

    for ax, (key, xlabel, log_x) in zip(axes.flat, params):
        vf = np.array([r[key] for r in frac
                       if not np.isnan(r[key]) and r[key] > 0])
        vn = np.array([r[key] for r in nofrac
                       if not np.isnan(r[key]) and r[key] > 0])

        if log_x:
            all_v = np.concatenate([vf, vn]) if len(vf)+len(vn) > 0 else [1e-10]
            bins  = np.logspace(np.log10(max(all_v.min(), 1e-10)),
                                np.log10(all_v.max()+1e-10), 25)
        else:
            all_v = np.concatenate([vf, vn]) if len(vf)+len(vn) > 0 else [0,1]
            bins  = np.linspace(all_v.min(), all_v.max(), 25)

        if len(vn) > 0:
            ax.hist(vn, bins=bins, color=col_n, alpha=0.6,
                    label=f'No fracture (n={len(vn)})',
                    edgecolor='white', linewidth=0.5)
        if len(vf) > 0:
            ax.hist(vf, bins=bins, color=col_f, alpha=0.6,
                    label=f'Fractured (n={len(vf)})',
                    edgecolor='white', linewidth=0.5)

        if log_x: ax.set_xscale('log')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(xlabel, fontsize=9)
        ax.legend(fontsize=7, framealpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, ls='--', axis='y')

        # Add mean lines
        if len(vf) > 0:
            ax.axvline(np.mean(vf), color=col_f, ls='--', lw=1.5,
                       alpha=0.8)
        if len(vn) > 0:
            ax.axvline(np.mean(vn), color=col_n, ls='--', lw=1.5,
                       alpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig6_distributions.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  -> fig6_distributions.png')

# ── Fig 8: Effective vs absolute porosity ─────────────────────────────────────
def plot_fig8(results, out):
    """Paper Fig 8 style: effective vs absolute porosity scatter."""
    frac  = [r for r in results if r['has_fracture']]
    nofrac= [r for r in results if not r['has_fracture']]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Effective Porosity vs Absolute Porosity\n'
                 '(Dwinanda & Dharmawan 2025 Fig 8 Style)',
                 fontsize=12, fontweight='bold')

    for ax, group, title, col in zip(
        axes,
        [nofrac, frac],
        ['No Fracture', 'Fractured'],
        ['#457b9d', '#e63946']
    ):
        if len(group) < 2:
            ax.set_title(f'{title} (insufficient data)')
            continue
        x = np.array([r['abs_porosity'] for r in group])
        y = np.array([r['eff_porosity'] for r in group])
        sc = ax.scatter(x, y, c=col, s=25, alpha=0.7, edgecolors='none')
        corr = np.corrcoef(x, y)[0,1] if len(x) > 2 else 0
        # Fit line
        if len(x) > 2:
            m, b = np.polyfit(x, y, 1)
            xr   = np.linspace(x.min(), x.max(), 50)
            ax.plot(xr, m*xr+b, color=col, lw=2, alpha=0.8)
        ax.set_xlabel('Absolute Porosity $\\phi_{abs}$', fontsize=10)
        ax.set_ylabel('Effective Porosity $\\phi_{eff}$', fontsize=10)
        ax.set_title(f'{title} (n={len(group)})', fontsize=10)
        ax.text(0.05, 0.92, f'Corr = {corr:.2f}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, ls='--')

    plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig8_eff_vs_abs_porosity.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  -> fig8_eff_vs_abs_porosity.png')

# ── Fig 9: K vs microstructural parameters ────────────────────────────────────
def plot_fig9(results, out):
    """Paper Fig 9 style: K vs each microstructural parameter."""
    frac  = [r for r in results if r['has_fracture'] and r['K']>0]
    nofrac= [r for r in results if not r['has_fracture'] and r['K']>0]

    params = [
        ('abs_porosity', 'Absolute Porosity $\\phi$'),
        ('eff_porosity', 'Effective Porosity $\\phi_{eff}$'),
        ('coord_number', 'Coordination Number'),
        ('tortuosity',   'Tortuosity $\\tau$'),
        ('ssa',          'Specific Surface Area'),
        ('throat_size',  'Throat Size (voxels)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle('Permeability vs Microstructural Parameters\n'
                 '(Dwinanda & Dharmawan 2025 Fig 9 Style)',
                 fontsize=12, fontweight='bold')

    col_f = '#e63946'
    col_n = '#457b9d'

    for ax, (key, xlabel) in zip(axes.flat, params):
        for group, col, label in [
            (nofrac, col_n, 'No fracture'),
            (frac,   col_f, 'Fractured')
        ]:
            x = np.array([r[key] for r in group
                          if not np.isnan(r[key])])
            y = np.array([r['K'] for r in group
                          if not np.isnan(r[key])])
            if len(x) < 2: continue
            corr = np.corrcoef(x, np.log10(y+1e-30))[0,1]
            ax.scatter(x, y, c=col, s=15, alpha=0.6,
                       edgecolors='none',
                       label=f'{label} (r={corr:.2f})')

        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Permeability K (D)', fontsize=9)
        ax.set_title(xlabel, fontsize=9)
        ax.legend(fontsize=7, framealpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, ls='--')

    plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig9_K_vs_parameters.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  -> fig9_K_vs_parameters.png')

# ── Fig 11: Correlation heatmaps ─────────────────────────────────────────────
def plot_fig11(results, out):
    """Paper Fig 11 style: correlation heatmaps."""
    keys  = ['abs_porosity','eff_porosity','ssa',
             'tortuosity','coord_number','throat_size','K']
    lbls  = ['Abs.Por','Eff.Por','SSA',
             'Tortuosity','CoordNum','Throat','K']

    frac   = [r for r in results if r['has_fracture']]
    nofrac = [r for r in results if not r['has_fracture']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle('Correlation Matrices — Fractured vs Unfractured\n'
                 '(Dwinanda & Dharmawan 2025 Fig 11 Style)',
                 fontsize=12, fontweight='bold')

    for ax, group, title in zip(
        axes, [nofrac, frac], ['No Fracture', 'Fractured']
    ):
        valid = [r for r in group
                 if all(not np.isnan(r[k]) and r[k] > 0
                        for k in keys)]
        if len(valid) < 3:
            ax.set_title(f'{title} (n={len(valid)}, insufficient)')
            continue

        data = np.array([[r[k] for k in keys] for r in valid])
        # Log-transform K
        data[:, -1] = np.log10(data[:, -1])

        corr = np.corrcoef(data.T)
        im   = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(keys)))
        ax.set_yticks(range(len(keys)))
        ax.set_xticklabels(lbls, fontsize=8, rotation=45, ha='right')
        ax.set_yticklabels(lbls, fontsize=8)
        ax.set_title(f'{title} (n={len(valid)})', fontsize=10)
        for i in range(len(keys)):
            for j in range(len(keys)):
                ax.text(j, i, f'{corr[i,j]:.2f}', ha='center',
                        va='center', fontsize=7,
                        color='white' if abs(corr[i,j])>0.6 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig11_correlation_heatmaps.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  -> fig11_correlation_heatmaps.png')

# ── Summary statistics ────────────────────────────────────────────────────────
def print_summary(results):
    frac  = [r for r in results if r['has_fracture']]
    nofrac= [r for r in results if not r['has_fracture']]

    print(f'\n=== Fracture Analysis Summary ===')
    print(f'Total samples:    {len(results)}')
    print(f'Fractured:        {len(frac)} ({100*len(frac)/len(results):.1f}%)')
    print(f'Unfractured:      {len(nofrac)} ({100*len(nofrac)/len(results):.1f}%)')

    for group, name in [(frac,'Fractured'), (nofrac,'Unfractured')]:
        if not group: continue
        Ks   = np.array([r['K']           for r in group if r['K']>0])
        taus = np.array([r['tortuosity']  for r in group if not np.isnan(r['tortuosity'])])
        phis = np.array([r['abs_porosity'] for r in group])
        print(f'\n{name}:')
        if len(Ks):
            print(f'  K:         mean={np.mean(Ks):.2e} '
                  f'median={np.median(Ks):.2e} D')
        if len(phis):
            print(f'  Porosity:  mean={np.mean(phis):.3f}')
        if len(taus):
            print(f'  Tortuosity:mean={np.mean(taus):.3f}')

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--subset',      default='test')
    parser.add_argument('--start_id',    type=int, default=901)
    parser.add_argument('--end_id',      type=int, default=1000)
    parser.add_argument('--output_dir',  default='./fracture_analysis_lbm')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading LBM results...')
    results = load_all(args)
    print(f'Loaded {len(results)} samples')

    if len(results) == 0:
        print('No samples found!'); exit(1)

    n_frac = sum(1 for r in results if r['has_fracture'])
    print(f'Fractured: {n_frac}/{len(results)}')

    print('Fig 5: velocity fields...');   plot_fig5(results, args.output_dir)
    print('Fig 6: distributions...');     plot_fig6(results, args.output_dir)
    print('Fig 8: porosity scatter...');  plot_fig8(results, args.output_dir)
    print('Fig 9: K vs parameters...');   plot_fig9(results, args.output_dir)
    print('Fig 11: heatmaps...');         plot_fig11(results, args.output_dir)

    print_summary(results)
