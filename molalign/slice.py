#!/usr/bin/env python3
"""
molalign.slice
--------------

Visualize interaction energy as a smoothed 2D slice.

Usage examples:
    slice XY
    slice XZ
    slice YZ
    slice 0 1 2 [--tol 0.2 --sigma 1.0]

Description:
 - Reads all .json or .sjson files in the current directory.
 - Plots either a fixed (XY/XZ/YZ) or arbitrary plane through three solute atoms.
 - Produces a smoothed, adaptive-contrast energy map with realistic atom rendering.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm
import argparse
from matplotlib.patches import Circle
from matplotlib import colors as mcolors

# ============================================================
#                COLOURS, RADII, ELEMENT MAPPING
# ============================================================
CPK_COLORS = {
    "H":  "#D3D3D3",
    "C":  "#555555",
    "N":  "#3050F8",
    "O":  "#E60A0A",
    "F":  "#90E050",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "I":  "#940094",
    "S":  "#FFFF30",
    "P":  "#FF8000",
    "default": "#BBBBBB",
}

CPK_RADII = {
    "H": 0.25, "C": 0.4, "N": 0.35, "O": 0.35, "F": 0.35,
    "S": 0.45, "P": 0.45, "Cl": 0.4, "Br": 0.45, "I": 0.5, "default": 0.4
}

Z_TO_SYMBOL = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
    15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"
}

# ============================================================
#                    DATA LOADING
# ============================================================
def load_grid_and_solute(json_file):
    """Load grid points, interaction energies, and solute atom coordinates."""
    with open(json_file) as f:
        data = json.load(f)

    grid_xyz = np.array(data["molA_grids_xyz"])
    e_solute = data["solute"]["energy_Eh"]
    e_solvent = data["solvent"]["energy_Eh"]

    energies = []
    for i in range(len(grid_xyz)):
        gp_key = f"gp{i}"
        if gp_key not in data:
            energies.append(np.nan)
            continue
        gp_data = data[gp_key]
        min_e = min([gp_data[k]["energy_Eh"] for k in gp_data])
        energies.append((min_e - (e_solute + e_solvent)) * 627.5)
    energies = np.array(energies)

    sol_atoms = data["solute"]["xyz"]
    sol_xyz, sol_elems = [], []
    for a in sol_atoms:
        val = a.get("element")
        try:
            symb = Z_TO_SYMBOL.get(int(val), "X")
        except Exception:
            symb = str(val)
        sol_elems.append(symb)
        sol_xyz.append([a["x"], a["y"], a["z"]])
    sol_xyz = np.array(sol_xyz)
    return grid_xyz, energies, sol_xyz, sol_elems


# ============================================================
#                    PLANE CONSTRUCTION
# ============================================================
def plane_from_atoms(sol_xyz, idx_a, idx_b, idx_c):
    A = sol_xyz[idx_a]
    B = sol_xyz[idx_b]
    C = sol_xyz[idx_c]
    v1 = B - A
    v2 = C - A
    n = np.cross(v1, v2)
    n /= np.linalg.norm(n)
    u = v1 / np.linalg.norm(v1)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return A, n, u, v


def slice_along_arbitrary_plane(grid_xyz, energies, sol_xyz, sol_elems, indices, tol=0.2):
    A, n, u, v = plane_from_atoms(sol_xyz, *indices)
    dist = np.dot(grid_xyz - A, n)
    mask = np.abs(dist) < tol
    points = grid_xyz[mask]
    x = np.dot(points - A, u)
    y = np.dot(points - A, v)
    c = energies[mask]
    sol_dist = np.dot(sol_xyz - A, n)
    near = np.abs(sol_dist) < tol
    sol_local = sol_xyz[near]
    sol_elems_near = [e for e, d in zip(sol_elems, sol_dist) if abs(d) < tol]
    sol_x = np.dot(sol_local - A, u)
    sol_y = np.dot(sol_local - A, v)
    xlabel, ylabel = "u-axis [Å]", "v-axis [Å]"
    return x, y, c, sol_x, sol_y, sol_elems_near, xlabel, ylabel


def get_standard_slice(grid_xyz, energies, sol_xyz, sol_elems, plane="XY", tol=0.1):
    if plane == "XY":
        mid = (grid_xyz[:, 2].min() + grid_xyz[:, 2].max()) / 2
        mask = np.abs(grid_xyz[:, 2] - mid) < tol
        x, y, c = grid_xyz[mask, 0], grid_xyz[mask, 1], energies[mask]
        sol_mask = np.abs(sol_xyz[:, 2] - mid) < tol
        sol_x, sol_y = sol_xyz[sol_mask, 0], sol_xyz[sol_mask, 1]
        elems = [e for e, m in zip(sol_elems, sol_mask) if m]
        xlabel, ylabel = "X [Å]", "Y [Å]"
    elif plane == "XZ":
        mid = (grid_xyz[:, 1].min() + grid_xyz[:, 1].max()) / 2
        mask = np.abs(grid_xyz[:, 1] - mid) < tol
        x, y, c = grid_xyz[mask, 0], grid_xyz[mask, 2], energies[mask]
        sol_mask = np.abs(sol_xyz[:, 1] - mid) < tol
        sol_x, sol_y = sol_xyz[sol_mask, 0], sol_xyz[sol_mask, 2]
        elems = [e for e, m in zip(sol_elems, sol_mask) if m]
        xlabel, ylabel = "X [Å]", "Z [Å]"
    else:
        mid = (grid_xyz[:, 0].min() + grid_xyz[:, 0].max()) / 2
        mask = np.abs(grid_xyz[:, 0] - mid) < tol
        x, y, c = grid_xyz[mask, 1], grid_xyz[mask, 2], energies[mask]
        sol_mask = np.abs(sol_xyz[:, 0] - mid) < tol
        sol_x, sol_y = sol_xyz[sol_mask, 1], sol_xyz[sol_mask, 2]
        elems = [e for e, m in zip(sol_elems, sol_mask) if m]
        xlabel, ylabel = "Y [Å]", "Z [Å]"
    return x, y, c, sol_x, sol_y, elems, xlabel, ylabel


# ============================================================
#                       SHADING FUNCTION
# ============================================================
def draw_sphere_2d(ax, x, y, radius, color, light_dir=(-0.4, 0.6), n_shades=25):
    """Draw a shaded, 3D-looking atom bubble in 2D."""
    base = np.array(mcolors.to_rgb(color))
    lx, ly = light_dir

    for i in range(n_shades):
        f = i / (n_shades - 1)
        # darker near edges
        shade = 0.5 + 0.5 * f
        light = shade * (1 + 0.3 * (lx + ly))
        col = np.clip(base * light, 0, 1)
        circ = Circle(
            (x, y),
            radius * (1 - 0.8 * f),
            facecolor=col,
            edgecolor="none",
            alpha=0.9,
            zorder=12 + f * 0.1,
        )
        ax.add_patch(circ)

    # small white highlight (reflection)
    hl_x = x + radius * (-lx * 0.3)
    hl_y = y + radius * (ly * 0.3)
    highlight = Circle(
        (hl_x, hl_y),
        radius * 0.2,
        facecolor="white",
        edgecolor="none",
        alpha=0.25,
        zorder=13,
    )
    #ax.add_patch(highlight)


# ============================================================
#                        PLOTTING
# ============================================================
def adaptive_color_scale(values, low_pct=1, high_pct=99):
    valid = np.array([v for v in values if not np.isnan(v)])
    vmin = np.percentile(valid, low_pct)
    vmax = np.percentile(valid, high_pct)
    return vmin, vmax


def run_slice(mode_args, tol=0.2, sigma=1.0):
    json_files = sorted(Path(".").glob("*.json")) + sorted(Path(".").glob("*.sjson"))
    if not json_files:
        print("No .json or .sjson files found in this folder.")
        return

    plane_mode = None
    atom_indices = None
    if len(mode_args) == 1 and mode_args[0].upper() in ["XY", "XZ", "YZ"]:
        plane_mode = mode_args[0].upper()
        print(f"Using standard plane: {plane_mode}")
    elif len(mode_args) == 3 and all(a.isdigit() for a in mode_args):
        atom_indices = tuple(map(int, mode_args))
        print(f"Using arbitrary plane defined by atoms {atom_indices}")
    else:
        print("Usage: slice XY/XZ/YZ  or  slice 0 1 2")
        return

    all_x, all_y, all_c = [], [], []
    solute_coords = []
    xlabel = ylabel = ""

    for f in json_files:
        grid_xyz, energies, sol_xyz, sol_elems = load_grid_and_solute(f)
        if plane_mode:
            x, y, c, sol_x, sol_y, elems, xlabel, ylabel = get_standard_slice(
                grid_xyz, energies, sol_xyz, sol_elems, plane_mode, tol
            )
        else:
            x, y, c, sol_x, sol_y, elems, xlabel, ylabel = slice_along_arbitrary_plane(
                grid_xyz, energies, sol_xyz, sol_elems, atom_indices, tol
            )
        all_x.append(x)
        all_y.append(y)
        all_c.append(c)
        solute_coords.append((sol_x, sol_y, elems))

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_c = np.concatenate(all_c)

    eps = 1e-8 * max(np.ptp(all_x), np.ptp(all_y))
    all_x += np.random.normal(0, eps, size=all_x.shape)
    all_y += np.random.normal(0, eps, size=all_y.shape)

    xi = np.linspace(all_x.min(), all_x.max(), 400)
    yi = np.linspace(all_y.min(), all_y.max(), 400)
    X, Y = np.meshgrid(xi, yi)

    try:
        Z = griddata((all_x, all_y), all_c, (X, Y), method="cubic")
    except Exception as e:
        print(f"⚠️ Cubic interpolation failed ({e.__class__.__name__}), using linear instead.")
        Z = griddata((all_x, all_y), all_c, (X, Y), method="linear")

    Z = np.nan_to_num(Z, nan=np.nanmean(all_c))
    Z = gaussian_filter(Z, sigma=sigma)

    vmin, vmax = adaptive_color_scale(all_c)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        Z,
        extent=(xi.min(), xi.max(), yi.min(), yi.max()),
        origin="lower",
        cmap="viridis",
        norm=norm,
        interpolation="bilinear",
    )
    levels = np.linspace(vmin, 0, 10)
    cs = ax.contour(X, Y, Z, colors="k", levels=levels, alpha=0.9, linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    # --- shaded 3D-look atom rendering ---
    for (sol_x, sol_y, elems) in solute_coords:
        for x, y, elem in zip(sol_x, sol_y, elems):
            color = CPK_COLORS.get(elem, CPK_COLORS["default"])
            radius = CPK_RADII.get(elem, CPK_RADII["default"])
            draw_sphere_2d(ax, x, y, radius * 0.55, color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #title = f"{plane_mode} mid-plane" if plane_mode else f"Plane through atoms {atom_indices}"
    #ax.set_title(f"{title}\n(smoothed σ={sigma}, tol={tol})")
    ax.set_aspect("equal", "box")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Interaction Energy [kcal/mol]")
    # --- Force colorbar tick visibility ---
    # Generate about 6 ticks from vmin to vmax


    if hasattr(im, 'norm'):
        ticks = np.linspace(im.norm.vmin, 10, 10)
    else:
        ticks = np.linspace(np.nanmin(all_c), 0, 10 )

    upper_ext = 10.0          # how far above 0 you want to extend
    n_neg_ticks = 8           # how many steps between vmin and 0
    n_pos_extra = 3           # how many manual positives (e.g., 2–3 points)

    if hasattr(im, "norm"):
        vmin = im.norm.vmin
    else:
        vmin = np.nanmin(all_c)

    # equally spaced negatives up to 0
    ticks_neg = np.linspace(vmin, 0, n_neg_ticks)

    # manually append a few positive ones
    ticks_pos = np.linspace(0, upper_ext, n_pos_extra + 1)[1:]  # omit the extra 0 duplicate

    # combine both parts
    ticks = np.concatenate((ticks_neg, ticks_pos))


    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=8, colors='black', width=0.6)
    cbar.outline.set_edgecolor('black')
    for t in cbar.ax.get_yticklabels():
        t.set_color('black')



    plt.tight_layout()

    out_file = f"grid_slice_{plane_mode or '_'.join(map(str, atom_indices))}_bubbles.png"
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"Saved: {out_file}")


# ============================================================
#                 CLI ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Visualize 2D energy slices from grid JSON files.")
    parser.add_argument("args", nargs="+", help="Plane type (XY/XZ/YZ) or three atom indices (e.g. 0 1 2)")
    parser.add_argument("--tol", type=float, default=0.1, help="Plane thickness tolerance in Å (default: 0.1)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma (default: 1.0)")
    parsed = parser.parse_args()
    run_slice(parsed.args, tol=parsed.tol, sigma=parsed.sigma)


if __name__ == "__main__":
    main()
