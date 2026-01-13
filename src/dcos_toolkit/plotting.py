import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .models import SessionState
from .utils import ensure_dir
from .peaks import (
    find_cross_peaks_max,
    find_cross_peaks_min,
    find_diagonal_peaks,
    add_mirror_peaks,
    CROSS_THRESHOLD_REL
)

import logging
logger = logging.getLogger(__name__)

#  HELPER FUNCTIONS FOR PEAK ANNOTATION
# ------------------------------------------------------

def _sign_str(value: float) -> str:
    # Label sign
    if value > 0:
        return "+"
    elif value < 0:
        return "−"
    return ""

def _format_coord(value: float) -> str:
    # Round to nearest integer for display
    return str(int(round(value)))

def _get_text_box_size(label: str, fontsize: int, ax_dims: tuple[float, float], px_per_pt: float) -> tuple[float, float]:
    # Estimate label size in axis units
    ax_w_px, ax_h_px = ax_dims

    char_width_factor = 0.62
    char_height_factor = 1.25

    width_pt = char_width_factor * fontsize * len(label) + 10
    height_pt = char_height_factor * fontsize + 8

    width_ax = width_pt * px_per_pt / ax_w_px
    height_ax = height_pt * px_per_pt / ax_h_px

    return width_ax, height_ax

def _solve_label_position(x_axis, y_axis, dx_pt, dy_pt, ha, va, w_ax, h_ax, px_per_pt, ax_dims):
    # Compute bounding box coordinates in axis units
    ax_w_px, ax_h_px = ax_dims

    offset_x = dx_pt * px_per_pt / ax_w_px
    offset_y = dy_pt * px_per_pt / ax_h_px

    x_pos = x_axis + offset_x
    y_pos = y_axis + offset_y

    if ha == "left":
        x0, x1 = x_pos, x_pos + w_ax
    else:
        x0, x1 = x_pos - w_ax, x_pos

    if va == "bottom":
        y0, y1 = y_pos, y_pos + h_ax
    else:
        y0, y1 = y_pos - h_ax, y_pos

    return x0, y0, x1, y1

def _check_overlap(rect1, rect2):
    x0, y0, x1, y1 = rect1
    a0, b0, a1, b1 = rect2
    # Return True if rectangles overlap
    return not (x1 <= a0 or a1 <= x0 or y1 <= b0 or b1 <= y0)

def _annotate_peaks(ax, peaks, x_vals, y_vals, marker_size=10, fontsize=7, offset_points=12, edge_frac=0.12):
    # Draw markers and labels, avoiding overlaps
    if not peaks:
        return

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    pad = offset_points
    edge_threshold = 1.0 - edge_frac

    fig = ax.figure
    dpi = fig.dpi
    pos = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_w_px = max(1, pos.width * fig_w * dpi)
    ax_h_px = max(1, pos.height * fig_h * dpi)
    px_per_pt = dpi / 72.0
    ax_dims = (ax_w_px, ax_h_px)

    placed_rects = []

    # Sort peaks by absolute value 
    peak_abs_list = [(abs(p[2]), p) for p in peaks]
    peak_abs_list.sort(reverse=True)
    sorted_peaks = [p for _, p in peak_abs_list]

    for xp, yp, pv in sorted_peaks:
        if not (x_min <= xp <= x_max and y_min <= yp <= y_max):
            continue

        data_coords = (xp, yp)  
        pixel_coords = ax.transData.transform(data_coords)  
        axis_coords = ax.transAxes.inverted().transform(pixel_coords)  
        xa, ya = axis_coords  

        # Determine base offsets depending on distance to axis edges
        base_dx = -pad if xa > edge_threshold else pad
        base_dy = -pad if ya > edge_threshold else pad
        base_ha = "right" if base_dx < 0 else "left"
        base_va = "top" if base_dy < 0 else "bottom"

        label = f"{_sign_str(pv)}({_format_coord(xp)},{_format_coord(yp)})"
        w_ax, h_ax = _get_text_box_size(label, fontsize, ax_dims, px_per_pt)

        # Candidate positions for label
        candidate_offsets = [
            (base_dx, base_dy, base_ha, base_va),
            (base_dx, -base_dy, base_ha, "top" if base_va == "bottom" else "bottom"),
            (-base_dx, base_dy, "right" if base_ha == "left" else "left", base_va),
            (int(base_dx*1.4), base_dy, base_ha, base_va)
        ]

        chosen = None
        for dx, dy, ha, va in candidate_offsets:
            rect = _solve_label_position(xa, ya, dx, dy, ha, va, w_ax, h_ax, px_per_pt, ax_dims)
            if rect[0] < 0 or rect[1] < 0 or rect[2] > 1 or rect[3] > 1:
                continue
            if any(_check_overlap(rect, r) for r in placed_rects):
                continue
            chosen = (dx, dy, ha, va)
            placed_rects.append(rect)
            break

        if not chosen:
            chosen = (base_dx, base_dy, base_ha, base_va)

        dx, dy, ha, va = chosen
        ax.scatter(xp, yp, s=marker_size, edgecolors="k", facecolors="none", zorder=20)
        ax.annotate(
            label,
            xy=(xp, yp),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.5, fc="white", ec="none"),
            arrowprops=dict(arrowstyle="->", lw=0.6, color="0.2", shrinkA=2, shrinkB=3),
            zorder=30,
            clip_on=True
        )


# ------------------------------------------------------
# HELPER FUNCTIONS FOR PLOTTING

def _padded_limits(min_val, max_val, pad_frac=0.05):
    span = max_val - min_val
    pad = pad_frac * span if span > 0 else 1.0
    return min_val - pad, max_val + pad

def _reversed_limits(min_val, max_val, pad_frac=0.05):
    lo, hi = _padded_limits(min_val, max_val, pad_frac)
    return hi, lo

def _calculate_levels_and_threshold(z: np.ndarray):
    # Compute contour levels using global threshold
    if z.size == 0:
        return None, 0.0

    zmax = abs(z).max()
    cutoff = CROSS_THRESHOLD_REL * zmax

    pos_levels = np.linspace(cutoff, zmax, 4)
    neg_levels = -pos_levels[::-1]
    levels = np.concatenate([neg_levels, pos_levels])

    visible_levels = np.sort(np.unique(np.abs(levels)))
    thresh_cross = visible_levels[0] if len(visible_levels) > 0 else 0.0

    return levels, thresh_cross

def _plot_map_panel(ax_map, ax_cbar, x, y, z, cmap, peaks_config, is_sync, mark_mirror=False):
    # Draw a single map panel with peaks and contours
    zmax = abs(z).max() if z.size else 0.0
    pcm = ax_map.pcolormesh(x, y, z, cmap=cmap, vmin=-zmax, vmax=zmax, shading="gouraud")
    ax_map.plot(x, y, lw=0.5, c="0.3", zorder=10)  # diagonal line

    cb = ax_map.figure.colorbar(pcm, cax=ax_cbar, pad=0.02)
    pos = ax_cbar.get_position()
    ax_cbar.set_position([pos.x0 + 0.02, pos.y0, pos.width, pos.height])
    cb.set_label("Correlation intensity", labelpad=15)
    cb.set_ticks([])
    cb.ax.yaxis.get_offset_text().set_visible(False)

    levels, thresh_cross = _calculate_levels_and_threshold(z)
    if levels is not None:
        ax_map.contour(x, y, z, levels=levels, colors="0", linewidths=0.5)

    # Detect peaks
    peaks_found = []

    if is_sync and (n := peaks_config.get('n_diag', 0)) > 0:
        peaks_found.extend(find_diagonal_peaks(z, x, n))

    if (n := peaks_config.get('n_cross_max', 0)) > 0:
        peaks_found.extend(find_cross_peaks_max(z, x, y, n))

    if (n := peaks_config.get('n_cross_min', 0)) > 0:
        peaks_found.extend(find_cross_peaks_min(z, x, y, n))

    if mark_mirror:
        peaks_found = add_mirror_peaks(peaks_found, z, x)

    _annotate_peaks(ax_map, peaks_found, x, y)

    ax_map.set_xlim(x.min(), x.max())
    ax_map.set_ylim(y.min(), y.max())
    ax_map.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

# ------------------------------------------------------
# MAIN FUNCTIONS FOR PLOTTING

def combine_plots(
    spec_syn: pd.DataFrame,
    spec_asyn: Optional[pd.DataFrame],
    base_series: pd.Series,
    dataset_label: str,
    out_path: str,
    *,
    cmap: str = "jet",
    mark_mirror_peaks: bool = False,
    peaks_config: dict
) -> None:

    x_base = np.array(base_series.index)
    base = base_series.values
    x = spec_syn.columns.values
    y = spec_syn.index.values
    z_syn = spec_syn.values.T

    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(
        2, 7,
        width_ratios=[0.7, 5, 0.6, 2.5, 0.7, 5, 0.6],
        height_ratios=[5, 0.7],
        wspace=0, hspace=0
    )

    # Define axes
    ax_s_y = fig.add_subplot(gs[0, 0])
    ax_s_map = fig.add_subplot(gs[0, 1])
    ax_s_cb = fig.add_subplot(gs[0, 2])
    ax_s_x = fig.add_subplot(gs[1, 1], sharex=ax_s_map)
    fig.add_subplot(gs[:, 3]).axis("off")  
    ax_a_y = fig.add_subplot(gs[0, 4])
    ax_a_map = fig.add_subplot(gs[0, 5])
    ax_a_cb = fig.add_subplot(gs[0, 6])
    ax_a_x = fig.add_subplot(gs[1, 5], sharex=ax_a_map)

    # Sync panel
    conf_s = {
        'n_diag': peaks_config.get('n_sync_diag', 0),
        'n_cross_max': peaks_config.get('n_sync_cmax', 0),
        'n_cross_min': peaks_config.get('n_sync_cmin', 0)
    }
    _plot_map_panel(ax_s_map, ax_s_cb, x, y, z_syn, cmap, conf_s, is_sync=True, mark_mirror=mark_mirror_peaks)

    # Async panel
    if spec_asyn is not None:
        z_asyn = spec_asyn.values.T
        conf_a = {
            'n_cross_max': peaks_config.get('n_async_cmax', 0),
            'n_cross_min': peaks_config.get('n_async_cmin', 0)
        }
        _plot_map_panel(ax_a_map, ax_a_cb, x, y, z_asyn, cmap, conf_a, is_sync=False, mark_mirror=mark_mirror_peaks)
    else:
        for ax in [ax_a_y, ax_a_map, ax_a_cb, ax_a_x]:
            ax.axis("off")

    # Side plots
    y0_1d, y1_1d = _padded_limits(base.min(), base.max())
    x0_1d, x1_1d = _reversed_limits(base.min(), base.max())

    def _setup_side_plots(ax_x, ax_y, title):
        ax_x.plot(x_base, base, c="0")
        ax_x.set_ylim(y0_1d, y1_1d)
        ax_x.set_xlabel(r"$\lambda$ (nm)", labelpad=2)
        ax_x.tick_params(left=False, labelleft=False, labelsize=9)

        ax_y.plot(base, x_base, c="0")
        ax_y.set_xlim(x0_1d, x1_1d)
        ax_y.set_ylim(y.min(), y.max())
        ax_y.set_ylabel(r"$\lambda$ (nm)", labelpad=10)
        ax_y.tick_params(bottom=False, labelbottom=False, labelsize=9)

        pos = ax_y.get_position()
        fig.text(pos.x0, pos.y1 + 0.03, title, ha="left", va="bottom", fontsize=16, c="0")

    _setup_side_plots(ax_s_x, ax_s_y, "Synchronous spectrum")
    if spec_asyn is not None:
        _setup_side_plots(ax_a_x, ax_a_y, "Asynchronous spectrum")

    # Plot label
    fig.text(0.5, 0.02, f"Sample: {dataset_label}", ha='center', va='bottom', fontsize=8, color='grey')

    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def visualize_session(
    session: SessionState,
    *,
    colormap: str = "jet",
    mark_mirror_peaks: bool = False,
    n_sync_diag_peaks: int = 1,
    n_sync_cross_max_peaks: int = 1,
    n_sync_cross_min_peaks: int = 1,
    n_async_cross_max_peaks: int = 1,
    n_async_cross_min_peaks: int = 1,
) -> None:

    if not session.datasets:
        raise RuntimeError("No parsed CD datasets found. Please load data first.")

    MAX_PEAKS = 5
    peaks_conf = {
        'n_sync_diag': max(0, min(n_sync_diag_peaks, MAX_PEAKS)),
        'n_sync_cmax': max(0, min(n_sync_cross_max_peaks, MAX_PEAKS)),
        'n_sync_cmin': max(0, min(n_sync_cross_min_peaks, MAX_PEAKS)),
        'n_async_cmax': max(0, min(n_async_cross_max_peaks, MAX_PEAKS)),
        'n_async_cmin': max(0, min(n_async_cross_min_peaks, MAX_PEAKS)),
    }

    output_dir = Path(session.output_dir)
    errors = []

    for idx, ds in enumerate(session.datasets):
        name = ds.name or f"dataset_{idx}"
        if ds.sync is None:
            continue

        logger.info(f"Visualizing dataset: {name}")
        try:
            lam = ds.lambda_axis
            sync_df = pd.DataFrame(ds.sync, index=lam, columns=lam)
            async_df = pd.DataFrame(ds.async_, index=lam, columns=lam) if ds.async_ is not None else None

            base_vec = ds.mre[0, :] if ds.mre is not None else ds.cd_mdeg[0, :]
            base_series = pd.Series(base_vec, index=lam).loc[sync_df.columns]

            combine_plots(
                spec_syn=sync_df,
                spec_asyn=async_df,
                base_series=base_series,
                dataset_label=name,
                out_path=str(output_dir / f"{name}_2DCOS_combined.png"),
                cmap=colormap,
                mark_mirror_peaks=mark_mirror_peaks,
                peaks_config=peaks_conf
            )
        except Exception as exc:
            msg = str(exc)
            logger.error(f"Visualization failed for '{name}': {msg}")
            errors.append((name, msg))

    if errors:
        failed = ", ".join(n for n, _ in errors)
        raise RuntimeError(f"Visualization finished with errors for: {failed}")
