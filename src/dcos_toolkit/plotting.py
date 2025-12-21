
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .models import SessionState
from .utils import ensure_dir
from .peaks import find_cross_peaks_max, find_cross_peaks_min, find_diagonal_peaks


def lambda_cut(
    axis_vals: np.ndarray,
    matrix: np.ndarray,
    lam_min: float,
    lam_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a square matrix to a given wavelength range.

    If lam_min >= lam_max, returns the full range (original behaviour).
    """
    lam = np.asarray(axis_vals, dtype=float)
    if lam_min < lam_max:
        mask = (lam >= lam_min) & (lam <= lam_max)
        if mask.sum() >= 2:
            lam_sel = lam[mask]
            mat_sel = matrix[np.ix_(mask, mask)]
            return lam_sel, mat_sel
    return lam, matrix


def _coords(spec: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return x, y axes and z matrix compatible with pcolormesh."""
    x_raw = spec.columns.to_numpy(dtype=float)
    y_raw = spec.index.to_numpy(dtype=float)
    z = spec.values

    m, n = z.shape
    nx = len(x_raw)
    ny = len(y_raw)

    if nx > n:
        x = x_raw[:n]
    else:
        x = x_raw
        z = z[:, :nx]

    if ny > m:
        y = y_raw[:m]
    else:
        y = y_raw
        z = z[:ny, :]

    zmax = np.abs(z).max() if z.size else 0.0
    return x, y, z, zmax


def _annotate_peaks(
    ax: plt.Axes,
    peaks: list[tuple[float, float, float]],
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    marker_size: int = 28,
    fontsize: int = 8,
    offset_points: int = 12,
    edge_frac: float = 0.12,
) -> None:
    """Draw peak markers and annotate them with boundary-aware labels.

    This implementation keeps label placement simple and deterministic:
    - labels are offset in screen units (points) to be scale-independent,
    - direction (left/right/top/bottom) is chosen to avoid leaving the axes,
    - a lightweight overlap reduction is applied by trying a small set of
      candidate offsets and rejecting placements that overlap previously placed labels.

    Notes
    -----
    - This does not guarantee a collision-free layout for dense peak clusters,
      but it significantly reduces overlap without complex optimization.
    - Labels are formatted as integers (rounded), matching the previous behavior.
    """
    if not peaks:
        return

    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
    y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))

    edge = float(np.clip(edge_frac, 0.0, 0.49))
    pad = int(offset_points)

    # Precompute conversion between points and axes-fraction units
    fig = ax.figure
    dpi = float(fig.dpi)
    fig_w_in, fig_h_in = fig.get_size_inches()
    pos = ax.get_position()  # in figure fraction

    ax_w_px = max(1.0, float(pos.width * fig_w_in * dpi))
    ax_h_px = max(1.0, float(pos.height * fig_h_in * dpi))
    px_per_pt = dpi / 72.0

    def axes_fraction(xp: float, yp: float) -> tuple[float, float]:
        return tuple(ax.transAxes.inverted().transform(ax.transData.transform((xp, yp))))

    def _format_int(v: float) -> str:
        return str(int(round(float(v))))

    def _estimate_text_box_axes(label: str) -> tuple[float, float]:
        # Very simple approximation: average glyph width ~0.62*fontsize
        # + a bit of padding for bbox ("round").
        w_pt = (0.62 * fontsize * len(label)) + 10.0
        h_pt = (1.25 * fontsize) + 8.0
        w_ax = (w_pt * px_per_pt) / ax_w_px
        h_ax = (h_pt * px_per_pt) / ax_h_px
        return w_ax, h_ax

    def _rect_for_label(xa: float, ya: float, dx_pt: int, dy_pt: int, ha: str, va: str, w_ax: float, h_ax: float):
        # Convert offset (points) to axes fraction
        xa2 = xa + (dx_pt * px_per_pt) / ax_w_px
        ya2 = ya + (dy_pt * px_per_pt) / ax_h_px

        if ha == "left":
            x0, x1 = xa2, xa2 + w_ax
        elif ha == "right":
            x0, x1 = xa2 - w_ax, xa2
        else:  # center
            x0, x1 = xa2 - w_ax / 2.0, xa2 + w_ax / 2.0

        if va == "bottom":
            y0, y1 = ya2, ya2 + h_ax
        elif va == "top":
            y0, y1 = ya2 - h_ax, ya2
        else:  # center
            y0, y1 = ya2 - h_ax / 2.0, ya2 + h_ax / 2.0

        return (x0, y0, x1, y1), (xa2, ya2)

    def _overlaps(r1, r2) -> bool:
        x0, y0, x1, y1 = r1
        a0, b0, a1, b1 = r2
        return not (x1 <= a0 or a1 <= x0 or y1 <= b0 or b1 <= y0)

    placed_rects: list[tuple[float, float, float, float]] = []

    for xp, yp, _ in peaks:
        xp_f = float(xp)
        yp_f = float(yp)

        # Skip outside
        if not (x_min <= xp_f <= x_max and y_min <= yp_f <= y_max):
            continue

        xa, ya = axes_fraction(xp_f, yp_f)

        # Base direction away from nearest edge
        if xa >= 1.0 - edge:
            base_dx, base_ha = -pad, "right"
        elif xa <= edge:
            base_dx, base_ha = pad, "left"
        else:
            base_dx, base_ha = (pad, "left") if xa < 0.5 else (-pad, "right")

        if ya >= 1.0 - edge:
            base_dy, base_va = -pad, "top"
        elif ya <= edge:
            base_dy, base_va = pad, "bottom"
        else:
            base_dy, base_va = (pad, "bottom") if ya < 0.5 else (-pad, "top")

        label = f"({_format_int(xp_f)}, {_format_int(yp_f)})"
        w_ax, h_ax = _estimate_text_box_axes(label)

        # Candidate offsets: try a few simple alternatives around the base direction
        candidates = [
            (base_dx, base_dy, base_ha, base_va),
            (base_dx, -base_dy, base_ha, "top" if base_va == "bottom" else "bottom"),
            (-base_dx, base_dy, "right" if base_ha == "left" else "left", base_va),
            (-base_dx, -base_dy, "right" if base_ha == "left" else "left", "top" if base_va == "bottom" else "bottom"),
            (int(base_dx * 1.4), base_dy, base_ha, base_va),
            (base_dx, int(base_dy * 1.4), base_ha, base_va),
        ]

        chosen = None
        chosen_anchor = None

        for dx_pt, dy_pt, ha, va in candidates:
            rect, anchor = _rect_for_label(xa, ya, dx_pt, dy_pt, ha, va, w_ax, h_ax)

            # Keep inside axes (with a tiny margin)
            x0, y0, x1, y1 = rect
            if x0 < 0.0 or y0 < 0.0 or x1 > 1.0 or y1 > 1.0:
                continue

            if any(_overlaps(rect, prev) for prev in placed_rects):
                continue

            chosen = (dx_pt, dy_pt, ha, va)
            chosen_anchor = anchor
            placed_rects.append(rect)
            break

        # If nothing fits (dense cluster), fall back to base placement (still edge-aware)
        if chosen is None:
            chosen = (base_dx, base_dy, base_ha, base_va)

        dx_pt, dy_pt, ha, va = chosen

        ax.scatter(xp_f, yp_f, s=marker_size, edgecolors="k", facecolors="none")
        ax.annotate(
            label,
            xy=(xp_f, yp_f),
            xycoords="data",
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            bbox=dict(boxstyle="round", alpha=0.5, facecolor="white", ec="none"),
            arrowprops=dict(arrowstyle="->", lw=0.7, shrinkA=0, shrinkB=0),
            annotation_clip=True,
            clip_on=True,
        )



def _padded_limits(min_val: float, max_val: float, *, pad_frac: float = 0.05) -> tuple[float, float]:
    if max_val > min_val:
        span = max_val - min_val
        pad = pad_frac * span
        return min_val - pad, max_val + pad
    return min_val - 1.0, max_val + 1.0


def _reversed_limits(min_val: float, max_val: float, *, pad_frac: float = 0.05) -> tuple[float, float]:
    """Return reversed-axis limits with padding (max -> min)."""
    lo, hi = _padded_limits(min_val, max_val, pad_frac=pad_frac)
    return hi, lo


def _shift_colorbar_axis(ax: plt.Axes, *, dx: float = 0.02) -> None:
    pos = ax.get_position()
    ax.set_position([pos.x0 + dx, pos.y0, pos.width, pos.height])


def _sync_peaks(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_diag: int,
    n_cross_max: int,
    n_cross_min: int,
) -> list[tuple[float, float, float]]:
    peaks: list[tuple[float, float, float]] = []
    if n_diag > 0:
        peaks.extend(find_diagonal_peaks(z, x, int(n_diag)))
    if n_cross_max > 0:
        peaks.extend(find_cross_peaks_max(z, x, y, int(n_cross_max)))
    if n_cross_min > 0:
        peaks.extend(find_cross_peaks_min(z, x, y, int(n_cross_min)))
    return peaks


def _async_peaks(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_cross_max: int,
    n_cross_min: int,
) -> list[tuple[float, float, float]]:
    peaks: list[tuple[float, float, float]] = []
    if n_cross_max > 0:
        peaks.extend(find_cross_peaks_max(z, x, y, int(n_cross_max)))
    if n_cross_min > 0:
        peaks.extend(find_cross_peaks_min(z, x, y, int(n_cross_min)))
    return peaks


def _mirror_peaks(
    peaks: list[tuple[float, float, float]],
    z: np.ndarray,
    axis: np.ndarray,
) -> list[tuple[float, float, float]]:
    """Return a list of peaks augmented with their mirrored counterparts.

    For each peak (x, y, v) where x != y, this function adds (y, x, v_mirror),
    where v_mirror is taken from the provided matrix ``z`` at the swapped indices.

    Notes
    -----
    This assumes that ``axis`` represents both x and y coordinates of ``z`` (typical λ–λ maps).
    The mirrored value is sampled by nearest-axis indexing.
    """
    if not peaks:
        return []

    axis = np.asarray(axis, dtype=float)
    z = np.asarray(z)

    def idx(val: float) -> int:
        return int(np.argmin(np.abs(axis - float(val))))

    out: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float]] = set()

    for px, py, pv in peaks:
        px = float(px)
        py = float(py)
        pv = float(pv)

        if (px, py) not in seen:
            out.append((px, py, pv))
            seen.add((px, py))

        if px == py:
            continue

        mx, my = py, px
        if (mx, my) in seen:
            continue

        i = idx(px)
        j = idx(py)
        mv = float(z[j, i])
        out.append((mx, my, mv))
        seen.add((mx, my))

    return out


def combine_plots(
    spec_syn: pd.DataFrame,
    spec_asyn: Optional[pd.DataFrame],
    base_series: pd.Series,
    dataset_label: str,
    out_path: str,
    *,
    cmap: str = "jet",
    mark_mirror_peaks: bool = False,
    mark_peaks_sync: bool = True,
    n_sync_diag_peaks: int = 0,
    n_sync_cross_max_peaks: int = 0,
    n_sync_cross_min_peaks: int = 0,
    mark_peaks_async: bool = True,
    n_async_cross_max_peaks: int = 0,
    n_async_cross_min_peaks: int = 0,
) -> None:
    """Create a side-by-side synchronous/asynchronous 2D-COS figure and save it.

    Parameters
    ----------
    spec_syn:
        Synchronous spectrum as a square DataFrame (index and columns are axis values).
    spec_asyn:
        Asynchronous spectrum as a square DataFrame. If None, only the synchronous
        panel is rendered.
    base_series:
        1D reference spectrum (e.g., mean dynamic spectrum) used for marginal plots.
        Its index should match the wavelength axis.
    dataset_label:
        Title shown above the figure (typically dataset name).
    out_path:
        Output PNG path.
    cmap:
        Matplotlib colormap name.
    mark_mirror_peaks:
        If True, peaks are mirrored (x, y) -> (y, x) before plotting.
    mark_peaks_sync, mark_peaks_async:
        Enable peak annotations for the synchronous/asynchronous maps.
    n_sync_diag_peaks, n_sync_cross_max_peaks, n_sync_cross_min_peaks:
        Number of diagonal / positive / negative peaks to annotate on the synchronous map.
    n_async_cross_max_peaks, n_async_cross_min_peaks:
        Number of positive / negative peaks to annotate on the asynchronous map.
    """
    x_base = base_series.index.values.astype(float)
    base = base_series.values

    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(
        nrows=2,
        ncols=7,
        width_ratios=[0.7, 5.0, 0.6, 2.5, 0.7, 5.0, 0.6],
        height_ratios=[5.0, 0.7],
    )

    # synchronous
    ax_s_y = fig.add_subplot(gs[0, 0], sharey=None)
    ax_s_map = fig.add_subplot(gs[0, 1])
    s_colorbar = fig.add_subplot(gs[0, 2])
    ax_s_x = fig.add_subplot(gs[1, 1], sharex=ax_s_map)

    # gap
    gap_ax = fig.add_subplot(gs[:, 3])
    gap_ax.axis("off")

    # asynchronous
    ax_a_y = fig.add_subplot(gs[0, 4], sharey=None)
    ax_a_map = fig.add_subplot(gs[0, 5])
    a_colorbar = fig.add_subplot(gs[0, 6])
    ax_a_x = fig.add_subplot(gs[1, 5], sharex=ax_a_map)

    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    # ===== SYNCHRONOUS =====
    x, y, z, zmax = _coords(spec_syn)
    y_min, y_max = y.min(), y.max()

    pcm_s = ax_s_map.pcolormesh(
        x, y, z,
        cmap=cmap,
        vmin=-zmax,
        vmax=zmax,
        shading="gouraud",
    )
    ax_s_map.add_artist(mlines.Line2D(x, y, linewidth=0.5, c="0"))

    cb = fig.colorbar(pcm_s, cax=s_colorbar, label="Energy values", pad=0.02)
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.yaxis.set_label_position("right")
    cb.ax.tick_params(labelsize=9)
    cb.ax.yaxis.get_offset_text().set_visible(False)
    _shift_colorbar_axis(s_colorbar, dx=0.02)

    ax_s_map.xaxis.get_offset_text().set_visible(False)
    ax_s_map.yaxis.get_offset_text().set_visible(False)
    ax_s_map.contour(x, y, z, 5, colors="0", linewidths=0.5)

    if mark_peaks_sync:
        peaks = _sync_peaks(
            z, x, y,
            n_diag=n_sync_diag_peaks,
            n_cross_max=n_sync_cross_max_peaks,
            n_cross_min=n_sync_cross_min_peaks,
        )
        if mark_mirror_peaks:
            peaks = _mirror_peaks(peaks, z, x)
        _annotate_peaks(ax_s_map, peaks, x, y)

    ax_s_map.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_s_map.spines["bottom"].set_visible(False)
    ax_s_map.set_xlabel("")
    ax_s_map.set_ylabel("")
    ax_s_map.set_xlim(x.min(), x.max())
    ax_s_map.set_ylim(y_min, y_max)

    # bottom 1D (sync)
    y0_s, y1_s = _padded_limits(float(np.min(base)), float(np.max(base)))
    ax_s_x.plot(x_base, base, c="0")
    ax_s_x.set_xlim(x.min(), x.max())
    ax_s_x.set_ylim(y0_s, y1_s)
    ax_s_x.set_ylabel("", labelpad=-4)
    ax_s_x.set_xlabel(r"$\lambda$ (nm)", labelpad=2)
    ax_s_x.tick_params(labelsize=9)
    ax_s_x.xaxis.get_offset_text().set_visible(False)
    ax_s_x.yaxis.get_offset_text().set_visible(False)
    ax_s_x.tick_params(axis="y", left=False, labelleft=False)

    # vertical 1D (sync)
    x0_s, x1_s = _reversed_limits(float(np.min(base)), float(np.max(base)))
    ax_s_y.plot(base, x_base, c="0")
    ax_s_y.set_xlim(x0_s, x1_s)
    ax_s_y.set_ylim(y_min, y_max)
    ax_s_y.set_xlabel("")
    ax_s_y.set_ylabel(r"$\lambda$ (nm)", labelpad=10)

    ax_s_y.yaxis.tick_left()
    ax_s_y.yaxis.set_label_position("left")
    ax_s_y.tick_params(axis="y", which="both", left=True, right=False, labelleft=True, labelright=False, labelsize=9)
    ax_s_y.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_s_y.xaxis.get_offset_text().set_visible(False)
    ax_s_y.yaxis.get_offset_text().set_visible(False)

    # ===== ASYNCHRONOUS =====
    if spec_asyn is not None:
        x_a, y_a, z_a, zmax_a = _coords(spec_asyn)
        y_a_min, y_a_max = y_a.min(), y_a.max()

        y0_a, y1_a = _padded_limits(float(np.min(base)), float(np.max(base)))
        ax_a_x.plot(x_base, base, c="0")
        ax_a_x.set_xlim(x_a.min(), x_a.max())
        ax_a_x.set_ylim(y0_a, y1_a)
        ax_a_x.set_ylabel("", labelpad=-4)
        ax_a_x.set_xlabel(r"$\lambda$ (nm)", labelpad=2)
        ax_a_x.tick_params(labelsize=9)
        ax_a_x.xaxis.get_offset_text().set_visible(False)
        ax_a_x.yaxis.get_offset_text().set_visible(False)
        ax_a_x.tick_params(axis="y", left=False, labelleft=False)

        x0_a, x1_a = _reversed_limits(float(np.min(base)), float(np.max(base)))
        ax_a_y.plot(base, x_a, c="0")
        ax_a_y.set_xlim(x0_a, x1_a)
        ax_a_y.set_ylim(y_a_min, y_a_max)
        ax_a_y.set_xlabel("")
        ax_a_y.set_ylabel(r"$\lambda$ (nm)", labelpad=10)

        ax_a_y.yaxis.tick_left()
        ax_a_y.yaxis.set_label_position("left")
        ax_a_y.tick_params(axis="y", which="both", left=True, right=False, labelleft=True, labelright=False, labelsize=9)
        ax_a_y.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax_a_y.xaxis.get_offset_text().set_visible(False)
        ax_a_y.yaxis.get_offset_text().set_visible(False)

        pcm_a = ax_a_map.pcolormesh(
            x_a, y_a, z_a,
            cmap=cmap,
            vmin=-zmax_a,
            vmax=zmax_a,
            shading="gouraud",
        )
        ax_a_map.add_artist(mlines.Line2D(x_a, y_a, linewidth=0.5, c="0"))

        cb_a = fig.colorbar(pcm_a, cax=a_colorbar, label="Energy values", pad=0.02)
        cb_a.ax.yaxis.set_ticks_position("right")
        cb_a.ax.yaxis.set_label_position("right")
        cb_a.ax.tick_params(labelsize=9)
        cb_a.ax.yaxis.get_offset_text().set_visible(False)
        _shift_colorbar_axis(a_colorbar, dx=0.02)

        ax_a_map.xaxis.get_offset_text().set_visible(False)
        ax_a_map.yaxis.get_offset_text().set_visible(False)

        ax_a_map.contour(x_a, y_a, z_a, 5, colors="0", linewidths=0.5)
        ax_a_map.set_xlim(x_a.min(), x_a.max())
        ax_a_map.set_ylim(y_a_min, y_a_max)
        ax_a_map.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        if mark_peaks_async:
            peaks = _async_peaks(
                z_a, x_a, y_a,
                n_cross_max=n_async_cross_max_peaks,
                n_cross_min=n_async_cross_min_peaks,
            )
            if mark_mirror_peaks:
                peaks = _mirror_peaks(peaks, z_a, x_a)
            _annotate_peaks(ax_a_map, peaks, x_a, y_a)

    else:
        ax_a_map.axis("off")
        ax_a_x.axis("off")
        ax_a_y.axis("off")
        a_colorbar.axis("off")

    sync_pos = ax_s_y.get_position()
    async_pos = ax_a_y.get_position()
    fig.text(sync_pos.x0, 0.97, "Synchronous spectrum", ha="left", va="center", fontsize=16, c="0")
    fig.text(async_pos.x0, 0.97, "Asynchronous spectrum", ha="left", va="center", fontsize=16, c="0")
    fig.text(0.5, 0.93, dataset_label, ha="center", va="center", fontsize=14, c="0")

    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def visualize_session(
    session: SessionState,
    *,
    colormap: str = "jet",
    lambda_min: float = 0.0,
    lambda_max: float = 0.0,
    mark_mirror_peaks: bool = False,
    mark_peaks_sync: bool = True,
    n_sync_diag_peaks: int = 2,
    n_sync_cross_max_peaks: int = 3,
    n_sync_cross_min_peaks: int = 3,
    mark_peaks_async: bool = True,
    n_async_cross_max_peaks: int = 3,
    n_async_cross_min_peaks: int = 3,
    dataset_indices: Optional[Iterable[int]] = None,
) -> None:
    """Render and save 2D-COS figures for datasets stored in a session.

    For each selected dataset, this function:
    - applies wavelength cropping (lambda_min/lambda_max) to the 2D maps,
    - renders synchronous and (if available) asynchronous maps,
    - optionally annotates peaks according to the provided counts,
    - saves the resulting PNG into ``session.output_dir``.

    Parameters
    ----------
    session:
        Analysis session containing computed spectra (and datasets list).
    colormap:
        Matplotlib colormap name used for both maps.
    lambda_min, lambda_max:
        If lambda_min < lambda_max, crop maps to this wavelength interval.
    mark_mirror_peaks:
        If True, peaks are mirrored (x, y) -> (y, x) before plotting.
    mark_peaks_sync, mark_peaks_async:
        Enable peak annotations for synchronous/asynchronous maps.
    n_sync_diag_peaks, n_sync_cross_max_peaks, n_sync_cross_min_peaks:
        Peak counts for synchronous map annotations.
    n_async_cross_max_peaks, n_async_cross_min_peaks:
        Peak counts for asynchronous map annotations.
    dataset_indices:
        Optional subset of dataset indices to render. If None, renders all datasets.

    Raises
    ------
    RuntimeError
        If no datasets are present in the session.
    """
    if not session.datasets:
        raise RuntimeError(" Visualization: No parsed CD datasets available. Run input/parsing first.")

    MAX_PEAKS = 5

    def _coerce_int(val, name: str) -> int:
        """
        Convert to int in a user-friendly way.
        Accepts ints and numeric strings (e.g. "3").
        """
        try:
            return int(val)
        except (TypeError, ValueError):
            raise ValueError(f" Visualization: '{name}' must be an integer (0–{MAX_PEAKS}). Got: {val!r}")

    def _clamp_peaks(n: int) -> int:
        if n < 0:
            return 0
        if n > MAX_PEAKS:
            return MAX_PEAKS
        return n

    output_dir = Path(session.output_dir)

    # dataset indices sanity
    if dataset_indices is None:
        indices = list(range(len(session.datasets)))
    else:
        indices = []
        for raw in dataset_indices:
            try:
                i = int(raw)
            except Exception:
                print(f" Visualization: ignored invalid dataset index: {raw!r}")
                continue
            if 0 <= i < len(session.datasets):
                indices.append(i)
            else:
                print(f" Visualization: ignored out-of-range dataset index: {i} (valid: 0..{len(session.datasets)-1})")

        if not indices:
            print(" Visualization: no valid dataset indices to visualize.")
            return

    errors: list[tuple[str, str]] = []

    for idx in indices:
        ds = session.datasets[idx]
        name = ds.name

        print(f"\n Visualising dataset [{idx}]: {name}")

        if ds.sync is None:
            print(f" Visualization: missing synchronous matrix for dataset: {name}. Run 2D-COS cell first.")
            continue

        if ds.async_ is None:
            print(f" Visualization: asynchronous matrix is missing for dataset: {name} (plot will omit async panel).")

        try:
            lam = ds.lambda_axis
            sync_df = pd.DataFrame(ds.sync, index=lam, columns=lam)
            async_df = pd.DataFrame(ds.async_, index=lam, columns=lam) if ds.async_ is not None else None

            lam_sync, sync_cut = lambda_cut(lam, sync_df.values, lambda_min, lambda_max)
            sync_df = pd.DataFrame(sync_cut, index=lam_sync, columns=lam_sync)

            if async_df is not None:
                lam_async, async_cut = lambda_cut(lam, async_df.values, lambda_min, lambda_max)
                async_df = pd.DataFrame(async_cut, index=lam_async, columns=lam_async)

            # base spectrum (for the 1D plot)
            if ds.mre is not None:
                base_vec = ds.mre[0, :]
            else:
                base_vec = ds.cd_mdeg[0, :] / 1000.0

            base_series = pd.Series(base_vec, index=lam).loc[sync_df.columns]

            base_name = (name or "").strip() or f"dataset_{idx}"
            out_png = str(output_dir / f"{base_name}_2DCOS_combined.png")

            # --- Peak settings: robust + no crashes when mark_peaks_* is False
            if mark_peaks_sync:
                n_diag = _clamp_peaks(_coerce_int(n_sync_diag_peaks, "n_sync_diag_peaks"))
                n_cmax = _clamp_peaks(_coerce_int(n_sync_cross_max_peaks, "n_sync_cross_max_peaks"))
                n_cmin = _clamp_peaks(_coerce_int(n_sync_cross_min_peaks, "n_sync_cross_min_peaks"))
            else:
                # Important: do NOT coerce user values at all -> avoids crashing on "abc"
                n_diag = 0
                n_cmax = 0
                n_cmin = 0

            if mark_peaks_async:
                n_acmax = _clamp_peaks(_coerce_int(n_async_cross_max_peaks, "n_async_cross_max_peaks"))
                n_acmin = _clamp_peaks(_coerce_int(n_async_cross_min_peaks, "n_async_cross_min_peaks"))
            else:
                n_acmax = 0
                n_acmin = 0

            # If someone left mark_peaks_* True but all counts are 0, just disable (no-op)
            eff_mark_sync = bool(mark_peaks_sync and (n_diag + n_cmax + n_cmin > 0))
            eff_mark_async = bool(mark_peaks_async and (n_acmax + n_acmin > 0))

            print(f"   λ range: {sync_df.index.min():.1f} – {sync_df.index.max():.1f} nm")
            print(f"   output figure: {out_png}")
            print(
                "   peaks: "
                f"sync={'ON' if eff_mark_sync else 'OFF'} "
                f"(diag={n_diag}, max={n_cmax}, min={n_cmin}), "
                f"async={'ON' if eff_mark_async else 'OFF'} "
                f"(max={n_acmax}, min={n_acmin}), "
                f"mirror={'ON' if mark_mirror_peaks else 'OFF'}"
            )

            combine_plots(
                spec_syn=sync_df,
                spec_asyn=async_df,
                base_series=base_series,
                dataset_label=base_name,
                out_path=out_png,
                cmap=colormap,
                mark_mirror_peaks=mark_mirror_peaks,
                mark_peaks_sync=eff_mark_sync,
                n_sync_diag_peaks=n_diag,
                n_sync_cross_max_peaks=n_cmax,
                n_sync_cross_min_peaks=n_cmin,
                mark_peaks_async=eff_mark_async,
                n_async_cross_max_peaks=n_acmax,
                n_async_cross_min_peaks=n_acmin,
            )

            print(f"✅ Visualization: figure saved -> {out_png}")

        except Exception as exc:
            msg = str(exc) or exc.__class__.__name__
            print(f" Visualization failed for dataset '{name}': {msg}")
            errors.append((name, msg))

    if errors:
        failed = ", ".join(n for n, _ in errors)
        raise RuntimeError(
            " Visualization finished with errors for some datasets.\n"
            f"   Failed: {failed}\n"
            "   See messages above for details."
        )

