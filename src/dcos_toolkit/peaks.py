import numpy as np
import scipy.ndimage as ndimage
import logging

logger = logging.getLogger(__name__)

# PEAK ANNOTATION PARAMETERS
# ------------------------------------------------------

# Cross-peaks
CROSS_THRESHOLD_REL = 0.10  # Noise threshold 
CROSS_MIN_DIST_REL = 0.20   # Minimal distance between peaks

# Diagonal peaks
DIAG_THRESHOLD_REL = 0.02  # Noise threshold
DIAG_MIN_DIST_REL = 0.05   # Minimal distance between peaks

# Local maximum filter size
FILTER_SIZE_REL = 0.03  # Size of the local maximum filter relative to axis length 
                        # Used for avoiding peak detection too close to diagonal
# ------------------------------------------------------

def _apply_peak_params(axis_len, is_diagonal=False):
    # Applies peak detection parameters based on matrix size
    filter_size = max(3, int(axis_len * FILTER_SIZE_REL))
    
    if is_diagonal:
        min_dist = int(axis_len * DIAG_MIN_DIST_REL)
        diag_margin = 0
    else:
        min_dist = int(axis_len * CROSS_MIN_DIST_REL)
        diag_margin = min_dist // 3
        if diag_margin < 2:
            diag_margin = 2
        
    return filter_size, min_dist, diag_margin


def _greedy_selection(candidates, values, n_peaks, min_dist, diag_margin=0):
    # Selects top n_peaks from candidates ensuring min_dist spacing
    if not candidates:
        return []

    coords_y, coords_x = zip(*candidates)
    cand_vals = values[coords_y, coords_x]
    sorted_idx = np.argsort(cand_vals)[::-1]
    
    selected = []
    
    for i in sorted_idx:
        cy, cx = candidates[i]
        
        if diag_margin > 0 and abs(cy - cx) < diag_margin:
            continue

        if any(np.hypot(cy - py, cx - px) < min_dist for py, px in selected):
            continue
            
        selected.append((cy, cx))
        if len(selected) >= n_peaks:
            break
            
    return selected


def add_mirror_peaks(peaks, z_matrix, axis):
    # Add mirror peaks for cross-peaks
    if not peaks: 
        return []
    
    axis = np.asarray(axis)
    def get_val(x_val, y_val):
        xi = np.argmin(np.abs(axis - x_val))
        yi = np.argmin(np.abs(axis - y_val))
        return z_matrix[yi, xi]

    final_peaks = []
    seen = set()

    for px, py, _ in peaks:
        for (ax, ay) in [(px, py), (py, px)]:
            if (ax, ay) not in seen:
                final_peaks.append((ax, ay, get_val(ax, ay)))
                seen.add((ax, ay))
                
    return final_peaks


def find_diagonal_peaks(z, axis, n_peaks):
    # Finds diagonal peaks
    if n_peaks <= 0: return []
    
    diag_data = np.abs(np.diag(z))
    threshold = np.max(diag_data) * DIAG_THRESHOLD_REL 
    
    clean_data = diag_data.copy()
    clean_data[clean_data < threshold] = 0

    filter_size, min_dist, _ = _apply_peak_params(len(axis), is_diagonal=True)

    # 1D filter along diagonal
    local_max = ndimage.maximum_filter1d(clean_data, size=filter_size)
    candidates = np.where((clean_data == local_max) & (clean_data > 0))[0]

    sorted_idx = candidates[np.argsort(clean_data[candidates])[::-1]]
    
    selected = []
    for idx in sorted_idx:
        if len(selected) >= n_peaks: break
        if all(abs(idx - s) >= min_dist for s in selected):
            selected.append(idx)

    return [(axis[i], axis[i], np.diag(z)[i]) for i in selected]


def find_cross_peaks(z, x, y, n_peaks, polarity='max', region='lower'):
    # Finds cross-peaks
    if n_peaks <= 0: return []
    
    work_mat = np.copy(z)
    
    if polarity == 'max':
        work_mat[work_mat < 0] = 0
    elif polarity == 'min':
        work_mat[work_mat > 0] = 0
    
    work_mat = np.abs(work_mat)
    np.fill_diagonal(work_mat, 0) 
    
    # Pick annotation region (upper/lower triangle)
    # Matrix its transposed before visualization, so upper/lower is swapped
    if region == 'upper':
        work_mat = np.tril(work_mat, k=-2)
    elif region == 'lower':
        work_mat = np.triu(work_mat, k=2)

    global_max = np.max(work_mat) if work_mat.size > 0 else 0
    threshold = global_max * CROSS_THRESHOLD_REL
    work_mat[work_mat < threshold] = 0

    if np.all(work_mat == 0):
        return []

    filter_size, min_dist, diag_margin = _apply_peak_params(work_mat.shape[0], is_diagonal=False)

    local_max = ndimage.maximum_filter(work_mat, size=filter_size)
    candidates_mask = (work_mat == local_max) & (work_mat > 0)
    candidates_y, candidates_x = np.where(candidates_mask)
    
    candidates = list(zip(candidates_y, candidates_x))
    selected_yx = _greedy_selection(candidates, work_mat, n_peaks, min_dist, diag_margin)

    return [(x[c], y[r], z[r, c]) for r, c in selected_yx]


def find_cross_peaks_max(z, x, y, n): 
    return find_cross_peaks(z, x, y, n, polarity='max', region='upper')

def find_cross_peaks_min(z, x, y, n): 
    return find_cross_peaks(z, x, y, n, polarity='min', region='upper')