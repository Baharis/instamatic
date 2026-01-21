from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy import ndimage as ndi


def ring_threshold_detection(
    frame: np.ndarray,
    min_radius: int = 40,
    percentile: float = 99.0,
    threshold_mult: float = 3.0,
    min_peak_count: int = 10,
    min_peak_sep: int = 5,
    mask: np.ndarray | None = None,
    n_bins: int = 10,
):
    """Fast diffraction detector with radial-binned background subtraction.

    - estimate center of incident beam based on a blurred central ROI
    - excludes (mask == False) and excludes rr <= beam_radius_px regions
    - estimates background and reflection threshold in `n_bins` radial shells
    - reflections must exceed `percentile` * `threshold_mult` of their ring
    - the algorithm is good at finding a small number of strongest reflections
    """

    cy, cx = estimate_beam_center(frame, sigma=3.0)
    ys, xs = np.indices(frame.shape)
    rr = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    valid = mask.astype(bool) if mask is not None else np.ones(frame.shape, dtype=bool)
    valid &= rr > min_radius

    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return {'has_diffraction': False, 'n_peaks': 0, 'mask': valid, 'center': (cy, cx)}

    vals = frame.flat[valid_idx].astype(np.float32, copy=False)
    rr_vals = rr.flat[valid_idx].astype(np.float32, copy=False)

    r_max = float(rr_vals.max())
    bin_edges = np.linspace(min_radius, r_max, n_bins + 1).astype(np.float32)

    bin_ids = np.digitize(rr_vals, bin_edges) - 1
    np.clip(bin_ids, 0, n_bins - 1, out=bin_ids)

    # Per-bin bg and thresholds (still a small loop)
    backgrounds = np.zeros(n_bins, dtype=np.float32)
    thresholds = np.full(n_bins, np.inf, dtype=np.float32)

    # residuals in 1D
    resid_all = np.zeros_like(vals, dtype=np.float32)

    for b in range(n_bins):
        sel = bin_ids == b
        if not np.any(sel):
            continue
        v = vals[sel]
        bg = np.median(v)
        backgrounds[b] = bg

        r = v - bg
        r[r < 0] = 0.0
        resid_all[sel] = r

        thresholds[b] = threshold_mult * np.percentile(r, percentile) if r.size else np.inf

    # Candidate selection purely in 1D
    keep = resid_all >= thresholds[bin_ids]

    # Scatter to 2D only once (needed for clustering / argmax)
    processed = np.zeros(frame.shape, dtype=np.float32)
    processed.flat[valid_idx] = resid_all

    peak_mask = np.zeros(frame.shape, dtype=bool)
    peak_mask.flat[valid_idx[keep]] = True

    peaks = peaks_one_per_cluster(peak_mask, processed, min_dist=min_peak_sep)
    n_peaks = int(peaks.shape[0])

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        'has_diffraction': n_peaks >= min_peak_count,
        'n_peaks': n_peaks,
        'peaks': peaks,
        'center': (cy, cx),
        'beam_radius': min_radius,
        'thr_per_bin': thresholds,
        'mask': valid,
        'radial_bin_edges': bin_edges,
        'radial_bin_centers': bin_centers,
        'bg_per_bin': backgrounds,
        'n_radial_bins': n_bins,
    }


def estimate_beam_center(frame: np.ndarray, sigma: float = 3.0):
    """Estimate beam center by Gaussian-blurring a small ROI and taking max."""
    h, w = image.shape
    cy0, cx0 = np.unravel_index(np.argmax(frame), frame.shape)

    y0 = max(0, cy0 - h // 8)
    y1 = min(h - 1, cy0 + h // 8)
    x0 = max(0, cx0 - w // 8)
    x1 = min(w - 1, cx0 + w // 8)

    roi = image[y0:y1, x0:x1].astype(np.float32)
    roi_blur = ndi.gaussian_filter(roi, sigma=sigma, mode='nearest')
    iy, ix = np.unravel_index(np.argmax(roi_blur), roi_blur.shape)
    return y0 + int(iy), x0 + int(ix)


def peaks_one_per_cluster(hot_mask: np.ndarray, intensity: np.ndarray, min_dist: int = 5):
    """
    hot_mask: boolean mask of candidate pixels (True = candidate)
    intensity: float/int image used to pick the representative (processed)
    min_dist: merge radius (pixels). Pixels within ~min_dist get clustered.
    Returns: (K,2) array of (y,x) peak positions (one per cluster)
    """
    if not hot_mask.any():
        return np.empty((0, 2), dtype=int)

    # Merge nearby pixels into clusters
    structure = ndi.generate_binary_structure(2, 1)  # 4-connectivity
    dilated = ndi.binary_dilation(hot_mask, iterations=min_dist, structure=structure)

    # Label clusters
    lbl, n = ndi.label(dilated, structure=structure)
    if n == 0:
        return np.empty((0, 2), dtype=int)

    peaks = []
    for k in range(1, n + 1):
        region = (lbl == k) & hot_mask  # restrict back to original hot pixels
        ys, xs = np.nonzero(region)
        if ys.size == 0:
            continue
        vals = intensity[ys, xs]
        j = int(np.argmax(vals))
        peaks.append((int(ys[j]), int(xs[j])))

    return np.array(peaks, dtype=int)


def plot_diffraction_debug(image, results):
    """Visualize diffraction detection results.

    - grayscale log image
    - green dots: all detected peaks
    - red dots: peaks outside central beam
    - cyan dot: center
    """

    # --- log-scaled image ---
    img_log = np.log10(image.astype(np.float32) + 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_log, cmap='gray')
    ax.set_title('Diffraction detection debug')
    ax.axis('off')

    # --- mask overlay (red, 25% opacity) ---
    mask = results.get('mask', None)
    if mask is not None:
        # mask == False → excluded area
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[~mask] = (1.0, 0.0, 0.0, 0.25)  # RGBA
        ax.imshow(overlay)

    # --- bins ---
    cy, cx = center = results.get('center', None)
    h, w = image.shape
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    edges = results.get('radial_bin_edges', None)
    for r in edges:
        ax.add_patch(Circle((cx, cy), float(r), fill=False, linewidth=0.5, clip_on=True))

    # --- peaks ---
    peaks = results.get('peaks', np.empty((0, 2)))
    if len(peaks):
        ax.scatter(peaks[:, 1], peaks[:, 0], s=2, c='lime', marker='o', label='peaks')

    # --- center ---
    center = results.get('center', None)
    if center is not None:
        ax.scatter(center[1], center[0], s=4, c='cyan', marker='s', label='center_pos')

    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.show()


def make_cross_mask():
    c = 511 / 2
    yy, xx = np.indices((512, 512))
    vertical = np.abs(xx - c) <= 1.9
    horizontal = np.abs(yy - c) <= 1.9
    cross = vertical | horizontal
    return ~cross


if __name__ == '__main__':
    from PIL import Image

    for i in range(0, 80):
        path = rf'C:\Users\tchon\x\Instamatic_RATS_cRED_benchmark\instamatic_19\tiff\000{i:02d}.tiff'
        tiff = Image.open(path)
        image = np.array(tiff)

        t0 = time.perf_counter()
        results = ring_threshold_detection(image, mask=make_cross_mask())
        t1 = time.perf_counter()
        print(f'TIME TAKEN: {t1 - t0}')
        print(len(results['peaks']))
        plot_diffraction_debug(image, results)
