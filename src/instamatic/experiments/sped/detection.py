from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy import ndimage as ndi


def detect_diffraction_fast(
    image: np.ndarray,
    beam_radius_px: int = 40,
    q: float = 99.9,
    min_peaks: int = 5,
    mask: np.ndarray | None = None,
    n_radial_bins: int = 20,
    bg_stat: str = 'median',  # "median" or "q"
    bg_q: float = 0.8,  # used if bg_stat == "q"
):
    """Fast diffraction detector with radial-binned background subtraction.

    - center estimated via blurred ROI
    - excludes (mask == False) and excludes rr <= beam_radius_px
    - estimates background as a function of radius using n_radial_bins shells
    - subtracts that per-shell background
    - detects peaks via global quantile threshold on residual
    """

    img = image.astype(np.float32)

    # center (fast, robust)
    cy0, cx0 = np.unravel_index(np.argmax(img), img.shape)
    cy, cx = estimate_beam_center(image, expected=(cy0, cx0), roi_half=64, sigma=3.0)

    # geometry
    yy, xx = np.indices(img.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # combine masks: user mask AND outside beam radius
    valid = mask.astype(bool) if mask is not None else np.ones(img.shape, dtype=bool)
    valid &= rr > beam_radius_px

    vals = img[valid]
    if vals.size == 0:
        return {'has_diffraction': False, 'n_peaks': 0, 'mask': valid, 'center': (cy, cx)}

    # --- radial binning setup ---
    # Bin edges: from beam_radius_px to max radius in valid region
    r_max = float(rr[valid].max())
    bin_edges = np.linspace(beam_radius_px, r_max, n_radial_bins + 1).astype(np.float32)

    # Assign each valid pixel to a bin id in [0, n_radial_bins-1]
    r_valid = rr[valid]
    bin_id = np.digitize(r_valid, bin_edges) - 1
    bin_id = np.clip(bin_id, 0, n_radial_bins - 1)

    # --- per-bin background statistic (robust) ---
    bg_per_bin = np.zeros(n_radial_bins, dtype=np.float32)
    for b in range(n_radial_bins):
        v = vals[bin_id == b]
        if v.size == 0:
            bg_per_bin[b] = 0.0
        else:
            if bg_stat == 'median':
                bg_per_bin[b] = np.median(v)
            elif bg_stat == 'q':
                bg_per_bin[b] = np.quantile(v, bg_q)
            else:
                raise ValueError("bg_stat must be 'median' or 'q'")

    # --- subtract radial background ---
    processed = np.zeros_like(img, dtype=np.float32)
    processed[valid] = vals - bg_per_bin[bin_id]
    processed[processed < 0] = 0.0

    # --- peak threshold via global quantile ---
    pvals = processed[valid]
    if pvals.size == 0:
        return {'has_diffraction': False, 'n_peaks': 0, 'mask': valid, 'center': (cy, cx)}

    thr_per_bin = np.zeros(n_radial_bins, dtype=np.float32)
    for b in range(n_radial_bins):
        v = pvals[bin_id == b]
        thr_per_bin[b] = 2 * np.percentile(v, q) if v.size else np.inf

    # Each valid pixel compares against its bin's threshold
    keep = pvals >= thr_per_bin[bin_id]

    # Build a boolean peak mask efficiently
    peak_mask = np.zeros_like(valid, dtype=bool)
    valid_idx = np.flatnonzero(valid)
    peak_mask.flat[valid_idx[keep]] = True

    peaks = peaks_one_per_cluster(peak_mask, processed, min_dist=5)
    n_peaks = int(peaks.shape[0])

    # Bin radii (useful for plotting)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        'has_diffraction': n_peaks >= min_peaks,
        'n_peaks': n_peaks,
        'peaks': peaks,
        'center': (cy, cx),
        'beam_radius': beam_radius_px,
        'thr_per_bin': thr_per_bin,
        'mask': valid,
        # radial background diagnostics:
        'radial_bin_edges': bin_edges,  # length n_bins+1
        'radial_bin_centers': bin_centers,  # length n_bins
        'bg_per_bin': bg_per_bin,  # length n_bins
        'n_radial_bins': n_radial_bins,
    }


def estimate_beam_center(
    image: np.ndarray,
    expected: tuple[int, int] | None,
    roi_half: int = 64,
    sigma: float = 3.0,
):
    """Estimate beam center by Gaussian-blurring a small ROI and taking its
    maximum.

    expected_center: if None, uses image center
    roi_half: ROI is (2*roi_half) x (2*roi_half)
    sigma: Gaussian sigma in pixels (2–5 is typical)
    """
    h, w = image.shape
    cy0, cx0 = expected

    y0 = max(0, cy0 - roi_half)
    y1 = min(h, cy0 + roi_half)
    x0 = max(0, cx0 - roi_half)
    x1 = min(w, cx0 + roi_half)

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

    for i in range(0, 5):
        path = rf'C:\Users\tchon\x\Instamatic_RATS_cRED_benchmark\instamatic_19\tiff\000{i:02d}.tiff'
        path = rf'C:\Users\tchon\x\granat\experiment_2\tiff\000{i:02d}.tiff'
        tiff = Image.open(path)
        image = np.array(tiff)

        for h in [10, 20]:
            t0 = time.perf_counter()
            results = detect_diffraction_fast(image, n_radial_bins=h, mask=make_cross_mask())
            t1 = time.perf_counter()
            print(f'TIME TAKEN: {t1 - t0}')
            print(len(results['peaks']))
            plot_diffraction_debug(image, results)
