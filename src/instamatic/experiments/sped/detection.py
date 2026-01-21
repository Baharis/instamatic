from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import blob_log  # optional alternative
from skimage.measure import label
from skimage.morphology import binary_dilation, disk, opening


def detect_diffraction_peaks(
    image: np.ndarray,
    opening_radius: int = 31,
    central_fraction: float = 0.005,
    central_dilate: int = 5,
    thr_sigma: float = 5.0,
    thr_floor: float = 2.0,
    max_filter_size: int = 3,
    min_distance_px: int = 1,
    method: str = 'localmax',
):
    """A diffraction detection method suggested by ChatGPT to be refined Detect
    candidate diffraction peaks in a single image.

    Parameters
    ----------
    image : 2D ndarray (integer counts)
    opening_radius : radius of disk used to compute background via morphological opening
    central_fraction : fraction of peak used to identify the primary beam component
    central_dilate : dilation radius for central-beam mask (pixels)
    thr_sigma : threshold = median + thr_sigma * std  (adaptive)
    thr_floor : minimal absolute threshold (counts)
    max_filter_size : neighborhood size for local-maximum test
    method : "localmax" (fast) or "blob_log" (scale-aware blob detection)

    Returns
    -------
    result : dict with keys
        n_peaks_total, n_peaks_outside_center, peaks (N x 2 array of (y,x)),
        processed (top-hat with center masked), thr, center_pos, center_radius
    """
    img = image.astype(np.float32)
    peak_pos = tuple(np.unravel_index(np.argmax(img), img.shape))
    peak_val = float(img[peak_pos])

    # background estimate (large-scale)
    selem = disk(opening_radius)
    background = opening(img, selem)

    top_hat = img - background
    top_hat[top_hat < 0] = 0.0

    # central beam mask via connected component containing the peak
    central_thr = peak_val * central_fraction
    central_mask = img > central_thr
    lbl = label(central_mask)
    peak_label = (
        lbl[peak_pos]
        if (0 <= peak_pos[0] < lbl.shape[0] and 0 <= peak_pos[1] < lbl.shape[1])
        else 0
    )
    if peak_label != 0:
        central_comp = lbl == peak_label
        central_comp = binary_dilation(central_comp, footprint=disk(central_dilate))
    else:
        central_comp = np.zeros_like(img, dtype=bool)

    processed = top_hat.copy()
    processed[central_comp] = 0.0

    # threshold
    adaptive_thr = np.median(processed) + thr_sigma * np.std(processed)
    thr = max(thr_floor, adaptive_thr)

    if method == 'localmax':
        neighborhood = ndi.maximum_filter(processed, size=max_filter_size)
        local_max = (processed == neighborhood) & (processed >= thr)
        # remove border pixels
        local_max[0, :] = local_max[-1, :] = local_max[:, 0] = local_max[:, -1] = False
        peaks = np.column_stack(np.nonzero(local_max))
    elif method == 'blob_log':
        # scale-aware detection (might be slower)
        blobs = blob_log(
            processed, min_sigma=1, max_sigma=4, threshold=thr / float(processed.max() + 1e-12)
        )
        # blob_log returns (y, x, sigma)
        peaks = blobs[:, :2].astype(int) if blobs.size else np.empty((0, 2), int)
    else:
        raise ValueError('unknown method')

    total_peaks = peaks.shape[0]

    # central radius (largest distance of any central component pixel from peak)
    comp_coords = np.column_stack(np.nonzero(central_comp))
    if comp_coords.size:
        comp_dists = np.sqrt(((comp_coords - np.array(peak_pos)) ** 2).sum(axis=1))
        center_radius = float(comp_dists.max())
    else:
        center_radius = 0.0

    # count only those peaks with distance > center_radius
    if peaks.shape[0] > 0:
        dists = np.sqrt(((peaks - np.array(peak_pos)) ** 2).sum(axis=1))
        outside_mask = dists > center_radius
        peaks_outside = peaks[outside_mask]
        n_outside = peaks_outside.shape[0]
    else:
        peaks_outside = np.empty((0, 2), int)
        n_outside = 0

    return {
        'n_peaks_total': int(total_peaks),
        'n_peaks_outside_center': int(n_outside),
        'peaks': peaks,
        'peaks_outside': peaks_outside,
        'processed': processed,
        'thr': float(thr),
        'center_pos': peak_pos,
        'center_radius': center_radius,
    }


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

    # --- peaks ---
    peaks = results.get('peaks', np.empty((0, 2)))
    if len(peaks):
        ax.scatter(peaks[:, 1], peaks[:, 0], s=10, c='lime', marker='o', label='peaks')

    # --- peaks outside central beam ---
    peaks_out = results.get('peaks_outside', np.empty((0, 2)))
    if len(peaks_out):
        ax.scatter(
            peaks_out[:, 1], peaks_out[:, 0], s=14, c='red', marker='x', label='peaks outside'
        )

    # --- center ---
    center = results.get('center_pos', None)
    if center is not None:
        ax.scatter(center[1], center[0], s=40, c='cyan', marker='s', label='center_pos')

    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from PIL import Image

    path = r'C:\Users\tchon\x\Instamatic_RATS_cRED_benchmark\instamatic_19\tiff\00020.tiff'
    tiff = Image.open(path)
    image = np.array(tiff)
    print(f'{image.shape=} {image.dtype=} {image.max()=} {image.min()=}')
    t0 = time.perf_counter()
    for i in range(10):
        results = detect_diffraction_peaks(image)
    t1 = time.perf_counter()
    print(f'TIME TAKEN: {t1 - t0}, PER ROUND: {(t1 - t0) / 100}')
    print(results)
    plot_diffraction_debug(image, results)
