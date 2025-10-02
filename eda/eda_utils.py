import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cmocean as cmo
from pandas import DataFrame


# # defining the labels for each band
# labels = [
#             "B1 - 60 m - 443 nm - Ultra Blue (Coastal and Aerosol)",
#             "B2 - 10 m - 490 nm - Blue",
#             "B3 - 10 m - 560 nm - Green",
#             "B4 - 10 m - 665 nm - Red",
#             "B5 - 20 m - 705 nm - VNIR",
#             "B6 - 20 m - 740 nm - VNIR",
#             "B7 - 20 m - 783 nm - VNIR",
#             "B8 - 10 m - 842 nm - VNIR",
#             "B8a - 20 m - 865 nm - VNIR",
#             "B9 - 60 m - 940 nm - SWIR",
#             "B11 - 20 m - 1610 nm - SWIR",
#             "B12 - 20 m - 2190 nm - SWIR"
# ]

def load_data(data_path):
    data = np.load(data_path)
    data_copy = copy.deepcopy(data)
    return data_copy

def replace_with_nan(arr: np.ndarray, no_data_value = 0):
    nan_replaced = arr.astype(float, copy=True)
    nan_replaced[arr == no_data_value] = np.nan # dam i love the automcomplete here lol
    return nan_replaced

def minmax_stretch(img):
    low = np.nanmin(img)
    high = np.nanmax(img)
    output = (img-low)/(high-low)
    return np.clip(output, 0, 1).astype(np.float32) #ensuring consistency in datatypes and clipping the output since did normalization.


def percentile_stretch(img, low_perc=10, high_per=90):
    low = np.nanpercentile(img, low_perc)
    high = np.nanpercentile(img, high_per)
    if not np.isfinite(low) or not np.isfinite(high) or high<=low:
        return np.zeros_like(img, dtype=np.float32)
    output = (img-low)/(high-low)
    return np.clip(output, 0, 1).astype(np.float32)

def plot_band(img: np.ndarray,
              labels: np.ndarray,
              band_i: int,
              cmap_name: str = "thermal",
              stretch: tuple | None = ("percentile",3,97),
              show_colorbar: bool = True,
              title: str | None = None,
              nodata: float | int | None = None
              ) -> None:
    img = img[:,:,band_i].astype(np.float32, copy=True) # plotting per band
    if nodata is not None:
        img[img == nodata] = np.nan
    if stretch is None:
        img_stretch = img
    elif stretch[0] == "percentile":
        img_stretch = percentile_stretch(img, stretch[1],stretch[2])
    elif stretch[0] == "minmax":
        img_stretch = minmax_stretch(img)
    else:
        raise ValueError("stretch must be: 'percentile' or 'minmax' or None")
    cmap = getattr(cmo.cm, cmap_name).copy()
    cmap.set_bad('black') # setting the nans to black. But I do not understand why they show up as white, maybe probably due to
    plt.imshow(img_stretch, cmap = cmap)
    plt.title(title or f"{labels[band_i]} - {stretch[0]} = {stretch[1]}-{stretch[2]}%")
    plt.axis("off")
    if show_colorbar:
        plt.colorbar()
    plt.show()
    return None

def calculate_band_statistics(img: np.ndarray, labels: np.ndarray) -> DataFrame:
    stats=[]
    for i in range(img.shape[-1]):
        band = img[:,:,i]
        label = labels[i]
        mean = np.nanmean(band)
        std = np.nanstd(band)
        q1, median, q3 = np.nanpercentile(band, [25, 50, 75])
        #statistical measures derived from moments that describe the shape of a frequency distribution
        skew = np.nanmean(((band-mean)/std)**3) # degree of assymetry around the mean (positive => right skewed, bright anomalies for us. High reflectance, so either rooftops and clouds and stuff) (negative => left skewed or dull anomalies or shadowed or water stuff)
        kurt = np.nanmean(((band-mean)/std)**4) #(>3 outliers more than normal) (degree of peakedness or heaviness of tails relative to normal distribution)(<3 fewer outliers than normal)
        stats.append([label, mean, median, q1, q3, std, skew, kurt])
    cols = ['label', 'mean', 'median', 'q1', 'q3', 'std', 'skew', 'kurt']
    band_stats_df = pd.DataFrame(stats, columns=cols)
    return pd.DataFrame(stats, columns=cols)

def standardize(img: np.ndarray):
    """
    standardize data using z-scores
    Why bother?
    From the above stats, every band has varying means and variances.
    Standardizing the bands would make it possible for us to compare one band with another. Will help us identify anomalies. it would just standardize around the respective means, letting us properly assess the relative distributions, and easily identify the outliers.
    And then plotting histograms for the before (raw reflectances) & after (standardized values z-scores with
    centered around mean and normalized with std dev
    :param img:
    :return: np.ndarray of standardized values with same shape as img
    """
    ht, wd, band = img.shape
    z = np.empty((ht, wd, band), dtype=np.float32)
    for b in range(band):
        band = img[:,:,b]
        mu = np.nanmean(band)
        sigma = np.nanstd(band)
        if not np.isfinite(sigma) or sigma == 0: #egde case to handle no data or whatevs
            z[:,:,b] = np.nan
        else:
            z[:,:,b] = (band-mu)/sigma
    return z

def histogram_before_after_standardization_z_score(img_raw: np.ndarray, z: np.ndarray, b: int, label: str, bins: int = 120, z_thresh: float = 2.2  ):
    # |z|>k is a good heuristic for anomalies detections in rs. experimented with 0.5 till 3.0
    # Settled with around 2.2, where datapoints out of the ~80% of the whole distribution are being considered anomalies.
    raw = img_raw[:,:,b].ravel()
    raw = raw[np.isfinite(raw)]
    z = z[:,:,b].ravel() # flatten to 1D to create histogram
    z = z[np.isfinite(z)]

    fig, axes = plt.subplots(1,2,figsize=(12,4))
    axes[0].hist(raw, bins=bins,color = "blue")
    axes[0].set_title(f"{label} - before")
    axes[0].set_xlabel("raw_before")
    axes[0].set_ylabel("pixel_count")

    axes[1].hist(z, bins=bins,color = "red")
    axes[1].axvline(-z_thresh, ls="--", lw=1, color="black")
    axes[1].axvline(z_thresh, ls="--", lw=1, color="black")
    axes[1].set_title(f"{label} - z-scores (|z| > {z_thresh})")
    axes[1].set_xlabel("after_z_score")
    axes[1].set_ylabel("pixel_count")

    plt.tight_layout()
    plt.show()

# I created another function that would help me visualize the raw, standardized, and z-threshold image of a band
def show_band_vs_z(img: np.ndarray,
                   z: np.ndarray,
                   band_i: int,
                   z_thresh: float = 2.2,
                   raw_stretch=("percentile", 2, 98)):
    """
    Tried to visualize one band: raw reflectance (stretched), its z-scores,
    and an overlay showing outliers based on |z| >= z_thresh.
    """

    # extract band slices
    raw = img[:, :, band_i].astype(float)
    z_b = z[:, :, band_i].astype(float)

    # stretch the raw band for better visualization
    if raw_stretch is None:
        raw_vis = raw
    elif raw_stretch[0] == "percentile":
        raw_vis = percentile_stretch(raw, raw_stretch[1], raw_stretch[2])
    else:  # "minmax"
        raw_vis = minmax_stretch(raw)

    # outlier mask: True where z-scores are above threshold
    mask = np.abs(z_b) >= z_thresh

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # left side: stretched raw band
    ax[0].imshow(raw_vis, cmap="cmo.balance")
    ax[0].set_title(f"Band {band_i} raw (stretched)")
    ax[0].axis("off")

    # middle: z-scores
    im = ax[1].imshow(z_b, cmap="seismic", vmin=-3, vmax=3)
    ax[1].set_title(f"Band {band_i} z-scores")
    ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="z")

    # right side: overlay raw with outliers highlighted
    ax[2].imshow(raw_vis, cmap="gray")
    ax[2].imshow(mask, cmap="autumn", alpha=0.45)  # red overlay = outliers
    ax[2].set_title(f"Outliers |z| ≥ {z_thresh}")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

def correlation_matrix(img: np.ndarray,
                       band_labels: list[str] | np.ndarray | None,
                       text_label: bool,
                       label_each_band: bool,
                       yes_show: bool,
                       cmocean: str) -> np.ndarray:
    # flatten to 2D and drop rows with nans
    x = img.reshape(-1, img.shape[-1])
    drop_nans = np.all(np.isfinite(x), axis=1)
    x_no_nans = x[drop_nans]

    # pearson correlation across pixels (bands as variables)
    corr = np.corrcoef(x_no_nans, rowvar=False)

    # colormap by name ( "ice", "thermal")
    cmap = getattr(cmo.cm, cmocean)

    # plot
    plt.figure(figsize=(12, 12))
    im = plt.imshow(corr, vmin=-1, vmax=1, cmap=cmap)
    plt.colorbar(im, label="pearson_r")

    b = corr.shape[0]

    # labels on axes
    if label_each_band:
        if band_labels is not None:
            lab = np.asarray(band_labels)
            # pretty-print numeric wavelengths
            if np.issubdtype(lab.dtype, np.number):
                lab = [f"{float(v):.0f}" for v in lab]
            else:
                lab = [str(v) for v in lab]
            plt.xticks(range(b), lab, rotation=80, fontsize=8)
            plt.yticks(range(b), lab, fontsize=8)
        else:
            plt.xticks(range(b))
            plt.yticks(range(b))
    else:
        plt.xticks([]); plt.yticks([])

    # optional per-cell text (r values)
    if text_label:
        for i in range(b):
            for j in range(b):
                val = corr[i, j]
                plt.text(j, i, f"{val:.2f}",
                         ha="center", va="center",
                         color="black" if abs(val) < 0.50 else "yellow",
                         fontsize=8 if abs(val) < 0.50 else 12)

    plt.title("band-band_correlation_matrix")
    plt.tight_layout()
    if yes_show is True:
        plt.show()
    # plt.show()
    return corr

#
# def correlation_matrix(img: np.ndarray,
#                        band_labels: list[str],
#                        text_label: bool,
#                        label_each_band: bool,
#                        cmocean: str ) -> np.ndarray:
#     x = img.reshape(-1, img.shape[-1]) #convert 3d to 2d or flatten it to 2d
#     drop_nans = np.all(np.isfinite(x), axis=1)
#     x_no_nans = x[drop_nans] #dropped the nans
#     corr = np.corrcoef(x_no_nans,rowvar=False) #
#     plt.figure(figsize=(12,12))
#     corr_plt = plt.imshow(corr, vmin=-1, vmax=1, cmap=cmo.cm.cmocean)
#     plt.colorbar(corr_plt, label="pearson_r")
#     b = corr.shape[0]
#     if label_each_band:
#         if band_labels is not None:
#             plt.xticks(range(b), band_labels, rotation=80, fontsize=8)
#             plt.yticks(range(b), band_labels, fontsize=8)
#         else:
#             plt.xticks(range(b))
#             plt.yticks(range(b))
#     if text_label:
#         for i in range(b):
#             for j in range(b):
#                 plt.text(j, i, f"{corr[i, j]:.2f}",
#                          horizontalalignment="center",
#                          verticalalignment="center",
#                          color="black" if abs(corr[i, j]) < 0.50 else "yellow",
#                          fontsize=8 if abs(corr[i, j]) < 0.50 else 12)
#     plt.title("band-band_correlation_matrix")
#     plt.tight_layout()
#     plt.show()
#     return corr

def correlation_plot(img: np.ndarray,
                     band_indices: list[int],
                     band_labels: list[str] | None = None,
                     sample: int = 100_000,
                     density: str = "hexbin",
                     gridsize: int = 80, # for hexbin
                     bins: int = 120): # for hist2d
    """
    Creating for every pair of bands:
        - plot 1 (left side): scatter plot that is created by subsampling.
        - plot 2 (right side): density plot where I include hexbin and hist2d, which I had read about in perplexity summaries and viewed the example images of
                               decided to go with hexbin finally. Felt intuitive to understand.
    img: (height, width, band) array with nans for no data.
    band_indices: list of band indices
    band_labels: list of band labels to be shown on the axes of every plot.
                 default value given to be B{i}.
    """
    assert img.ndim == 3, f"expect (height, width, channels) shape: {img.shape}"
    assert len(band_indices) >= 2, "band_indices must contain at least 2 elements"

    H, W, B = img.shape
    for idx in band_indices:
        if not (0 <= idx < B):
            raise ValueError(f"band index {idx} must be in [0, {B-1}]")

    X = img.reshape(-1, B)[:, band_indices]
    valid = np.all(np.isfinite(X), axis=1)
    X = X[valid]
    if X.size == 0:
        raise ValueError("there are 0 valid pixels after filtering out nans (no data)")

    if band_labels is None:
        ax_lbls = [f"B{idx}" for idx in band_indices]  # fixed
    else:
        if len(band_labels) != len(band_indices):
            raise ValueError(f"band_labels must match band_indices length ({len(band_indices)})")
        ax_lbls = band_labels

    k = len(band_indices)
    pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
    rng = np.random.default_rng(0) #for scatter the sample limit points only for speed, (since we have close to 700k points, i sample upto 100k)

    for i, j in pairs:
        xi, xj = X[:, i], X[:, j]  # grabbing all reflectances for band i and j; xi is vector of for band i, and similarly for xj
        n = xi.shape[0]
        if (sample is None) or (sample >= n):
            xs, ys = xi, xj
        else:
            sel = rng.choice(n, sample, replace=False)
            xs, ys = xi[sel], xj[sel]


        r = float(np.corrcoef(xi, xj)[0, 1])
        xmin, xmax = np.nanpercentile(xi, [0.5, 99.5])
        ymin, ymax = np.nanpercentile(xj, [0.5, 99.5])

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].plot(xs, ys, '.', ms=1, alpha=0.3)  # scatter points at (xi, xj)
        ax[0].set_xlim(xmin, xmax); ax[0].set_ylim(ymin, ymax) # only plotting 0.5 - 99.5 percentile values to reduce extreme outliers
        ax[0].set_aspect('equal', 'box')
        ax[0].set_xlabel(ax_lbls[i]); ax[0].set_ylabel(ax_lbls[j])
        ax[0].set_title(f"scatter - {ax_lbls[i]} vs {ax_lbls[j]}\npearson r = {r:.2f}")

        if density == "hexbin":
            hb = ax[1].hexbin(xi, xj, gridsize=gridsize, mincnt=1, cmap=cmo.cm.thermal)
            fig.colorbar(hb, ax=ax[1], label="counts")
        elif density == "hist2d":
            h, xe, ye, im = ax[1].hist2d(xi, xj, bins=bins, cmap=cmo.cm.thermal)
            fig.colorbar(im, ax=ax[1], label="counts")
        else:
            raise ValueError("density must be 'hexbin' or 'hist2d'")

        ax[1].set_xlim(xmin, xmax); ax[1].set_ylim(ymin, ymax)
        ax[1].set_aspect('equal', 'box')
        ax[1].set_xlabel(ax_lbls[i]); ax[1].set_ylabel(ax_lbls[j])
        ax[1].set_title(f"density ({density})")

        plt.tight_layout(); plt.show()

# computing the spectral angle mapper: using the forumla. If low angle then that spectra more similar. To be applied b/w each s2 spectrum & ecostress reference spectrum
def sam(v1, v2):
    x = np.nansum(v1 * v2)
    norm1 = np.sqrt(np.nansum(v1**2))
    norm2 = np.sqrt(np.nansum(v2**2))
    denom = norm1 * norm2
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return np.arccos(np.clip(x / denom, -1.0, 1.0))




















