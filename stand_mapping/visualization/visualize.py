from matplotlib import pyplot as plt
import numpy as np
from affine import Affine


def draw_contours(dem, transform=None, interval=10, ax=None, **kwargs):
    """Returns a set of (optionally) geo-referenced contours from a Digital
    Elevation Model.

    Additional keyword arguments are passed to `matplotlib.axes.Axes.contour`.

    Parameters
    ----------
    dem : array
      Digital Elevation Model (DEM)
    transform : Affine
      Affine transformation matrix that will convert row,column coordinates
      into x,y coordinates with appropriate georeferencing. If not provided,
      returned contours are not georeferenced.
    interval : numeric
      interval for contour lines (each contour represents an increase or
      decrease of this value along the DEM).

    Returns
    -------
    cs : QuadContourSet
      contours rendered by `matplotlib` from the DEM
    """
    if transform is None:
        transform = Affine(1, 0, 0, 0, -1, 0)

    # reshape row, column indices into column, row (x, y) order
    indices = np.indices(dem.shape)
    indices = np.stack([indices[1, :, :], indices[0, :, :]])

    if not ax:
        ax = plt.gca()

    cs = ax.contour(
        (transform * indices)[0], (transform * indices)[1],
        dem,
        levels=np.arange((dem.min() // interval) * interval,
                         (dem.max() // interval) * interval + interval,
                         interval),
        **kwargs)

    return cs


def colorize_landcover(img, soft=False):
    COLOR_MAP = {
        0: [1.0, 1.0, 1.0],  # unlabeled but mapped
        1: [0.0, 0.0, 1.0],  # water
        2: [0.0, 0.5, 0.0],  # trees
        3: [0.5, 1.0, 0.5],  # field
        4: [0.5, 0.375, 0.375],  # barren/non-vegetated\
        5: [0.0, 0.0, 0.0],  # developed/building
        6: [1.0, 1.0, 0.0],  # boundaries between objects
        255: [1.0, 0.0, 0.0]  # unmapped, nodata
    }
    cover_colors = np.zeros((img.shape[0], img.shape[1], 3))
    if not soft:
        for cov in np.unique(img):
            mask = img == cov
            cover_colors[mask] = COLOR_MAP[cov]
    else:  # we're given the probability of each class for each pixel
        num_classes = img.shape[-1]
        for i in range(num_classes):
            cover_colors[:, :, 0] += img[:, :, i] * COLOR_MAP[i+1][0]  # R
            cover_colors[:, :, 1] += img[:, :, i] * COLOR_MAP[i+1][1]  # G
            cover_colors[:, :, 2] += img[:, :, i] * COLOR_MAP[i+1][2]  # B

    return (cover_colors * 255).astype(np.uint8)
