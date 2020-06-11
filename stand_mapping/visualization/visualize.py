from matplotlib import pyplot as plt
import numpy as np
from affine import Affine


def draw_contours(dem,
                  transform=Affine(1, 0, 0, 0, -1, 0),
                  interval=10,
                  ax=None,
                  **kwargs):
    """Returns a set of (optionally) geo-referenced contours from a Digital
    Elevation Model.

    Additional keyword arguments are passed to `matplotlib.axes.Axes.contour`.

    Parameters
    ----------
    dem : array
      Digital Elevation Model (DEM)
    transform : affine.Affine instance
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
