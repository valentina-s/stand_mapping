import numpy as np
from skimage.filters import sobel

def slope_from_dem(dem, res, degrees=False):
    """Calculates slope from a Digital Elevation Model using a Sobel filter.

    Parameters
    ----------
    dem : array
      a Digital Elevation Model
    res : numeric
      spatial resolution of the Digital Elevation Model
    degrees : bool
      whether to return the slope as a percent (default) or to convert to
      degrees

    Returns
    -------
    slope : array
      slope of DEM
    """
    slope = sobel(dem) / (res*2)
    if degrees:
        slope = np.rad2deg(np.arctan(slope))
    return slope


def classify_slope_position(tpi, slope):
    """Classifies an image of normalized Topograhic Position Index into 6 slope
    position classes:

    1 = Valley
    2 = Lower Slope
    3 = Flat Slope
    4 = Middle Slope
    5 = Upper Slope
    6 = Ridge

    Classification following Weiss, A. 2001. "Topographic Position and
    Landforms Analysis." Poster presentation, ESRI User Conference, San Diego,
    CA.  http://www.jennessent.com/downloads/tpi-poster-tnc_18x22.pdf

    Parameters
    ----------
    tpi : array
      TPI values, assumed to be normalized to have mean = 0 and standard
      deviation = 1
    slope : array
      slope of terrain, in degrees
    """
    assert tpi.shape == slope.shape
    pos = np.empty(tpi.shape, dtype=int)

    pos[(tpi<=-1)] = 1
    pos[(tpi>-1)*(tpi<-0.5)] = 2
    pos[(tpi>-0.5)*(tpi<0.5)*(slope<=5)] = 3
    pos[(tpi>-0.5)*(tpi<0.5)*(slope>5)] = 4
    pos[(tpi>0.5)*(tpi<=1.0)] = 5
    pos[(tpi>1)] = 6

    return pos


def classify_landform(tpi_near, tpi_far, slope):
    """Classifies a landscape into 10 landforms given "near" and "far" values
    of Topographic Position Index (TPI) and a slope raster.

    1 = canyons, deeply-incised streams
    2 = midslope drainages, shallow valleys
    3 = upland drainages, headwaters
    4 = U-shape valleys
    5 = plains
    6 = open slopes
    7 = upper slopes, mesas
    8 = local ridges, hills in valleys
    9 = midslope ridges, small hills in plains
    10 = mountain tops, high ridges

    Classification following Weiss, A. 2001. "Topographic Position and
    Landforms Analysis." Poster presentation, ESRI User Conference, San Diego,
    CA.  http://www.jennessent.com/downloads/tpi-poster-tnc_18x22.pdf

    Parameters
    ----------
    tpi_near : array
      TPI values calculated using a smaller neighborhood, assumed to be
      normalized to have mean = 0 and standard deviation = 1
    tpi_far : array
      TPI values calculated using a smaller neighborhood, assumed to be
      normalized to have mean = 0 and standard deviation = 1
    slope : array
      slope of terrain, in degrees
    """
    assert tpi_near.shape == tpi_far.shape == slope.shape
    lf = np.empty(tpi_near.shape, dtype=int)

    lf[(tpi_near<1)*(tpi_near>-1)*(tpi_far<1)*(tpi_far>-1)*(slope<=5)] = 5
    lf[(tpi_near<1)*(tpi_near>-1)*(tpi_far<1)*(tpi_far>-1)*(slope>5)] = 6
    lf[(tpi_near<1)*(tpi_near>-1)*(tpi_far>=1)] = 7
    lf[(tpi_near<1)*(tpi_near>-1)*(tpi_far<=-1)] = 4
    lf[(tpi_near<=-1)*(tpi_far<1)*(tpi_far>-1)] = 2
    lf[(tpi_near>=1)*(tpi_far<1)*(tpi_far>-1)] = 9
    lf[(tpi_near<=-1)*(tpi_far>=1)] = 3
    lf[(tpi_near<=-1)*(tpi_far<=-1)] = 1
    lf[(tpi_near>=1)*(tpi_far>=1)] = 10
    lf[(tpi_near>=1)*(tpi_far<=-1)] = 8

    return lf

LANDFORM_PALETTE = np.array(
    [[0,0,0],[49,54,159],[69,117,180],[116,173,209],[171,217,233],
     [255,255,191],[254,224,144],[253,174,97],[244,109,67],[215,48,39],
     [165,0,38]])

LANDFORM_NAMES = {
    1: 'canyons, deeply-incised streams',
    2: 'midslope drainages, shallow valleys',
    3: 'upland drainages, headwaters',
    4: 'U-shape valleys',
    5: 'plains',
    6: 'open slopes',
    7: 'upper slopes, mesas',
    8: 'local ridges, hills in valleys',
    9: 'midslope ridges, small hills in plains',
    10: 'mountain tops, high ridges'
    }


# indices = np.indices(dem.shape)
# indices = np.stack([indices[1,:,:], indices[0,:,:]])
# CS = ax.contour((trf*indices)[0], (trf*indices)[1], dem_ft,
#                 colors='k', alpha=1.0,
#                 levels=np.arange(dem_ft.min().round(-10)-10, dem_ft.max()+5, 100),
#                 linewidths=0.25)
#
# CS = ax.contour((trf*indices)[0], (trf*indices)[1], dem_ft,
#                 colors='k', alpha=0.5,
#                 levels=np.arange(dem_ft.min().round(-10)-10, dem_ft.max()+5, 25),
#                 linewidths=0.25)
