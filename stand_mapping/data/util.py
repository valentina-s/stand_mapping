import numpy as np
from skimage.filters import sobel
from multiprocessing.pool import ThreadPool
from functools import partial


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

    =======  ============
    Slope #  Description
    =======  ============
    1        Valley
    2        Lower Slope
    3        Flat Slope
    4        Middle Slope
    5        Upper Slope
    6        Ridge
    =======  ============

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

    ==========  ======================================
    Landform #   Description
    ==========  ======================================
    1           canyons, deeply-incised streams
    2           midslope drainages, shallow valleys
    3           upland drainages, headwaters
    4           U-shape valleys
    5           plains
    6           open slopes
    7           upper slopes, mesas
    8           local ridges, hills in valleys
    9           midslope ridges, small hills in plains
    10          mountain tops, high ridges
    ==========  ======================================

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


def multi_to_single_linestring(geom):
    """Converts a MultiLineString geometry into a single LineString

    Parameters
    ----------
    geom : LineString or MultiLineString
      a LineString or MultiLineString geometry object

    Returns
    -------
    ls : LineString
      LineString based on connecting lines within MultiLineString in the same
      order they are originally read.
    """
    if type(geom) == MultiLineString:
        coords = [list(line.coords) for line in geom]
        ls = LineString([x for sublist in coords for x in sublist])
    elif type(geom) == LineString:
        ls = geom
    else:
        raise TypeError

    return ls


def quad_naip_from_tnm(bbox, res, bboxSR=4326, imageSR=4326, **kwargs):
    """Retrieves NAIP imagery from The National Map by breaking bbox into four
    quadrants and requesting four images and returning them stitched together.

    This can be used, for example, to retrieve images that are high resolution
    and which would be too large to retrieve in a single request to the web
    service.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    res : numeric
      spatial resolution to use for returned DEM (grid cell size)
    bboxSR : integer
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    imageSR : integer
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)

    Returns
    -------
    img : array
      NAIP image as a 3-band or 4-band array
    """
    xmin, ymin, xmax, ymax = bbox
    nw_bbox = [xmin, (ymin + ymax) / 2, (xmin + xmax)/2, ymax]
    ne_bbox = [(xmin + xmax)/2, (ymin + ymax)/2, xmax, ymax]
    sw_bbox = [xmin, ymin, (xmin + xmax)/2, (ymin + ymax)/2]
    se_bbox = [(xmin + xmax)/2, ymin, xmax, (ymin + ymax)/2]

    bboxes = [nw_bbox, ne_bbox, sw_bbox, se_bbox]

    get_naip = partial(naip_from_tnm, res=res,
                       bboxSR=bboxSR, imageSR=imageSR,
                       **kwargs)

    with ThreadPool(4) as p:
        naip_images = p.map(get_naip, bboxes)
    nw, ne, sw, se = naip_images
    img = np.vstack([np.hstack([nw, ne]), np.hstack([sw, se])])

    return img


def quad_fetch(fetcher, bbox, num_threads=4, *args, **kwargs):
    """Breaks user-provided bounding box into quadrants and retrieves data
    using `fetcher` for each quadrant in parallel using a ThreadPool.

    Parameters
    ----------
    fetcher : callable
      data-fetching function, expected to return an array-like object
    bbox : 4-tuple or list
      coordinates of x_min, y_min, x_max, and y_max for bounding box of tile
    num_threds : int
      number of threads to use for parallel executing of data requests
    *args
      additional positional arguments that will be passed to `fetcher`
    **kwargs
      additional keyword arguments that will be passed to `fetcher`

    Returns
    -------
    quad_img : array
      image return with quads stitched together into a single array

    """
    xmin, ymin, xmax, ymax = bbox
    nw_bbox = [xmin, (ymin + ymax) / 2, (xmin + xmax)/2, ymax]
    ne_bbox = [(xmin + xmax)/2, (ymin + ymax)/2, xmax, ymax]
    sw_bbox = [xmin, ymin, (xmin + xmax)/2, (ymin + ymax)/2]
    se_bbox = [(xmin + xmax)/2, ymin, xmax, (ymin + ymax)/2]

    bboxes = [nw_bbox, ne_bbox, sw_bbox, se_bbox]

    get_quads = partial(fetcher, *args, **kwargs)

    with ThreadPool(num_threads) as p:
        quads = p.map(get_quads, bboxes)
        nw, ne, sw, se = quads

    quad_img = np.vstack([np.hstack([nw, ne]), np.hstack([sw, se])])

    return quad_img
