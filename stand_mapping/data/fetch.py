"""
Functions that retrieve images and features from public web services.
"""

import base64
import io
import numpy as np
import requests
import warnings
from bs4 import BeautifulSoup
from functools import partial
from multiprocessing.pool import ThreadPool

from pyproj.crs.crs import CRS
from rasterio import transform, windows
from rasterio.features import rasterize
from shapely.errors import TopologicalError
from shapely.geometry import box, LineString, MultiLineString
import geopandas as gpd
import osmnx as ox

from imageio import imread
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import distance_transform_edt as edt
from skimage.morphology import disk
from skimage.transform import resize
from skimage.util import apply_parallel

from .util import multi_to_single_linestring


def landcover_from_ai4earth(bbox,
                            inSR,
                            api_key,
                            weights=[0.25, 0.25, 0.25, 0.25],
                            prediction_type='hard'):
    """
    Retrieve land cover classification image from the Microsoft AI for Earth,
    v2 API within a user-defined bounding box.

    Parameters
    ----------
    bbox : 4-tuple or list
      coordinates of x_min, y_min, x_max, and y_max for bounding box of tile
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    api_key : str
      key for accessing the AI for Earth Landcover API
    weights : list of 4 numerics
      weights assigned by the land cover predictive model to the four land
      cover classes (water, forest, field, built)
    prediction_type : str
      type of landcover prediction to return, available options are 'hard',
      'soft', or 'both'.

    Returns
    -------
    hard_cover : array
      RGB image with a distinct color assigned to each of four cover types:
      water, forest, field, or impervious.
    soft_cover : array
      RGB image with color determined as a mixture of the colors for each
      cover type based on the predicted probability of each cover type.

    Raises
    ------
    ValueError
      If `prediction_type` is not 'hard', 'soft', or 'both'.
    """
    BASE_URL = 'https://aiforearth.azure-api.net/landcover/v2/classifybyextent'
    api_header = {
        'Ocp-Apim-Subscription-Key': api_key,
        'Content-Type': 'application/json'
    }

    xmin, ymin, xmax, ymax = bbox
    extent = {
        "extent": {
            "xmax": xmax,
            "xmin": xmin,
            "ymax": ymax,
            "ymin": ymin,
            "spatialReference": {
                "latestWkid": inSR
            },
        },
        "weights": weights
    }
    print('Retrieving image from AI for Earth Land Cover API', end='... ')
    r = requests.post(BASE_URL, json=extent, headers=api_header).json()

    hard_cover = imread(io.BytesIO(base64.b64decode(r['output_hard'])))
    soft_cover = imread(io.BytesIO(base64.b64decode(r['output_soft'])))
    print('Done.')

    if prediction_type == 'hard':
        return hard_cover
    elif prediction_type == 'soft':
        return soft_cover
    elif prediction_type == 'both':
        return hard_cover, soft_cover
    else:
        raise ValueError(
            "prediction_type must be one of 'hard', 'soft', or 'both'.")


def naip_from_ai4earth(bbox, inSR, api_key):
    """
    Retrieve 3-band NAIP image from the Microsoft AI for Earth, v2 API within
    a bounding box.

    Parameters
    ----------
    bbox : 4-tuple or list
      coordinates of x_min, y_min, x_max, and y_max for bounding box of tile
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    api_key : str
      key for accessing the AI for Earth Landcover API

    Returns
    -------
    cover : array
      a natural color aerial image
    """
    BASE_URL = 'https://aiforearth.azure-api.net/landcover/v2/tilebyextent'
    api_header = {
        'Ocp-Apim-Subscription-Key': api_key,
        'Content-Type': 'application/json'
    }

    xmin, ymin, xmax, ymax = bbox
    extent = {
        "extent": {
            "xmax": xmax,
            "xmin": xmin,
            "ymax": ymax,
            "ymin": ymin,
            "spatialReference": {
                "latestWkid": inSR
            },
        }
    }
    print('Retrieving image from AI for Earth Land Cover API', end='... ')
    r = requests.post(BASE_URL, json=extent, headers=api_header).json()
    naip = imread(io.BytesIO(base64.b64decode(r['input_naip'])))
    print('Done.')

    return naip


def naip_from_tnm(bbox, res, inSR=4326, **kwargs):
    """
    Retrieves a NAIP image from The National Map (TNM) web service.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    res : numeric
      spatial resolution to use for returned image (grid cell size)
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)

    Returns
    -------
    img : array
      NAIP image as a 3-band or 4-band array
    """
    BASE_URL = ''.join([
        'https://services.nationalmap.gov/arcgis/rest/services/',
        'USGSNAIPImagery/ImageServer/exportImage?'
    ])

    width = int(abs(bbox[2] - bbox[0]) // res)
    height = int(abs(bbox[3] - bbox[1]) // res)

    params = dict(
        bbox=','.join([str(x) for x in bbox]),
        bboxSR=inSR,
        size=f'{width},{height}',
        imageSR=inSR,
        format='tiff',
        pixelType='U8',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        time=None,
        noData=None,
        compression=None,
        compressionQuality=None,
        bandIds=None,
        mosaicRule=None,
        renderingRule=None,
        f='image',
    )
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    img = imread(io.BytesIO(r.content))

    return img


def dem_from_tnm(bbox, res, inSR=4326, **kwargs):
    """
    Retrieves a Digital Elevation Model (DEM) image from The National Map (TNM)
    web service.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    res : numeric
      spatial resolution to use for returned DEM (grid cell size)
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)

    Returns
    -------
    dem : numpy array
      DEM image as array
    """
    width = int(abs(bbox[2] - bbox[0]) // res)
    height = int(abs(bbox[3] - bbox[1]) // res)

    BASE_URL = ''.join([
        'https://elevation.nationalmap.gov/arcgis/rest/',
        'services/3DEPElevation/ImageServer/exportImage?'
    ])

    params = dict(bbox=','.join([str(x) for x in bbox]),
                  bboxSR=inSR,
                  size=f'{width},{height}',
                  imageSR=inSR,
                  time=None,
                  format='tiff',
                  pixelType='F32',
                  noData=None,
                  noDataInterpretation='esriNoDataMatchAny',
                  interpolation='+RSP_BilinearInterpolation',
                  compression=None,
                  compressionQuality=None,
                  bandIds=None,
                  mosaicRule=None,
                  renderingRule=None,
                  f='image')
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    dem = imread(io.BytesIO(r.content))

    return dem


def tpi_from_tnm(bbox,
                 irad,
                 orad,
                 dem_resolution,
                 tpi_resolution=30,
                 parallel=True,
                 norm=True,
                 **kwargs):
    """
    Produces a raster of Topographic Position Index (TPI) by fetching a Digital
    Elevation Model (DEM) from The National Map (TNM) web service.

    TPI is the difference between the elevation at a location from the average
    elevation of its surroundings, calculated using an annulus (ring). This
    function permits the calculation of average surrounding elevation using
    a coarser grain, and return the TPI user a higher-resolution DEM.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    irad : numeric
      inner radius of annulus used to calculate TPI
    orad : numeric
      outer radius of annulus used to calculate TPI
    dem_resolution : numeric
      spatial resolution of Digital Elevation Model (DEM)
    tpi_resolution : numeric
      spatial resolution of DEM used to calculate TPI
    norm : bool
      whether to return a normalized version of TPI, with mean = 0 and SD = 1

    Returns
    -------
    tpi : array
      TPI image as array
    """
    tpi_bbox = np.array(bbox)
    tpi_bbox[0:2] = tpi_bbox[0:2] - orad
    tpi_bbox[2:4] = tpi_bbox[2:4] + orad
    k_orad = orad // tpi_resolution
    k_irad = irad // tpi_resolution

    kernel = disk(k_orad) - np.pad(disk(k_irad), pad_width=(k_orad - k_irad))
    weights = kernel / kernel.sum()

    if dem_resolution != tpi_resolution:
        dem = np.pad(dem_from_tnm(bbox, dem_resolution, **kwargs),
                     orad // dem_resolution)
        tpi_dem = dem_from_tnm(tpi_bbox, tpi_resolution, **kwargs)

    else:
        tpi_dem = dem_from_tnm(tpi_bbox, tpi_resolution, **kwargs)
        dem = tpi_dem

    if parallel:

        def conv(tpi_dem):
            return convolve(tpi_dem, weights)

        convolved = apply_parallel(conv, tpi_dem, compute=True, depth=k_orad)
        if tpi_resolution != dem_resolution:
            tpi = dem - resize(convolved, dem.shape)
        else:
            tpi = dem - convolved

    else:
        if tpi_resolution != dem_resolution:
            tpi = dem - resize(convolve(tpi_dem, weights), dem.shape)
        else:
            tpi = dem - convolve(tpi_dem, weights)

    # trim the padding around the dem used to calculate TPI
    tpi = tpi[orad // dem_resolution:-orad // dem_resolution,
              orad // dem_resolution:-orad // dem_resolution]

    if norm:
        tpi_mean = (tpi_dem - convolved).mean()
        tpi_std = (tpi_dem - convolved).std()
        tpi = (tpi - tpi_mean) / tpi_std

    return tpi


def nlcd_from_mrlc(bbox, res, layer, inSR=4326, nlcd=True, **kwargs):
    """
    Retrieves National Land Cover Data (NLCD) Layers from the Multiresolution
    Land Characteristics Consortium's web service.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    res : numeric
      spatial resolution to use for returned image (grid cell size)
    layer : str
      title of layer to retrieve (e.g., 'NLCD_2001_Land_Cover_L48')
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    nlcd : bool
      if True, will re-map the values returned to the NLCD land cover codes

    Returns
    -------
    img : numpy array
      map image as array
    """
    width = int(abs(bbox[2] - bbox[0]) // res)
    height = int(abs(bbox[3] - bbox[1]) // res)
    BASE_URL = ''.join([
        'https://www.mrlc.gov/geoserver/mrlc_display/wms?',
        'service=WMS&request=GetMap',
    ])

    params = dict(bbox=','.join([str(x) for x in bbox]),
                  crs=f'epsg:{inSR}',
                  width=width,
                  height=height,
                  format='image/tiff',
                  layers=layer)
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    img = imread(io.BytesIO(r.content), format='tiff')

    if nlcd:
        MAPPING = {
            1: 11,  # open water
            2: 12,  # perennial ice/snow
            3: 21,  # developed, open space
            4: 22,  # developed, low intensity
            5: 23,  # developed, medium intensity
            6: 24,  # developed, high intensity
            7: 31,  # barren land (rock/stand/clay)
            8: 32,  # unconsolidated shore
            9: 41,  # deciduous forest
            10: 42,  # evergreen forest
            11: 43,  # mixed forest
            12: 51,  # dwarf scrub (AK only)
            13: 52,  # shrub/scrub
            14: 71,  # grasslands/herbaceous,
            15: 72,  # sedge/herbaceous (AK only)
            16: 73,  # lichens (AK only)
            17: 74,  # moss (AK only)
            18: 81,  # pasture/hay
            19: 82,  # cultivated crops
            20: 90,  # woody wetlands
            21: 95,  # emergent herbaceous wetlands
        }

        k = np.array(list(MAPPING.keys()))
        v = np.array(list(MAPPING.values()))

        mapping_ar = np.zeros(k.max() + 1, dtype=v.dtype)
        mapping_ar[k] = v
        img = mapping_ar[img]

    return img


def ways_from_osm(bbox,
                  crs=None,
                  dissolve=False,
                  polygonize=False,
                  raster_resolution=None,
                  distance_transform=False,
                  **kwargs):
    """Retrieves ways from Open Street Map clipped to a bounding box.

    This utilizes the `OSMnx` package which can execute web queries using the
    Overpass API for Open Street Map.

    Optional keyword arguments are passed to the `graph_from_polygon` function
    in the `osmnx.graph` module.

    One of the most useful keyword arguments for this command is
    `custom_filter`, which can be used to request, for example, waterways by
    specifying `custom_filter = '["waterway"]'`. Similarly, lakes and ponds
    may be returned with `custom_filter = '["natural"="water"]'`. This
    function returns "ways" from Open Street Map as a `NetworkX` graph, so
    filters are limited to different "ways".

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy).
    crs : coordinate reference system
      a string, integer, or class instance which can be interpreted by
      GeoPandas as a Coordinate References System.
    polygonize : bool
      whether to dissolve features by `osmid` and convert into a polygon using
      the convex hull of the features. this may be useful when features
      represent areas rather than lines (e.g., using when using `custom_filter
      = '["waterway"]'`).
    raster_resolution : numeric
      if provided, the results will be returned as a raster with grid cell size
    distance_transform : bool
      if result is rasterized, a value of True will return the distance from
      the nearest feature rather than the binary raster of features.

    Returns
    -------
    clip_gdf : GeoDataFrame
      features in vector format, clipped to bbox
    raster : array
      features rasterized into an integer array with 0s as background values
      and 1s wherever features occured; only returned if `raster_resolution` is
      specified
    """
    if crs:
        geom = box(*bbox)
        bbox_gdf = gpd.GeoDataFrame(geometry=[geom], crs=crs)
        latlon_bbox = bbox_gdf.to_crs(epsg=4326)['geometry'].iloc[0]
    else:
        latlon_bbox = box(*bbox)

    # retrieve the data from OSM as a networkx graph
    g = ox.graph.graph_from_polygon(polygon=latlon_bbox, **kwargs)

    # convert to geodataframe
    gdf = ox.utils_graph.graph_to_gdfs(g, nodes=False)

    if polygonize:
        gdf = gdf.dissolve(by='osmid',
                           aggfunc={
                               'length': 'sum',
                               'name': 'first',
                               'landuse': 'first'
                           })
        gdf['geometry'] = gdf['geometry'].apply(
            lambda x: multi_to_single_linestring(x))
        gdf['geometry'] = gdf['geometry'].convex_hull

    if len(gdf) > 0:
        clip_gdf = gpd.clip(gdf, latlon_bbox)

    if crs:
        clip_gdf = clip_gdf.to_crs(crs)

    if raster_resolution:
        width = int(abs(bbox[2] - bbox[0]) // raster_resolution)
        height = int(abs(bbox[3] - bbox[1]) // raster_resolution)
        gdf_box = [int(x) for x in gdf.unary_union.bounds]
        gdf_width = int(abs(gdf_box[2] - gdf_box[0]) // raster_resolution)
        gdf_height = int(abs(gdf_box[3] - gdf_box[1]) // raster_resolution)

        full_transform = transform.from_bounds(*gdf_box, gdf_width, gdf_height)
        clip_win = windows.from_bounds(*bbox,
                                       transform=full_transform,
                                       width=width,
                                       height=height)
        if len(gdf) > 0:
            full_raster = rasterize(gdf.geometry,
                                    out_shape=(gdf_height, gdf_width),
                                    transform=full_transform,
                                    dtype='uint8')
            clip_ras = full_raster[
                clip_win.round_offsets().round_lengths().toslices()]
        if distance_transform:
            neg = np.logical_not(clip_ras)
            raster = edt(neg)
        else:
            raster = np.zeros((height, width), dtype='uint8')
        return raster

    else:
        return clip_gdf


def water_bodies_from_dnr(layer_num,
                          bbox,
                          inSR=4326,
                          raster_resolution=None,
                          distance_transform=False):
    """
    Returns hydrographic features from the Washington DNR web service.

    Parameters
    ----------
    layer_num : int
      0 will request water courses, 1 will request water bodies
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy).
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    raster_resolution : numeric
      if provided, the results will be returned as a raster with grid cell size
    distance_transform : bool
      if result is rasterized, a value of True will return the distance from
      the nearest feature rather than the binary raster of features.

    Returns
    -------
    clip_gdf : GeoDataFrame
      features in vector format, clipped to bbox
    raster : array
      features rasterized into an integer array with 0s as background values
      and 1s wherever features occured; only returned if `raster_resolution` is
      specified

    """
    BASE_URL = ''.join([
        'https://gis.dnr.wa.gov/site2/rest/services/Public_Water/',
        'WADNR_PUBLIC_Hydrography/MapServer/',
        str(layer_num), '/query?'
    ])

    params = dict(text=None,
                  objectIds=None,
                  time=None,
                  geometry=','.join([str(x) for x in bbox]),
                  geometryType='esriGeometryEnvelope',
                  inSR=inSR,
                  spatialRel='esriSpatialRelEnvelopeIntersects',
                  relationParam=None,
                  outFields='*',
                  returnGeometry='true',
                  returnTrueCurves='false',
                  maxAllowableOffset=None,
                  geometryPrecision=None,
                  outSR=inSR,
                  having=None,
                  returnIdsOnly='false',
                  returnCountOnly='false',
                  orderByFields=None,
                  groupByFieldsForStatistics=None,
                  outStatistics=None,
                  returnZ='false',
                  returnM='false',
                  gdbVersion=None,
                  historicMoment=None,
                  returnDistinctValues='false',
                  resultOffset=None,
                  resultRecordCount=None,
                  queryByDistance=None,
                  returnExtentOnly='false',
                  datumTransformation=None,
                  parameterValues=None,
                  rangeValues=None,
                  quantizationParameters=None,
                  f='geojson')
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    gdf = gpd.GeoDataFrame.from_features(r.json(), crs=inSR)

    if len(gdf) > 0:
        clip_gdf = gpd.clip(gdf, box(*bbox))

    if raster_resolution:
        width = int(abs(bbox[2] - bbox[0]) // raster_resolution)
        height = int(abs(bbox[3] - bbox[1]) // raster_resolution)
        gdf_box = [int(x) for x in gdf.unary_union.bounds]
        gdf_width = int(abs(gdf_box[2] - gdf_box[0]) // raster_resolution)
        gdf_height = int(abs(gdf_box[3] - gdf_box[1]) // raster_resolution)

        full_transform = transform.from_bounds(*gdf_box, gdf_width, gdf_height)
        clip_win = windows.from_bounds(*bbox,
                                       transform=full_transform,
                                       width=width,
                                       height=height)
        if len(gdf) > 0:
            full_raster = rasterize(gdf.geometry,
                                    out_shape=(gdf_height, gdf_width),
                                    transform=full_transform,
                                    dtype='uint8')
            clip_ras = full_raster[
                clip_win.round_offsets().round_lengths().toslices()]
        if distance_transform:
            neg = np.logical_not(clip_ras)
            raster = edt(neg)
        else:
            raster = np.zeros((height, width), dtype='uint8')
        return raster

    else:
        return clip_gdf


def parcels_from_wa(bbox, inSR=4326, **kwargs):
    """
    Returns tax lot boundaries as features from the Washington Geospatial
    Open Data Portal

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy).
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)

    Returns
    -------
    clip_gdf : GeoDataFrame
      features in vector format, clipped to bbox
    """
    BASE_URL = ''.join([
        'https://services.arcgis.com/jsIt88o09Q0r1j8h/arcgis/rest/services/',
        'Current_Parcels_2020/FeatureServer/0/query?'
    ])

    params = dict(
        where=None,
        objectIds=None,
        time=None,
        geometry=','.join([str(x) for x in bbox]),
        geometryType='esriGeometryEnvelope',
        inSR=inSR,
        spatialRel='esriSpatialRelEnvelopeIntersects',
        resultType='none',
        distance=0.0,
        units='esriSRUnit_Meter',
        returnGeodetic='false',
        outFields='*',
        returnGeometry='true',
        returnCentroid='false',
        featureEncoding='esriDefault',
        multipatchOption='xyFootprint',
        maxAllowableOffset=None,
        geometryPrecision=None,
        outSR=inSR,
        datumTransformation=None,
        applyVCSProjection='false',
        returnIdsOnly='false',
        returnUniqueIdsOnly='false',
        returnCountOnly='false',
        returnExtentOnly='false',
        returnQueryGeometry='false',
        returnDistinctValues='false',
        cacheHint='false',
        orderByFields=None,
        groupByFieldsForStatistics=None,
        outStatistics=None,
        having=None,
        resultOffset=None,
        resultRecordCount=None,
        returnZ='false',
        returnM='false',
        returnExceededLimitFeatures='true',
        quantizationParameters=None,
        sqlFormat='none',
        f='pgeojson',
        token=None,
    )
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    jsn = r.json()
    if len(jsn['features']) == 0:
        clip_gdf = gpd.GeoDataFrame(geometry=[Polygon()], crs=inSR)
    else:
        gdf = gpd.GeoDataFrame.from_features(jsn, crs=inSR)
        gdf['geometry'] = gdf.buffer(0)
        clip_gdf = gpd.clip(gdf, box(*bbox))

    return clip_gdf


def parcels_from_or(bbox, inSR=4326, **kwargs):
    """
    Returns tax lot boundaries as features from ORMAP

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy).
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)

    Returns
    -------
    clip_gdf : GeoDataFrame
      features in vector format, clipped to bbox
    """
    BASE_URL = ''.join([
        'https://utility.arcgis.com/usrsvcs/servers/',
        '78bbb0d0d9c64583ad5371729c496dcc/rest/services/',
        'Secure/DOR_ORMAP/MapServer/3/query?',
    ])

    params = dict(f='geojson',
                  returnGeometry='true',
                  spatialRel='esriSpatialRelIntersects',
                  geometry=(f'{{"xmin":{bbox[0]},"ymin":{bbox[1]},'
                            f'"xmax":{bbox[2]},"ymax":{bbox[3]},'
                            f'"spatialReference":{{"wkid":{inSR}}}}}'),
                  geometryType='esriGeometryEnvelope',
                  outFields='*',
                  outSR=inSR)
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    jsn = r.json()
    if len(jsn['features']) == 0:
        clip_gdf = gpd.GeoDataFrame(geometry=[Polygon()], crs=inSR)
    else:
        gdf = gpd.GeoDataFrame.from_features(jsn, crs=inSR)
        gdf['geometry'] = gdf.buffer(0)
        clip_gdf = gpd.clip(gdf, box(*bbox))

    return clip_gdf


def nhd_from_tnm(nhd_layer,
                 bbox,
                 inSR=4326,
                 raster_resolution=None,
                 distance_transform=False,
                 **kwargs):
    """Returns features from the National Hydrography Dataset Plus High
    Resolution web service from The National Map.

    Available layers are:

    =========  ======================
    NHD Layer  Description
    =========  ======================
    0          NHDPlusSink
    1          NHDPoint
    2          NetworkNHDFlowline
    3          NonNetworkNHDFlowline
    4          FlowDirection
    5          NHDPlusWall
    6          NHDPlusBurnLineEvent
    7          NHDLine
    8          NHDArea
    9          NHDWaterbody
    10         NHDPlusCatchment
    11         WBDHU12
    =========  ======================

    Parameters
    ----------
    nhd_layer : int
       a value from 0-11 indicating the feature layer to retrieve.
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy).
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    raster_resolution : numeric
      causes features to be returned in raster (rather than vector) format,
      with spatial resolution defined by this parameter.
    distance_transform : bool
      if result is rasterized, a value of True will return the distance from
      the nearest feature rather than the binary raster of features.

    Returns
    -------
    clip_gdf : GeoDataFrame
      features in vector format, clipped to bbox
    raster : array
      features rasterized into an integer array with 0s as background values
      and 1s wherever features occured; only returned if `raster_resolution` is
      specified
    """
    BASE_URL = ''.join([
        'https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/',
        'MapServer/',
        str(nhd_layer), '/query?'
    ])

    params = dict(where=None,
                  text=None,
                  objectIds=None,
                  time=None,
                  geometry=','.join([str(x) for x in bbox]),
                  geometryType='esriGeometryEnvelope',
                  inSR=inSR,
                  spatialRel='esriSpatialRelIntersects',
                  relationParam=None,
                  outFields=None,
                  returnGeometry='true',
                  returnTrueCurves='false',
                  maxAllowableOffset=None,
                  geometryPrecision=None,
                  outSR=inSR,
                  having=None,
                  returnIdsOnly='false',
                  returnCountOnly='false',
                  orderByFields=None,
                  groupByFieldsForStatistics=None,
                  outStatistics=None,
                  returnZ='false',
                  returnM='false',
                  gdbVersion=None,
                  historicMoment=None,
                  returnDistinctValues='false',
                  resultOffset=None,
                  resultRecordCount=None,
                  queryByDistance=None,
                  returnExtentOnly='false',
                  datumTransformation=None,
                  parameterValues=None,
                  rangeValues=None,
                  quantizationParameters=None,
                  featureEncoding='esriDefault',
                  f='geojson')
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    try:
        gdf = gpd.GeoDataFrame.from_features(r.json(), crs=inSR)

    # this API seems to return M and Z values even if not requested
    # this catches the error and keeps only the first two coordinates (x and y)
    except AssertionError:
        js = r.json()
        for f in js['features']:
            f['geometry'].update({
                'coordinates': [c[0:2] for c in f['geometry']['coordinates']]
            })
        gdf = gdf = gpd.GeoDataFrame.from_features(js)

    if len(gdf) > 0:
        clip_gdf = gpd.clip(gdf, box(*bbox))

    if raster_resolution:
        width = int(abs(bbox[2] - bbox[0]) // raster_resolution)
        height = int(abs(bbox[3] - bbox[1]) // raster_resolution)
        gdf_box = [int(x) for x in gdf.unary_union.bounds]
        gdf_width = int(abs(gdf_box[2] - gdf_box[0]) // raster_resolution)
        gdf_height = int(abs(gdf_box[3] - gdf_box[1]) // raster_resolution)

        full_transform = transform.from_bounds(*gdf_box, gdf_width, gdf_height)
        clip_win = windows.from_bounds(*bbox,
                                       transform=full_transform,
                                       width=width,
                                       height=height)
        if len(gdf) > 0:
            full_raster = rasterize(gdf.boundary.geometry,
                                    out_shape=(gdf_height, gdf_width),
                                    transform=full_transform,
                                    dtype='uint8')
            clip_ras = full_raster[
                clip_win.round_offsets().round_lengths().toslices()]
        if distance_transform:
            neg = np.logical_not(clip_ras)
            raster = edt(neg)
        else:
            raster = np.zeros((height, width), dtype='uint8')
        return raster

    else:
        return clip_gdf


def watersheds_from_tnm(huc_level,
                        bbox,
                        inSR,
                        raster_resolution=None,
                        distance_transform=False):
    """Returns features for watershed boundaries at the geographic extent
    specified by the user from The National Map web service.

    Available Hydrologic Unit Codes (`huc_level`) are translated to the
    following feature services:

    =========  ============  ==========
    HUC Level  Description   Feature ID
    =========  ============  ==========
    2          Region            1
    4          Subregion         2
    6          Basin             3
    8          Subbasin          4
    10         Watershed         5
    12         Subwatershed      6
    14         --                7
    =========  ============  ==========

    Parameters
    ----------
    huc_level : int
       the number of digits for the Hydrologic Unit Code, higher numbers
       correspond to smaller regional extents (more detailed delineation of
       watersheds). Must be one of {2, 4, 6, 8, 10, 12, 14}
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    raster_resolution : numeric
      causes features to be returned in raster (rather than vector) format,
      with spatial resolution defined by this parameter.
    distance_transform : bool
      if result is rasterized, a value of True will return the distance from
      the nearest feature rather than the binary raster of features.

    Returns
    -------
    clip_gdf : GeoDataFrame
      features in vector format, clipped to bbox
    raster : array
      features rasterized into an integer array with 0s as background values
      and 1s wherever features occured; only returned if `raster_resolution` is
      specified
    """
    feature_ids = {2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6, 14: 7}
    keys = feature_ids.keys()
    if huc_level not in keys:
        raise ValueError(f'huc_level not recognized, must be one of {keys}')

    feat_id = feature_ids[huc_level]
    BASE_URL = ''.join([
        'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/',
        'MapServer/',
        str(feat_id), '/query?'
    ])

    params = dict(where=None,
                  text=None,
                  objectIds=None,
                  time=None,
                  geometry=','.join([str(x) for x in bbox]),
                  geometryType='esriGeometryEnvelope',
                  inSR=inSR,
                  spatialRel='esriSpatialRelIntersects',
                  relationParam=None,
                  outFields=None,
                  returnGeometry='true',
                  returnTrueCurves='false',
                  maxAllowableOffset=None,
                  geometryPrecision=None,
                  outSR=inSR,
                  having=None,
                  returnIdsOnly='false',
                  returnCountOnly='false',
                  orderByFields=None,
                  groupByFieldsForStatistics=None,
                  outStatistics=None,
                  returnZ='false',
                  returnM='false',
                  gdbVersion=None,
                  historicMoment=None,
                  returnDistinctValues='false',
                  resultOffset=None,
                  resultRecordCount=None,
                  queryByDistance=None,
                  returnExtentOnly='false',
                  datumTransformation=None,
                  parameterValues=None,
                  rangeValues=None,
                  quantizationParameters=None,
                  featureEncoding='esriDefault',
                  f='geojson')

    r = requests.get(BASE_URL, params=params)
    gdf = gpd.GeoDataFrame.from_features(r.json(), crs=inSR)

    if len(gdf) > 0:
        clip_gdf = gpd.clip(gdf, box(*bbox))

    if raster_resolution:
        width = int(abs(bbox[2] - bbox[0]) // raster_resolution)
        height = int(abs(bbox[3] - bbox[1]) // raster_resolution)
        gdf_box = [int(x) for x in gdf.unary_union.bounds]
        gdf_width = int(abs(gdf_box[2] - gdf_box[0]) // raster_resolution)
        gdf_height = int(abs(gdf_box[3] - gdf_box[1]) // raster_resolution)

        full_transform = transform.from_bounds(*gdf_box, gdf_width, gdf_height)
        clip_win = windows.from_bounds(*bbox,
                                       transform=full_transform,
                                       width=width,
                                       height=height)
        if len(gdf) > 0:
            full_raster = rasterize(gdf.boundary.geometry,
                                    out_shape=(gdf_height, gdf_width),
                                    transform=full_transform,
                                    dtype='uint8')
            clip_ras = full_raster[
                clip_win.round_offsets().round_lengths().toslices()]
        if distance_transform:
            neg = np.logical_not(clip_ras)
            raster = edt(neg)
        else:
            raster = np.zeros((height, width), dtype='uint8')
        return raster

    else:
        return clip_gdf


def buildings_from_microsoft(bbox, inSR, raster_resolution=None):
    """Returns building footprints generated by Microsoft and hosted by an
    ArcGIS Feature Server.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy).
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    raster_resolution : numeric
      causes features to be returned in raster (rather than vector) format,
      with spatial resolution defined by this parameter.
    """
    BASE_URL = ''.join([
        'https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/',
        'MSBFP2/FeatureServer/0/query?'
    ])

    params = dict(where=None,
                  objectIds=None,
                  time=None,
                  geometry=','.join([str(x) for x in bbox]),
                  geometryType='esriGeometryEnvelope',
                  inSR=inSR,
                  spatialRel='esriSpatialRelIntersects',
                  resultType='none',
                  distance=0.0,
                  units='esriSRUnit_Meter',
                  returnGeodetic='false',
                  outFields=None,
                  returnGeometry='true',
                  returnCentroid='false',
                  featureEncoding='esriDefault',
                  multipatchOption='xyFootprint',
                  maxAllowableOffset=None,
                  geometryPrecision=None,
                  outSR=inSR,
                  datumTransformation=None,
                  applyVCSProjection='false',
                  returnIdsOnly='false',
                  returnUniqueIdsOnly='false',
                  returnCountOnly='false',
                  returnExtentOnly='false',
                  returnQueryGeometry='false',
                  returnDistinctValues='false',
                  cacheHint='false',
                  orderByFields=None,
                  groupByFieldsForStatistics=None,
                  outStatistics=None,
                  having=None,
                  resultOffset=None,
                  resultRecordCount=None,
                  returnZ='false',
                  returnM='false',
                  returnExceededLimitFeatures='true',
                  quantizationParameters=None,
                  sqlFormat='none',
                  f='pgeojson',
                  token=None)

    r = requests.get(BASE_URL, params=params)
    gdf = gpd.GeoDataFrame.from_features(r.json(), crs=inSR)

    if len(gdf) > 0:
        gdf = gpd.clip(gdf, box(*bbox))

    if raster_resolution:
        width = int(abs(bbox[2] - bbox[0]) // raster_resolution)
        height = int(abs(bbox[3] - bbox[1]) // raster_resolution)
        trf = transform.from_bounds(*bbox, width, height)
        if len(gdf) > 0:
            raster = rasterize(gdf.geometry,
                               out_shape=(height, width),
                               transform=trf,
                               dtype='uint8')
        else:
            raster = np.zeros((height, width), dtype='uint8')
        return raster

    else:
        return gdf


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
    nw_bbox = [xmin, (ymin + ymax) / 2, (xmin + xmax) / 2, ymax]
    ne_bbox = [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax, ymax]
    sw_bbox = [xmin, ymin, (xmin + xmax) / 2, (ymin + ymax) / 2]
    se_bbox = [(xmin + xmax) / 2, ymin, xmax, (ymin + ymax) / 2]

    bboxes = [nw_bbox, ne_bbox, sw_bbox, se_bbox]

    get_quads = partial(fetcher, *args, **kwargs)

    with ThreadPool(num_threads) as p:
        quads = p.map(get_quads, bboxes)
        nw, ne, sw, se = quads

    quad_img = np.vstack([np.hstack([nw, ne]), np.hstack([sw, se])])

    return quad_img


def metadata_from_noaa_digital_coast(bbox, inSR=4326, **kwargs):
    """Returns metadata about lidar acquisitions from NOAA Digital Coast.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy).
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)

    Returns
    -------
    gdf : GeoDataFrame
      features in vector format
    """
    BASE_URL = ''.join([
        'https://maps.coast.noaa.gov/arcgis/rest/services/DAV/',
        'ElevationFootprints/MapServer/0/query?'
    ])

    params = dict(where=None,
                  text=None,
                  objectIds=None,
                  time=None,
                  geometry=','.join([str(x) for x in bbox]),
                  geometryType='esriGeometryEnvelope',
                  inSR=inSR,
                  spatialRel='esriSpatialRelIntersects',
                  relationParam=None,
                  outFields='*',
                  returnGeometry='true',
                  returnTrueCurves='false',
                  maxAllowableOffset=None,
                  geometryPrecision=None,
                  outSR=inSR,
                  returnIdsOnly='false',
                  returnCountOnly='false',
                  orderByFields=None,
                  groupByFieldsForStatistics=None,
                  outStatistics=None,
                  returnZ='false',
                  returnM='false',
                  gdbVersion=None,
                  returnDistinctValues='false',
                  resultOffset=None,
                  resultRecordCount=None,
                  f='geojson')

    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    if params['outSR'] != inSR:
        crs = params['outSR']
    else:
        crs = params['inSR']

    gdf = gpd.GeoDataFrame.from_features(r.json(), crs=crs)
    for col in gdf.columns:
        if gdf[col].dtype == 'O':
            gdf[col] = gdf[col].str.strip(',')

    return gdf


def scrape_hyperlinks(url):
    """Parses a web page and returns a list of all hyperlinks on it.

    Parameters
    ----------
    url : str
      URL of web page to scrape links from

    Returns
    -------
    links : list
      list of hyperlinks found

    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    links = [x.get('href') for x in soup.findAll('a')]
    return links


def tileindex_from_noaa_digital_coast(url, crs=None):
    """Retrieves a tile index from a NOAA Digital Coast web page and returns
    the result as a GeoDataFrame.

    Parameters
    ----------
    url : str
      URL of web page for a dataset hosted by NOAA Digital Coast.

    Returns
    -------
    gdf : GeoDataFrame
      tile index for the dataset
    """
    links = scrape_hyperlinks(url)
    zip_url = [url + l for l in links if 'tileindex' in l]

    if len(zip_url) > 1:
        warnings.warn(f'''More than one tileindex found at {url}.
                      Returning only the first: {zip_url[0]}.''')
    if len(zip_url) == 0:
        raise IndexError("No files found with 'tileindex' in name.")

    gdf = gpd.read_file(zip_url[0])

    for col in gdf.columns:
        if gdf[col].dtype == 'O':
            gdf[col] = gdf[col].str.strip()

    if crs:
        return gdf.to_crs(crs)
    else:
        return gdf


def intersecting_tiles_noaa_digital_coast(aoi, crs=4326, **kwargs):
    """Returns a list of tiles that intersect a user-defined Area of Interest
    (AOI) from datasets hosted on NOAA Digital Coast.

    Parameters
    ----------
    aoi : geometry
      Shapely geometry (e.g., Polygon) depicting the user's AOI
    crs : str, int, or CRS
      anything that can be interpreted by GeoPandas as a Coordinate Reference
      System
    kwargs
      additional keyword arguments that will be passed to the request to NOAA
      Digital Coast retrieving metadata on available datasets

    Returns
    -------
    tile_dict : dictionary
      dictionary where each key is the project ID used by NOAA with two entries
      per project: 'metadata' containing a dictionary with all metadata
      retrieved, and 'tile_gdf' containing a GeoDataFrame with information
      about any tiles that intersect the user's AOI
    """
    if type(crs) == CRS:
        crs = crs.to_epsg()

    bbox = aoi.bounds
    metadata = metadata_from_noaa_digital_coast(bbox, inSR=crs, **kwargs)

    tile_dict = {}
    for idx, row in metadata.iterrows():
        url = metadata.loc[idx]['ExternalProviderLink']
        tileindex = tileindex_from_noaa_digital_coast(url, crs=crs)
        aoi_tiles = tileindex.loc[tileindex.geometry.intersects(aoi)]

        if len(aoi_tiles) > 0:
            tile_dict[metadata.loc[idx]['ID']] = {
                'metadata': metadata.loc[idx].to_dict(),
                'tile_gdf': aoi_tiles
            }
    return tile_dict


def contour_images_from_tnm(bbox, img_width, img_height, inSR=3857,
                            index_contour_style=None,
                            intermediate_contour_style=None,
                            contour_label_style=None):
    """
    Retrieves an image of contour lines and labels from The National Map.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    img_width : int
      width of image to return, in pixels
    img_height : int
      height of image to return, in pixels
    inSR : int, optional
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    index_contour_style : dict, optional
      dict with key, value pairs indicating elements of the line style to
      update for the 100-foot contour lines
    intermediate_contour_style : dict, optional
      dict with key, value pairs indicating elements of the line style to
      update for intermediate (50-foot) contour lines
    contour_label_style : dict, option
      dict with key, value pairs indicating elements of label style to update
      for the 100-foot contour labels

    Returns
    -------
    img : array
      image as a 3-band or 4-band array
    """
    BASE_URL = ''.join([
        'https://carto.nationalmap.gov/arcgis/rest/services/',
        'contours/MapServer/export?'
    ])

    index_contour_symbol = {
      "type": "esriSLS",
      "style": "esriSLSSolid",
      "color": [32,96,0,255],
      "width": 1.5
      }
    if index_contour_style is not None:
        index_contour_symbol.update(index_contour_style)

    intermediate_contour_symbol = {
      "type": "esriSLS",
      "style": "esriSLSSolid",
      "color": [32,96,0,255],
      "width": 0.5
      }
    if intermediate_contour_style is not None:
        intermediate_contour_symbol.update(intermediate_contour_style)

    label_symbol = {
      "type":"esriTS",
      "color":[15,39,3,255],
      "backgroundColor":None,
      "outlineColor":None,
      "verticalAlignment":"baseline",
      "horizontalAlignment":"left",
      "rightToLeft":False,
      "angle":0,
      "xoffset":0,
      "yoffset":0,
      "kerning":True,
      "haloSize":2,
      "haloColor":[255,255,255,255],
      "font":{
          "family":"Arial",
          "size":12,
          "style":"italic",
          "weight":"normal",
          "decoration":"none"
          }
      }
    if contour_label_style is not None:
        for key in contour_label_style:
            if isinstance(contour_label_style[key], dict):
                label_symbol[key].update(contour_label_style[key])
            else:
                label_symbol.update(((key, contour_label_style[key]),))

    styles = [
    {"id":25,
     "source":{"type":"mapLayer", "mapLayerId":25},
     "drawingInfo":{
         "renderer":{
             "type":"simple",
              "symbol":index_contour_symbol,
              },
         },
     },
    {"id":26,
     "source":{"type":"mapLayer", "mapLayerId":26},
     "drawingInfo":{
         "renderer":{
             "type":"simple",
             "symbol":intermediate_contour_symbol,
             },
         },
     },
     {"id":21,
      "source":{"type":"mapLayer", "mapLayerId":21},
      "drawingInfo":{
          "renderer":{
              "type":"uniqueValue",
              "field1":"FCODE",
               "fieldDelimiter":",",
               },
          "labelingInfo":[
              {
               "labelPlacement":"esriServerLinePlacementCenterAlong",
               "labelExpression":"[CONTOURELEVATION]",
               "useCodedValues":True,
               "symbol":label_symbol,
               "minScale":0,
               "maxScale":0
               }
              ]
          }
      }
     ]

    params = dict(
        bbox=','.join([str(x) for x in bbox]),
        bboxSR=inSR,
        layers='show:21,25,26',
        layerDefs=None,
        size=f'{img_width},{img_height}',
        imageSR=inSR,
        historicMoment=None,
        format='png',
        transparent=True,
        dpi=None,
        time=None,
        layerTimeOptions=None,
        dynamicLayers=json.dumps(styles),
        gdbVersion=None,
        mapScale=None,
        rotation=None,
        datumTransformations=None,
        layerParameterValues=None,
        mapRangeValues=None,
        layerRangeValues=None,
        f='image',
    )

    r = requests.get(BASE_URL, params=params)
    img = imread(io.BytesIO(r.content))

    return img

def elevation_point_query_tnm(lon, lat, units='Feet', format='json'):
    BASE_URL = 'https://nationalmap.gov/epqs/pqs.php?'
    data = {'x': lon, 'y': lat, 'units':units, 'output':format}
    r = request.post(BASE_URL, data=data)
    elev = r.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']
    return elev

def contours_from_tnm_dem(bbox, width, height, inSR=3857):
    from matplotlib import ticker
    from matplotlib import patheffects as pe

    dem = dem_from_tnm(bbox, width, height, inSR=inSR) * 3.28084
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1.697, 2.407)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_aspect('equal')
    fig.add_axes(ax)
    COLOR = (32/255.,96/255.,0.,255/255.)

    min_40, max_40 = np.floor(dem.min()/40)*40, np.ceil(dem.max()/40)*40+40
    min_200, max_200 = np.floor(dem.min()/200)*200, np.ceil(dem.max()/200)*200+200

    cont_40 =  ax.contour(dem, levels=np.arange(min_40,max_40,40),
                          colors=[COLOR],
                          linewidths=[0.15])
    cont_200 = ax.contour(dem, levels=np.arange(min_200,max_200,200),
                          colors=[COLOR],
                          linewidths=[0.5])
    fmt = ticker.StrMethodFormatter("{x:,.0f} ft")
    labels = ax.clabel(cont_200, fontsize=3,
                       colors=[COLOR], fmt=fmt,
                       inline_spacing=0)
    for label in labels:
        label.set_path_effects([pe.withStroke(linewidth=1, foreground='w')])

    return plt_to_pil_image(fig, dpi=300)


def plt_to_pil_image(plt_figure, dpi=200, transparent=False):
    """
    Converts a matplotlib figure to a PIL Image (in memory).
    Parameters
    ---------
    plt_figure : matplotlib Figure object
      the figure to convert
    dpi : int
      the number of dots per inch to render the image
    transparent : bool, optional
      render plt with a transparent background
    Returns
    -------
    pil_image : Image
      the figure converted to a PIL Image
    """
    fig = plt.figure(plt_figure.number)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, transparent=transparent)
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close()

    return pil_image
