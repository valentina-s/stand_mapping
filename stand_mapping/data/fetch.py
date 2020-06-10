"""
Functions that retrieve images and features from public web services.
"""

import requests
import io
import base64
from imageio import imread
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box, LineString, MultiLineString
from skimage.morphology import disk
from skimage.util import apply_parallel
from skimage.transform import resize
from scipy.ndimage.filters import convolve
from rasterio.transform import from_bounds
from rasterio.features import rasterize

def landcover_from_ai4earth(bbox,
                            epsg,
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
    epsg : int
      code for the Coordinate Reference System defining bbox coordinates
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
                "latestWkid": epsg
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
    elif prediciton_type == 'both':
        return hard_cover, soft_cover
    else:
        raise ValueError(
            "prediction_type must be one of 'hard', 'soft', or 'both'.")


def naip_from_ai4earth(bbox, epsg, api_key):
    """
    Retrieve 3-band NAIP image from the Microsoft AI for Earth, v2 API within
    a bounding box.

    Parameters
    ----------
    bbox : 4-tuple or list
      coordinates of x_min, y_min, x_max, and y_max for bounding box of tile
    epsg : int
      code for the Coordinate Reference System defining bbox coordinates
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
                "latestWkid": epsg
            },
        }
    }
    print('Retrieving image from AI for Earth Land Cover API', end='... ')
    r = requests.post(BASE_URL, json=extent, headers=api_header).json()
    naip = imread(io.BytesIO(base64.b64decode(r['input_naip'])))
    print('Done.')

    return naip


def naip_from_tnm(bbox, res, bboxSR=4326, imageSR=4326, **kwargs):
    """
    Retrieves a NAIP image from The National Map (TNM) web service.

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
    BASE_URL = ''.join(
        ['https://services.nationalmap.gov/arcgis/rest/services/',
         'USGSNAIPImagery/ImageServer/exportImage?']
        )

    width = int(abs(bbox[2] - bbox[0]) // res)
    height = int(abs(bbox[3] - bbox[1]) // res)

    params = {
      'bbox': ','.join([str(x) for x in bbox]),
      'bboxSR': bboxSR,
      'size': f'{width},{height}',
      'imageSR': imageSR,
      'format': 'tiff',
      'pixelType': 'U8',
      'noDataInterpretation': 'esriNoDataMatchAny',
      'interpolation': '+RSP_BilinearInterpolation',
      'time': None,
      'noData': None,
      'compression': None,
      'compressionQuality': None,
      'bandIds': None,
      'mosaicRule': None,
      'renderingRule': None,
      'f': 'image',
    }
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)

    if params['f'] == 'image':
        img = imread(io.BytesIO(r.content))
        return img
    else:
        return r


def dem_from_tnm(bbox, res, bboxSR=4326, imageSR=4326, **kwargs):
    """
    Retrieves a Digital Elevation Model (DEM) image from The National Map (TNM)
    web service.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    res : numeric
      spatial resolution to use for returned DEM (grid cell size)
    bboxSR : integer
      spatial reference code, e.g., EPSG code, for interpreting bbox
      coordinates
    imageSR : integer
      spatial reference code, e.g., EPSG code, for returned DEM

    Returns
    -------
    dem : numpy array
      DEM image as array
    """
    width = int(abs(bbox[2] - bbox[0]) // res)
    height = int(abs(bbox[3] - bbox[1]) // res)

    BASE_URL = ''.join(['https://elevation.nationalmap.gov/arcgis/rest/',
                        'services/3DEPElevation/ImageServer/exportImage?'])
    params = dict(
        bbox=','.join([str(x) for x in bbox]),
        bboxSR=bboxSR,
        size=f'{width},{height}',
        imageSR=imageSR,
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
        f='image'
    )
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    dem = imread(io.BytesIO(r.content))

    return dem


def tpi_from_tnm(bbox, irad, orad, dem_resolution, tpi_resolution=30,
                 parallel=True, norm=True,
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
    tpi_bbox = np.asanyarray(bbox)
    tpi_bbox[0:2] = tpi_bbox[0:2] - orad
    tpi_bbox[2:4] = tpi_bbox[2:4] + orad
    k_orad = orad // tpi_resolution
    k_irad = irad // tpi_resolution

    kernel = disk(k_orad) - np.pad(disk(k_irad), pad_width=(k_orad-k_irad))
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
    tpi = tpi[orad//dem_resolution:-orad//dem_resolution,
              orad//dem_resolution:-orad//dem_resolution]

    if norm:
        tpi_mean = (tpi_dem - convolved).mean()
        tpi_std = (tpi_dem - convolved).std()
        tpi = (tpi - tpi_mean)/ tpi_std

    return tpi


def ways_from_osm(bbox, crs=None, clip_to_bbox=True, dissolve=False,
                  polygonize=False, raster_resolution=None, **kwargs):
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
    clip_to_bbox : bool
      whether to clip returned features to bounding box. ways with nodes that
      extend beyond the bounding box may be returned if set to False.
    polygonize : bool
      whether to dissolve features by `osmid` and convert into a polygon using
      the convex hull of the features. this may be useful when features
      represent areas rather than lines (e.g., using when using `custom_filter
      = '["waterway"]'`).
    raster_resolution : numeric
      if provided, the results will be returned as a raster with grid cell size

    Returns
    -------
    gdf : GeoDataFrame
      GeoPandas GeoDataFrame containing all ways as vector features
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
        gdf = gdf.dissolve(by='osmid', aggfunc=
                           {'length': 'sum',
                            'name':'first',
                            'landuse': 'first'
                            })
        gdf['geometry'] = gdf['geometry'].apply(
            lambda x: multi_to_single_linestring(x))
        gdf['geometry'] = gdf['geometry'].convex_hull

    if clip_to_bbox:
        gdf = gpd.clip(gdf, latlon_bbox)

    if crs and not raster_resolution:  # reproject if user provided CRS
        return gdf.to_crs(crs)

    if crs and raster_resolution:
        width = int(abs(bbox[2] - bbox[0]) // raster_resolution)
        height = int(abs(bbox[3] - bbox[1]) // raster_resolution)
        transform = from_bounds(*bbox, width, height)
        raster = rasterize(gdf.to_crs(crs).geometry,
                           out_shape=(height, width),
                           transform=transform)
        return raster

    if not crs and not raster_resolution:
        return gdf


def water_bodies_from_dnr(layer_num, bbox, bbox_epsg=None,
                          raster_resolution=None):
    """
    Returns hydrographic features from the Washington DNR web service.

    Parameters
    ----------
    layer_num : int
      0 will request water courses, 1 will request water bodies
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy).
    bbox_epsg : int
      EPSG code for the Coordinate Reference System defining bbox coordinates
    raster_resolution : numeric
      if provided, the results will be returned as a raster with grid cell size

    Returns
    -------
    gdf : GeoDataFrame
      features in vector format
    raster : array
      features rasterized into an integer array with 0s as background values
      and 1s wherever features occured; only returned if `raster_resolution` is
      specified

    """
    BASEURL = ''.join(
        ['https://gis.dnr.wa.gov/site2/rest/services/Public_Water/'
         'WADNR_PUBLIC_Hydrography/MapServer/',
         str(layer_num),
         '/query?']
        )
    params = {
        'text': None,
        'objectIds': None,
        'time': None,
        'geometry': ','.join([str(x) for x in bbox]),
        'geometryType': 'esriGeometryEnvelope',
        'inSR': bbox_epsg,
        'spatialRel': 'esriSpatialRelEnvelopeIntersects',
        'relationParam': None,
        'outFields':'*',
        'returnGeometry': 'true',
        'returnTrueCurves': 'false',
        'maxAllowableOffset': None,
        'geometryPrecision': None,
        'outSR': bbox_epsg,
        'having': None,
        'returnIdsOnly': 'false',
        'returnCountOnly': 'false',
        'orderByFields': None,
        'groupByFieldsForStatistics': None,
        'outStatistics': None,
        'returnZ': 'false',
        'returnM': 'false',
        'gdbVersion': None,
        'historicMoment': None,
        'returnDistinctValues': 'false',
        'resultOffset': None,
        'resultRecordCount': None,
        'queryByDistance': None,
        'returnExtentOnly': 'false',
        'datumTransformation': None,
        'parameterValues': None,
        'rangeValues': None,
        'quantizationParameters': None,
        'f':'geojson'
    }

    r = requests.get(BASEURL, params=params)
    gdf = gpd.GeoDataFrame.from_features(r.json(), crs=bbox_epsg)
    gdf = gpd.clip(gdf, box(*bbox))

    if raster_resolution:
        width = int(abs(bbox[2] - bbox[0]) // raster_resolution)
        height = int(abs(bbox[3] - bbox[1]) // raster_resolution)
        transform = from_bounds(*bbox, width, height)
        raster = rasterize(gdf.geometry,
                           out_shape=(height, width),
                           transform=transform)
        return raster

    else:
        return gdf


def nhd_from_tnm(nhd_layer, bbox, bbox_epsg, raster_resolution=None):
    """Returns features from the National Hydrography Dataset Plus High
    Resolution web service from The National Map.

    Available layers are:
    0: NHDPlusSink
    1: NHDPoint
    2: NetworkNHDFlowline
    3: NonNetworkNHDFlowline
    4: FlowDirection
    5: NHDPlusWall
    6: NHDPlusBurnLineEvent
    7: NHDLine
    8: NHDArea
    9: NHDWaterbody
    10: NHDPlusCatchment
    11: WBDHU12

    Parameters
    ----------
    nhd_layer : int
       a value from 0-11 indicating the feature layer to retrieve.
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy).
    bbox_epsg : integer
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    raster_resolution : numeric
      causes features to be returned in raster (rather than vector) format,
      with spatial resolution defined by this parameter.
    """
    BASEURL = ''.join(
        ['https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/',
         'MapServer/', str(nhd_layer), '/query?'])

    params = dict(
        where=None,
        text=None,
        objectIds=None,
        time=None,
        geometry=','.join([str(x) for x in bbox]),
        geometryType='esriGeometryEnvelope',
        inSR=bbox_epsg,
        spatialRel='esriSpatialRelIntersects',
        relationParam=None,
        outFields=None,
        returnGeometry='true',
        returnTrueCurves='false',
        maxAllowableOffset=None,
        geometryPrecision=None,
        outSR=bbox_epsg,
        having=None,
        returnIdsOnly='false',
        returnCountOnly='false',
        orderByFields=None,
        groupByFieldsForStatistics=None,
        outStatistics=None,
        returnZ='true',
        returnM='true',
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
        f='geojson'
        )

    r = requests.get(BASEURL, params=params)
    try:
        gdf = gpd.GeoDataFrame.from_features(r.json(), crs=bbox_epsg)
    except AssertionError:
        js = r.json()
        for f in js['features']:
            f['geometry'].update(
                {'coordinates':[c[0:2] for c in f['geometry']['coordinates']]})
        gdf = gdf = gpd.GeoDataFrame.from_features(js)

    if len(gdf) > 0:
        gdf = gpd.clip(gdf, box(*bbox))

    if raster_resolution:
        width = int(abs(bbox[2] - bbox[0]) // raster_resolution)
        height = int(abs(bbox[3] - bbox[1]) // raster_resolution)
        transform = from_bounds(*bbox, width, height)
        if len(gdf) > 0:
            raster = rasterize(gdf.geometry,
                               out_shape=(height, width),
                               transform=transform)
        else:
            raster = np.zeros((height, width), dtype=int)
        return raster

    else:
        return gdf


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


from multiprocessing.pool import ThreadPool
from functools import partial

def quad_naip_from_tnm(bbox, res, bboxSR=4326, imageSR=4326, **kwargs):
    xmin, ymin, xmax, ymax = bbox
    nw_bbox = [xmin, (ymin + ymax) / 2, (xmin + xmax)/2, ymax]
    ne_bbox = [(xmin + xmax)/2, (ymin + ymax)/2, xmax, ymax]
    sw_bbox = [xmin, ymin, (xmin + xmax)/2, (ymin + ymax)/2]
    se_bbox = [(xmin + xmax)/2, ymin, xmax, (ymin + ymax)/2]

    bboxes = [nw_bbox, ne_bbox, sw_bbox, se_bbox]

    get_naip = partial(naip_from_tnm, res=res, bboxSR=bboxSR, imageSR=imageSR, **kwargs)
    with ThreadPool(4) as p:
        naip_images = p.map(get_naip, bboxes)
    nw, ne, sw, se = naip_images

    return np.vstack([np.hstack([nw, ne]), np.hstack([sw, se])])

def quad_dem_from_tnm(bbox, res, bboxSR=4326, imageSR=4326, **kwargs):
    xmin, ymin, xmax, ymax = bbox
    nw_bbox = [xmin, (ymin + ymax) / 2, (xmin + xmax)/2, ymax]
    ne_bbox = [(xmin + xmax)/2, (ymin + ymax)/2, xmax, ymax]
    sw_bbox = [xmin, ymin, (xmin + xmax)/2, (ymin + ymax)/2]
    se_bbox = [(xmin + xmax)/2, ymin, xmax, (ymin + ymax)/2]

    bboxes = [nw_bbox, ne_bbox, sw_bbox, se_bbox]

    get_dem = partial(dem_from_tnm, res=res, bboxSR=bboxSR, imageSR=imageSR, **kwargs)
    with ThreadPool(4) as p:
        dems = p.map(get_dem, bboxes)
    nw, ne, sw, se = dems

    return np.vstack([np.hstack([nw, ne]), np.hstack([sw, se])])
