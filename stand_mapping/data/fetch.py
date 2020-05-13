"""
Functions that retrieve images from publicly-available web services.
"""

import requests
import io
import base64
from imageio import imread


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


def naip_from_tnm_url(xmin,
                      ymin,
                      xmax,
                      ymax,
                      bbox_epsg=3857,
                      img_epsg=3857,
                      img_format='jpgpng',
                      width=500,
                      height=500):
    """
  Returns URL for requesting an image from The National Map NAIP REST service.

  Parameters
  ----------
  minx, miny, maxx, maxy : numeric
    the coordinates of the bounding box of the image to return, assumed to be
    in coordinate reference system EPSG:3857
  img_format : string
    image format to request, valid options are 'jpg', 'jpgpng', 'png', 'png8',
    'png24', 'png32', 'bmp', 'gif', 'tiff'.


  Returns
  -------
  url : string
    the url that will return the NAIP image
  """

    url = ''.join([
        'https://services.nationalmap.gov/arcgis/rest/services/',
        'USGSNAIPImagery/ImageServer/exportImage?',
        'bbox={}%2C{}%2C{}%2C{}&'.format(
            xmin, ymin, xmax,
            ymax), 'bboxSR={}&'.format(bbox_epsg), 'size={}%2C{}&'.format(
                width, height), 'imageSR={}&'.format(img_epsg), 'time=&',
        'format={}&'.format(img_format), 'pixelType=U8&', 'noData=&',
        'noDataInterpretation=esriNoDataMatchAny&',
        'interpolation=+RSP_BilinearInterpolation&'
        'compression=&', 'compressionQuality=&', 'bandIds=&', 'mosaicRule=&',
        'renderingRule=&', 'f=image'
    ])

    return url


def naip_from_tnm(bbox, bboxSR, size, imageSR,
                  format='tiff', pixelType='U8', f='image',
                  noDataInterpretation = 'esriNoDataMatchAny',
                  interpolation='+RSP_BilinearInterpolation',
                  **kwargs):
    """
    Retrieves a NAIP image from The National Map (TNM) web service.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    bboxSR : integer
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    size : list-like
      width, height of image to be returned
    imageSR : integer
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)
    f : str
      format of returned web request: 'image', 'json', 'html', or 'kmz'.
    pixelType: str
      data type for raster values
    **kwargs :
      additional parameters to the web service request. See https://services.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/

    Returns
    -------
    img : array
      NAIP image as a 3-band or 4-band array
    """
    BASEURL = ''.join(
        ['https://services.nationalmap.gov/arcgis/rest/services/',
         'USGSNAIPImagery/ImageServer/exportImage?']
        )

    KWARGS = {
      'time': None,
      'noData': None,
      'compression': None,
      'compressionQuality': None,
      'bandIds': None,
      'mosaicRule': None,
      'renderingRule': None,
    }

    params = {
      'bbox': ','.join([str(x) for x in bbox]),
      'bboxSR': bboxSR,
      'size': ','.join([str(x) for x in size]),
      'imageSR': imageSR,
      'format': format,
      'pixelType': pixelType,
      'noDataInterpretation': noDataInterpretation,
      'interpolation': interpolation,
      **KWARGS,
      'f': f,
    }

    r = requests.get(BASEURL, params=params)

    if f == 'image':
        img = imread(io.BytesIO(r.content))
        return img
    else:
        return r


def dem_from_tnm_url(xmin,
                     ymin,
                     xmax,
                     ymax,
                     bbox_epsg=3857,
                     img_epsg=3857,
                     img_format='tiff',
                     width=1024,
                     height=1024):
    """
  Returns URL for requesting an image from The National Map's Digital Elevation
  Model (DEM) REST service.

  Gotcha: If you request an image in a format other than 'tiff', the web
  service will return a hillshade rather than a DEM.

  Parameters
  ----------
  minx, miny, maxx, maxy : numeric
    the coordinates of the bounding box of the image to return, assumed to be
    in coordinate reference system EPSG:3857
  img_format : string
    image format to request, valid options are 'jpg', 'jpgpng', 'png', 'png8',
    'png24', 'png32', 'bmp', 'gif', 'tiff'.


  Returns
  -------
  url : string
    the url that will return the DEM image
  """

    url = ''.join([
        'https://elevation.nationalmap.gov/arcgis/rest/services/',
        '3DEPElevation/ImageServer/exportImage?',
        'bbox={}%2C{}%2C{}%2C{}&'.format(
            xmin, ymin, xmax,
            ymax), 'bboxSR={}&'.format(bbox_epsg), 'size={}%2C{}&'.format(
                width, height), 'imageSR={}&'.format(img_epsg), 'time=&',
        'format={}&'.format(img_format), 'pixelType=F32&', 'noData=&',
        'noDataInterpretation=esriNoDataMatchAny&',
        'interpolation=+RSP_BilinearInterpolation&'
        'compression=&', 'compressionQuality=&', 'bandIds=&', 'mosaicRule=&',
        'renderingRule=&', 'f=image'
    ])

    return url


def dem_from_tnm(bbox, **kwargs):
    """
    Retrieves a Digital Elevation Model (DEM) image from The National Map (TNM)
    web service.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)

    Returns
    -------
    img : numpy array
      DEM image as array
    """
    url = dem_from_tnm_url(*bbox, **kwargs)
    print('Retrieving image from {}'.format(url), end='... ')
    r = requests.get(url)
    img = imread(io.BytesIO(r.content))
    print('Done.')

    return img
