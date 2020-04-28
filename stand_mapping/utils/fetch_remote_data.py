import requests
import io
import base64
from imageio import imread


def get_cover_from_extent(bbox, epsg, api_key,
                          weights=[0.25, 0.25, 0.25, 0.25]):
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


    Returns
    -------
    cover : array
      a 3-band image with distinct colors for each cover type
    """
    BASE_URL= 'https://aiforearth.azure-api.net/landcover/v2/classifybyextent'
    api_header = {'Ocp-Apim-Subscription-Key': api_key,
                  'Content-Type':'application/json'}

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
    print('Retrieving image from AI for Earth Land Cover API', end='...')
    r = requests.post(BASE_URL, json=extent, headers=api_header).json()
    cover = imread(io.BytesIO(base64.b64decode(r['output_hard'])))
    print('Done.')

    return cover


def get_naip_tnm_url(xmin, ymin, xmax, ymax,
                     bbox_epsg=3857,
                     img_epsg=3857,
                     img_format='jpgpng',
                     width=500, height=500):
  """
  Returns the URL for making a request to get an image from The National Map's
  NAIP REST service.

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

  url = ''.join(['https://services.nationalmap.gov/arcgis/rest/services/',
  'USGSNAIPImagery/ImageServer/exportImage?',
  'bbox={}%2C{}%2C{}%2C{}&'.format(xmin, ymin, xmax, ymax),
  'bboxSR={}&'.format(bbox_epsg),
  'size={}%2C{}&'.format(width, height),
  'imageSR={}&'.format(img_epsg),
  'time=&',
  'format={}&'.format(img_format),
  'pixelType=U8&',
  'noData=&',
  'noDataInterpretation=esriNoDataMatchAny&',
  'interpolation=+RSP_BilinearInterpolation&'
  'compression=&',
  'compressionQuality=&',
  'bandIds=&',
  'mosaicRule=&',
  'renderingRule=&',
  'f=image'])

  return url


def get_naip_from_tnm(bbox, **kwargs):
  """
  Retrieves a NAIP image from The National Map web service and returns it as
  a numpy array.

  Parameters
  ----------
  bbox : list-like
    list of bounding box coordinates (minx, miny, maxx, maxy)

  Returns
  -------
  img : array
    NAIP image as a 3-band or 4-band array
  """
  url = get_naip_tnm_url(*bbox, **kwargs)
  print('Retrieving NAIP image from The National Map', end='...')
  r = requests.get(url)
  img = imread(io.BytesIO(r.content))
  print('Done.')

  return img
