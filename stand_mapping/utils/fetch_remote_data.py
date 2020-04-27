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
    ---------
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
    r = requests.post(BASE_URL, json=extent, headers=api_header).json()
    cover = imread(io.BytesIO(base64.b64decode(r['output_hard'])))

    return cover
