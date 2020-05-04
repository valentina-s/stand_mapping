import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon


def make_buffered_fishnet(xmin, ymin, xmax, ymax, crs, spacing=1000,
                          overlap=50):
    """Makes GeoDataFrame with a fishnet grid with optional overlapping edges.

    Converts an existing lidar tiling scheme into one that has overlapping
    tiles and which is aligned with a grid based on the spacing parameter.

    Parameters
    ----------
    xmin, ymin, xmax, ymax : numeric
      Values indicating the extent of the existing lidar data.
    crs : Coordinate Reference System
      Must be readable by GeoPandas to create a GeoDataFrame.
    spacing : int
      Length and width of tiles in new tiling scheme prior to buffering
    overlap : int
      Amount of overlap between neighboring tiles.
    """
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax + spacing, spacing),
        np.arange(ymin, ymax + spacing, spacing))

    xx_leftbuff = xx[:, :-1] - overlap
    xx_rightbuff = xx[:, 1:] + overlap
    yy_downbuff = yy[:-1, :] - overlap
    yy_upbuff = yy[1:, :] + overlap

    ll = np.stack((
        xx_leftbuff[1:, :].ravel(),  # skip top row
        yy_downbuff[:, :-1].ravel())).T  # skip right-most column

    ul = np.stack((
        xx_leftbuff[:-1, :].ravel(),  # skip bottom row
        yy_upbuff[:, :-1].ravel())).T  # skip right-most column

    ur = np.stack((
        xx_rightbuff[:-1, :].ravel(),  # skip bottom row
        yy_upbuff[:, 1:].ravel())).T  # skip left-most column

    lr = np.stack((
        xx_rightbuff[1:, :].ravel(),  # skip top row
        yy_downbuff[:, 1:].ravel())).T  # skip left-most column

    buff_fishnet = np.stack([ll, ul, ur, lr])

    polys = [
        Polygon(buff_fishnet[:, i, :]) for i in range(buff_fishnet.shape[1])
    ]
    ll_names = [x for x in (ll).astype(int).astype(str)]
    tile_ids = [
        '_'.join(tile) + '_{}'.format(str(spacing)) for tile in ll_names
    ]

    buff_fishnet_gdf = gpd.GeoDataFrame(geometry=polys, crs=crs)
    buff_fishnet_gdf['tile_id'] = tile_ids

    return buff_fishnet_gdf.set_index('tile_id')
