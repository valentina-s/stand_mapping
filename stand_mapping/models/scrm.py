import os
import numpy as np

import geopandas as gpd
import rasterio
from rasterio.plot import reshape_as_image
from rasterio import features, windows
from affine import Affine
from shapely.geometry import box
from rastachimp import as_shapely, simplify_dp, smooth_chaikin

# pre-processing images
from skimage.color import rgb2hsv
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage.exposure import equalize_adapthist, match_histograms
from skimage import filters
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import downscale_local_mean
from skimage.util import img_as_float

# segmenting images
import heapq
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.future import graph
from skimage.future.graph.graph_merge import (_revalidate_node_edges,
                                              _invalidate_edge,
                                              _rename_node)


def parse_stand_path(path_to_file):
    """Parses useful information from the path to a stand delineation layer."""
    dirname, basename = os.path.split(path_to_file)
    cell_id = int(basename.split('_')[0])
    year = int(basename.split('_')[-1].split('.')[0])
    agency = basename.split('_')[1]

    if 'oregon' in dirname:
        state_name = 'oregon'
    elif 'washington' in dirname:
        state_name = 'washington'
    return dirname, cell_id, state_name, year, agency


def get_naip_path(root_dir, cell_id, state_name, stands_year):
    """Fetch path to the NAIP image for a tile nearest to stands_year"""
    if state_name == 'washington':
        YEARS = np.array([2009, 2011, 2015, 2017])
    elif state_name == 'oregon':
        YEARS = np.array([2009, 2011, 2014, 2016])
    best_year = YEARS[np.argmin(abs(YEARS - stands_year))]

    root_dir = (root_dir.replace('interim', 'processed')
                .replace('stands', 'naip'))
    dirname = f'{root_dir}/{best_year}'
    fname = f'{cell_id}_naip_{best_year}.tif'
    path_to_file = os.path.join(dirname, fname)
    return path_to_file


def get_landsat_path(root_dir, cell_id, state_name, stands_year):
    """Fetch path to the LANDSAT leaf-on image for a tile nearest to
    stands_year.
    """
    if state_name == 'washington':
        YEARS = np.array([2009, 2011, 2015, 2017])
    elif state_name == 'oregon':
        YEARS = np.array([2009, 2011, 2014, 2016])
    best_year = YEARS[np.argmin(abs(YEARS - stands_year))]

    root_dir = (root_dir.replace('interim', 'processed')
                .replace('stands', 'landsat'))
    dirname = f'{root_dir}/{best_year}'
    fname = f'{cell_id}_landsat-leaf-on_{best_year}.tif'
    path_to_file = os.path.join(dirname, fname)
    return path_to_file


def load_data(stand_path, chip_size=None, offsets=None):
    """Loads NAIP, LANDSAT, and stand delineation data"""
    dirname, cell_id, state_name, year, agency = parse_stand_path(stand_path)
    naip_path = get_naip_path(dirname, cell_id, state_name, year)
    landsat_path = get_landsat_path(dirname, cell_id, state_name, year)

    with rasterio.open(naip_path) as src:
        profile = src.profile
        height, width = src.shape
        if chip_size is not None:
            if offsets is not None:
                row_off, col_off = offsets
            else:
                row_off = np.random.randint(0, height-chip_size)
                col_off = np.random.randint(0, width-chip_size)
            window = windows.Window(col_off, row_off, chip_size, chip_size)
        else:
            window = None

        naip = reshape_as_image(src.read(window=window))
        if window is not None:
            trf = src.window_transform(window)
            bbox = src.window_bounds(window)
        else:
            trf = src.transform
            bbox = src.bounds

    with rasterio.open(landsat_path) as src:
        if chip_size is not None:
            window = windows.from_bounds(*bbox,
                                         transform=src.transform,
                                         height=chip_size,
                                         width=chip_size)
        else:
            window = windows.from_bounds(*bbox,
                                         transform=src.transform,
                                         height=height,
                                         width=width)
        landsat = ((
            reshape_as_image(
                np.stack(
                    [src.read(band+1, window=window) for band in range(4)]
                    ))/3000).clip(0, 1)*255).astype(np.uint8)

    stands = gpd.read_file(stand_path)
    stands = gpd.clip(stands, box(*bbox))

    return naip, landsat, profile, trf, stands


@adapt_rgb(each_channel)
def sobel_each(image, *args, **kwargs):
    return filters.sobel(image, *args, **kwargs)


@adapt_rgb(hsv_value)
def sobel_hsv(image, *args, **kwargs):
    return filters.sobel(image, *args, **kwargs)


def calc_ndvi(img):
    r, nir = img[:, :, 0] + 1e-9, img[:, :, 3] + 1e-9
    ndvi = (nir - r) / (nir + r)
    return ndvi.clip(-1.0, 1.0)


def transform_image(src_img, transform, match_img=None, ndvi=False,
                    enhance_contrast=True, downscale=5, denoise=True):
    """Applies some transformations to an image useful before segmentation."""
    if ndvi:
        img, multichannel = calc_ndvi(src_img), False
        if match_img is not None:
            match_img = calc_ndvi(match_img)
    else:
        img, multichannel = img_as_float(src_img[:, :, 0:3]), True
        if match_img is not None:
            match_img = img_as_float(match_img[:, :, 0:3])

    if match_img is not None:
        img = match_histograms(img, match_img, multichannel=multichannel)

    if enhance_contrast:
        img = equalize_adapthist(img)

    if downscale is not None:
        factors = (downscale, downscale, 1) if multichannel else \
                  (downscale, downscale)
        img = downscale_local_mean(img, factors=factors)
        a, b, c, d, e, f, _, _, _ = transform
        trf = Affine(downscale, b, c, d, -downscale, f)

    if denoise:
        img = denoise_tv_chambolle(img, multichannel=multichannel)

    return img, trf


def oversegment(image, min_distance, downscale, multichannel=False):
    """Oversegments an image using watershed segmentation on image gradient."""
    if multichannel:
        grad = sobel_hsv(rgb2hsv(image), mode='reflect').max(axis=-1)
    else:
        grad = filters.sobel(image, mode='reflect')

    pixel_dist = max(min_distance//downscale, 1)
    peaks = peak_local_max(-grad, min_distance=pixel_dist,
                           indices=False, exclude_border=0)
    markers = (peaks.ravel() * peaks.ravel().cumsum()).reshape(*grad.shape)
    basins = watershed(grad, markers=markers)

    return basins


def scrm(image, labels, dms, mmu, mas, downscale):
    """Applies Size-Constrained Region Merging.

    Parameters
    ----------
    image : arr
      image being segmented
    labels : arr
      initial (oversegmented) regions
    dms : int
      desired mean size of merged regions, in acres
    mas : int
      maximum allowed size of merged regions, in acres
    mmu : int
      minimum mappable unit, in acres

    Returns
    -------
    regions : arr
      array of same shape as image, with each distinct region indicated by
      increasing integer values
    """
    dms_pixels = 4047 * dms / (downscale*downscale)
    mas_pixels = 4047 * mas / (downscale*downscale)
    mmu_pixels = 4047 * mmu / (downscale*downscale)

    rag = graph.rag_mean_color(image, labels)
    regions = merge_size_constrained(labels, rag,
                                     dms_pixels, mas_pixels, mmu_pixels,
                                     rag_copy=False, in_place_merge=True,
                                     merge_func=merge_scrm,
                                     weight_func=weight_scrm,
                                     ).astype(np.int16)

    return regions


def vectorize(regions, transform, crs, simp_dist=5, smooth=True):
    """Vectorizes boundaries of regions in a labeled image."""
    # vectorize regions to GeoJSON
    shapes = features.shapes(regions, transform=transform)
    # buffer each polygon geometry by 0 to resolve topological errors
    shapes = [(x[0].buffer(0), x[1]) for x in as_shapely(shapes)]
    # simplify boundaries using Douglas-Peucker algorithm
    if simp_dist is not None:
        shapes = simplify_dp(shapes, distance=simp_dist)
    # smooth boundaries using Chaikin corner cutting algorithm
    if smooth:
        shapes = smooth_chaikin(shapes, keep_border=True)
    # convert to a GeoDataFrame
    gdf = gpd.GeoDataFrame(shapes, columns=['geometry', 'stand_id'], crs=crs)

    return gdf


def run_segmentation(path_to_stands, chip_size=None, downscale=5, ndvi=False,
                     min_marker_distance=10, dms=10, mmu=1, mas=30,
                     simp_dist=10):
    data = load_data(path_to_stands, chip_size=chip_size)
    naip, landsat, profile, full_trf, obs_stands = data

    img, down_trf, = transform_image(naip, full_trf, match_img=landsat,
                                     ndvi=ndvi, downscale=downscale)

    basins = oversegment(img, min_distance=min_marker_distance,
                         downscale=downscale,
                         multichannel=~ndvi)

    regions = scrm(img, basins, dms=dms, mmu=mmu, mas=mas, downscale=downscale)

    pred_stands = vectorize(regions, down_trf,
                            crs=profile['crs'],
                            simp_dist=simp_dist)

    return obs_stands, pred_stands, naip, full_trf


def merge_size_constrained(labels, rag, dms, mas, mmu,
                           rag_copy, in_place_merge,
                           merge_func, weight_func):
    """Perform Size-Constrained Region Merging on a RAG.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The Region Adjacency Graph.
    dms : int
        Desired Mean Size of regions, in pixels.
    mas : int
        Maximum Allowed Size of regions, in pixels. Note: Not a hard cap.
    mmu : int
        Minimum Mappable Unit, minimum size of regions, in pixels.
    rag_copy : bool
        If set, the RAG copied before modifying.
    in_place_merge : bool
        If set, the nodes are merged in place. Otherwise, a new node is
        created for each merge..
    merge_func : callable
        This function is called before merging two nodes. For the RAG `graph`
        while merging `src` and `dst`, it is called as follows
        ``merge_func(graph, src, dst)``.
    weight_func : callable
        The function to compute the new weights of the nodes adjacent to the
        merged node. This is directly supplied as the argument `weight_func`
        to `merge_nodes`.
    Returns
    -------
    out : ndarray
        The new labeled array.
    """
    if rag_copy:
        rag = rag.copy()

    edge_heap = []

    # a couple attributes we'll track to enforce a partial stopping criterion
    rag.graph.update({
        'num_ge_mmu': 0,  # number of regions >= mmu size
        'area_lt_mmu': 0,  # total area in regions smaller than mmu size
     })

    total_area = 0  # total area in regions/image

    for n in rag:
        area = rag.nodes[n]['pixel count']
        total_area += area
        if area < mmu:
            rag.graph['area_lt_mmu'] += area
        else:
            rag.graph['num_ge_mmu'] += 1

    exp_final_num = total_area // dms  # expected number of regions

    for n1, n2, data in rag.edges(data=True):
        # Push a valid edge in the heap
        wt = data['weight']
        heap_item = [wt, n1, n2, True]
        heapq.heappush(edge_heap, heap_item)

        # Reference to the heap item in the graph
        data['heap item'] = heap_item

    partial_stop = False
    while len(edge_heap) > 0:
        _, n1, n2, valid = heapq.heappop(edge_heap)

        num_ge_mmu = rag.graph['num_ge_mmu']
        area_lt_mmu = rag.graph['area_lt_mmu']
        if ((num_ge_mmu + (area_lt_mmu/dms)) < exp_final_num) and \
                not partial_stop:
            partial_stop = True

        # if the best fitting pair consists of two regions both exceeding MAS,
        # then it is not allowed to merge

        # The merging continues this way until the sum of (a) the number of
        # regions currently larger than the minimum allowed size MMU, plus (b)
        # the expected number of final regions that may result from the area
        # currently occupied by regions smaller than MMU, is less than the
        # expected number of final regions (i.e., the image area divided by
        # DMS).

        # Thereafter, the candidate list is restricted only to those pairs
        # where at least one of both regions is smaller than MMU.

        if valid:
            n1_area = rag.nodes[n1]['pixel count']
            n2_area = rag.nodes[n2]['pixel count']
            if n1_area > mas and n2_area > mas:
                valid = False
            if n1_area > mas and n2_area > mmu:
                valid = False
            if n1_area > mmu and n2_area > mas:
                valid = False
            if partial_stop:
                if n1_area >= mmu and n2_area >= mmu:
                    valid = False

        # Ensure popped edge is valid, if not, the edge is discarded
        if valid:
            # Invalidate all neigbors of `src` before its deleted
            for nbr in rag.neighbors(n1):
                _invalidate_edge(rag, n1, nbr)

            for nbr in rag.neighbors(n2):
                _invalidate_edge(rag, n2, nbr)

            if not in_place_merge:
                next_id = rag.next_id()
                _rename_node(rag, n2, next_id)
                src, dst = n1, next_id
            else:
                src, dst = n1, n2

            merge_func(rag, src, dst, mmu)
            new_id = rag.merge_nodes(src, dst, weight_func)
            _revalidate_node_edges(rag, new_id, edge_heap)

    label_map = np.arange(labels.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes(data=True)):
        for label in d['labels']:
            label_map[label] = ix

    return label_map[labels]


def weight_scrm(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)

    return {'weight': diff}


def merge_scrm(graph, src, dst, mmu):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    src_area = graph.nodes[src]['pixel count']
    dst_area = graph.nodes[dst]['pixel count']

    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = graph.nodes[dst]['total color'] /\
        graph.nodes[dst]['pixel count']

    new_area = graph.nodes[dst]['pixel count']

    d_num_ge_mmu = (new_area >= mmu) - (src_area >= mmu) - (dst_area >= mmu)

    d_area_lt_mmu = (new_area < mmu)*new_area - \
                    (src_area < mmu)*src_area - \
                    (dst_area < mmu)*dst_area

    graph.graph['num_ge_mmu'] += d_num_ge_mmu
    graph.graph['area_lt_mmu'] += d_area_lt_mmu
