"""
Utility functions for scoring instance segmentations against ground truth
images.

`masks_iou` function is adapted from
https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py

Mask R-CNN
The MIT License (MIT)
Copyright (c) 2017 Matterport, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

`boundary_f1_score` is adapted from
https://github.com/fperazzi/davis/blob/master/python/lib/davis/measures/f_boundary.py

                  Copyright (c) 2016, Federico Perazzi
                        All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* The names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Federico Perazzi 'AS IS' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL Federico Perazzi BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk
from rasterio import features
from scipy.ndimage.morphology import distance_transform_edt as edt


def rasterize_polygons(gdf, out_shape, transform):
    """Rasterizes a GeoDataFrame such that each distinct geometry is rendered
    with a distinct integer value in the output raster.

    Parameters
    ----------
    gdf : GeoDataFrame
      GeoDataFrame to be rasterized
    out_shape : 2-tuple or list-like
      (height, width) of desired output raster
    transform : Affine
      rasterio-style (not GDAL-style) affine transformation matrix which
      translates pixel coordinates to geographic coordinates

    Returns
    -------
    ras : arr
      array with each geometry labeled with a distinct integer
    """
    ras = np.zeros(out_shape, dtype=np.int16)
    for i, geom in enumerate(gdf.geometry.buffer(0).dropna()):
        mask = features.geometry_mask([geom], out_shape=out_shape,
                                      transform=transform, invert=True)
        ras[mask] = i+1

    return ras


def rasterize_boundary(gdf, out_shape, transform, dist=False):
    """Rasterizes lines and boundaries of polygons in a GeoDataFrame such that
    the presence of a boundary or line is indicated with a 1 (present) or 0
    (absent). Optionally, can return the Euclidean distance from the nearest
    boundary instead of the binary mask.

    Parameters
    ----------
    gdf : GeoDataFrame
      GeoDataFrame whose lines and boundaries will be rasterized
    out_shape : 2-tuple or list-like
      (height, width) of desired output raster
    transform : Affine
      rasterio-style (not GDAL-style) affine transformation matrix which
      translates pixel coordinates to geographic coordinates
    dist : bool
      if False (default), returns a binary mask for boundaries in the
      GeoDataFrame, otherwise returns Euclidean distance to the nearest
      boundary.

    Returns
    -------
    ras : arr
      array with binary boundary presence/absence or distance to boundary
    """
    geoms = gdf.loc[~gdf.geometry.is_empty].buffer(0).dropna().boundary
    if dist:
        ras = features.rasterize(geoms, out_shape=out_shape,
                                 transform=transform, fill=1, default_value=0)
        ras = edt(ras)
    else:
        ras = features.rasterize(geoms, out_shape=out_shape,
                                 transform=transform, fill=0, default_value=1)
    return ras


def single_to_multichannel_mask(mask_img, num_classes=None):
    """From a single array with instances indicated by distinct non-zero
    integer masks, returns a stack of arrays with each object shown in a
    single array.

    Parameters
    ----------
    mask_img : array, shape (width, height)
      array with each instance to be detected indicated by unique non-zero
      integer mask.

    Returns
    -------
    masks : array, shape (width, height, instances)
      boolean arrays with mask of each instance indicated as True.
    """
    # filter out background pixels (value of 0)
    if num_classes is not None:
        classes = np.unique(mask_img[mask_img > 0])
    else:
        classes = np.arange(num_classes) + 1

    masks = np.dstack([np.array(mask_img == cls) for cls in classes])

    return masks


def masks_iou(gt_masks, pred_masks, nodata=None, num_classes=None):
    """Computes the Intersection over Union (IoU) for each combination of
    masks in a ground-truth image and predicted segmentation/labels image.

    Parameters
    ----------
    gt_masks : array, shape (height, width)
      ground truth masks represented as a single channel array with each
      distinct object in the image indicate with a different non-zero integer
    pred_masks : array, shape (height, width)
      predicted masks represented as a single channel array with each
      distinct object in the image indicate with a different non-zero integer
    nodata : array, shape (height, width)
      optional nodata mask; pixels where nodata is True will be excluded from
      scoring (set to 0)
    num_classes : int, optional
      number of classes that occur across gt_masks and pred_masks. If not
      specified, only the unique classes observed in each layer will be used.

    Returns
    -------
    iou : array, shape (gt_instances, pred_instances)
      array containing iou scores for each ground truth mask and each predicted
      mask.
    """
    if nodata is not None:
        gt_masks[nodata] = 0
        pred_masks[nodata] = 0

    # check for all zeros, if all zeros, returns ious as all nans
    if gt_masks.max() == 0 or pred_masks.max() == 0:
        if num_classes is not None:
            return np.full((num_classes, num_classes), np.nan)
        else:
            return np.full((1,1), np.nan)

    masks1 = single_to_multichannel_mask(gt_masks, num_classes=num_classes)
    masks2 = single_to_multichannel_mask(pred_masks, num_classes=num_classes)

    # if either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    ious = intersections / union

    return ious


def mask_f1_score(gt_masks, pred_masks, nodata=None,
                  num_classes=None, iou_thresh=0.5):
    """Calculates Precision, Recall, and F1 Score given an array with IoU
    scores for ground truth vs. predicted instance masks.

    IoU scores greater than `iou_thresh` are considered true positive
    detections.

    Parameters
    ----------
    gt_masks : array, shape (height, width)
      ground truth masks represented as a single channel array with each
      distinct object in the image indicate with a different non-zero integer
    pred_masks : array, shape (height, width)
      predicted masks represented as a single channel array with each
      distinct object in the image indicate with a different non-zero integer
    nodata : array, shape (height, width)
      optional nodata mask; pixels where nodata is True will be excluded from
      scoring (set to 0)
    num_classes : int, optional
      number of classes that occur across gt_masks and pred_masks. If not
      specified, only the unique classes observed in each layer will be used.
    iou_thresh : numeric, default = 0.5
      value for intersection over union between two masks for which an instance
      is considered detected or not detected

    Returns
    -------
    iou, f1, precision, recall : numeric
      average IOU, F1, Precision, and Recall scores comparing ground truth and
      predicted masks
    """
    if nodata is not None:
        gt_masks[nodata] = 0
        pred_masks[nodata] = 0
    ious = masks_iou(gt_masks, pred_masks,
                     nodata=nodata, num_classes=num_classes)

    # non-max suppression to calculate average IOU
    # choosing best-matching predicted mask for each ground-truth class
    iou = ious.max(axis=1).mean()  # predicted class, max overlap

    above_thresh = ious > iou_thresh

    if above_thresh.shape[1] == 0:
        precision = 0
    else:
        precision = above_thresh.max(
            axis=0).sum() / above_thresh.shape[1]  # tp / (tp + fp)

    if (above_thresh.shape[0] - above_thresh.max(axis=0)).sum() == 0:
        recall = 0
    else:
        recall = above_thresh.max(
            axis=0).sum() / (above_thresh.shape[0] -
                             above_thresh.max(axis=0)).sum()  # tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return iou, f1, precision, recall


def boundary_f1_score(gt_masks, pred_masks, nodata=None, bound_thresh=2):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Parameters
    ----------
    gt_masks : array, shape (height, width)
      ground truth masks represented as a single channel array with each
      distinct object in the image indicate with a different non-zero integer
    pred_masks : array, shape (height, width)
      predicted masks represented as a single channel array with each
      distinct object in the image indicate with a different non-zero integer
    bound_thresh : int
      distance threshold from a boundary that will be accepted as a detection,
      in pixels
    nodata : array, shape (height, width)
      optional nodata mask; pixels where nodata is True will be excluded from
      scoring (set to 0)

    Returns
    -------
    iou, dilated_iou, f1, precision, recall : numeric
      IOU, IOU with dilated boundaries, F1, Precision, and Recall scores
      comparing ground truth and predicted masks
    """
    if nodata is not None:
        gt_masks[nodata] = 0
        pred_masks[nodata] = 0
    gt_boundary = find_boundaries(gt_masks)
    pred_boundary = find_boundaries(pred_masks)

    intersection = np.logical_and(gt_boundary > 0, pred_boundary > 0)
    union = np.logical_or(gt_boundary > 0, pred_boundary > 0)
    iou = intersection.sum() / union.sum()

    pred_dil = binary_dilation(pred_boundary, disk(bound_thresh))
    gt_dil = binary_dilation(gt_boundary, disk(bound_thresh))

    dilated_inter = np.logical_and(gt_dil > 0, pred_dil > 0)
    dilated_union = np.logical_or(gt_dil > 0, pred_dil > 0)
    dilated_iou = dilated_inter.sum() / dilated_union.sum()

    # Get the intersection
    gt_match = gt_boundary * pred_dil
    pred_match = pred_boundary * gt_dil

    # Area of the intersection
    n_pred = np.sum(pred_boundary)
    n_gt = np.sum(gt_boundary)

    # Compute precision and recall
    if n_pred == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_pred > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_pred == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(pred_match)/float(n_pred)
        recall = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)

    return iou, dilated_iou, f1, precision, recall
