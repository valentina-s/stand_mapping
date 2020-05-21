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
from skimage.morphology import binary_dilation,disk

def single_to_multichannel_mask(mask_img):
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
    obj_ids = np.unique(mask_img[mask_img > 0])
    masks = np.dstack([np.array(mask_img == id) for id in obj_ids])

    return masks


def masks_iou(gt_masks, pred_masks):
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

    Returns
    -------
    overlaps : array, shape (gt_instances,pred_instances)
      array containing iou scores for each ground truth mask and each predicted
      mask.
    """
    masks1 = single_to_multichannel_mask(gt_masks)
    masks2 = single_to_multichannel_mask(pred_masks)

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
    overlaps = intersections / union

    if threshold:
        overlaps = np.where(overlaps > threshold, overlaps, 0)

    return overlaps


def mask_f1_score(mask_ious, iou_thresh=0.5):
    """Calculates Precision, Recall, and F1 Score given an array with IoU
    scores for ground truth vs. predicted instance masks.

    IoU scores greater than `iou_thresh` are considered true positive
    detections.

    Parameters
    ----------
    mask_ious : array, shape (gt_instances, pred_instances)
      iou scores for each combination of ground truth instance masks and
      predicted instance masks, generated using the `masks_iou` function
    iou_thresh : numeric, default = 0.5
      value for intersection over union between two masks for which an instance
      is considered detected or not detected

    Returns
    -------
    f1, precision, recall : numeric
      F1, Precision, and Recall scores comparing ground truth and predicted
      masks
    """

    above_thresh = mask_ious > iou_thresh

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

    return f1, precision, recall


def boundary_f1_score(gt_masks, pred_masks, bound_thresh=0.008):
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
    bound_thresh : numeric
      distance threshold from a boundary that will be accepted as a detection

	Returns
    -------
    f1, precision, recall : numeric
      F1, Precision, and Recall scores comparing ground truth and predicted
      masks
	"""
    gt_boundary = find_boundaries(gt_masks)
    pred_boundary = find_boundaries(pred_masks)

    bound_pix = bound_thresh if bound_thresh >= 1 else \
    np.ceil(bound_thresh*np.linalg.norm(pred_masks.shape))

    pred_dil = binary_dilation(pred_boundary,disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * pred_dil
    pred_match = pred_boundary * gt_dil

    # Area of the intersection
    n_pred = np.sum(pred_boundary)
    n_gt = np.sum(gt_boundary)

    #% Compute precision and recall
    if n_pred == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_pred > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_pred == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(pred_match)/float(n_pred)
        recall = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall);

    return f1, precision, recall
