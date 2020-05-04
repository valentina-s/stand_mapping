from skimage.segmentation import (slic, quickshift, felzenszwalb, watershed,
                                  mark_boundaries)

SEG_FUNCS = {
    'slic': slic,
    'quickshift': quickshift,
    'felzenszwalb': felzenszwalb,
    'watershed': watershed,
}


def seg2labels(img, seg_func, with_boundaries=False, **kwargs):
    """
    Wrapper function that applies a segmentation routine to an image and
    optionally returns segment boundaries.

    Parameters
    ----------
    img : array
      image to be segmented
    seg_func : str or function
      name of a segmentation function (currently supports 'slic', 'quickshift',
      'felzenszwalb', and 'watershed') or a callable that accepts an image as
      its first argument and returns an integer-labeled array.
    with_boundaries : bool
      whether or not to return an image with boundaries between segments
      superimposed on the original image in addition to the
    **kwargs
      keyword arguments that will be passed to the segmentation function.

    Returns
    -------
    labels : array
      Integer mask indicating segment labels.
    marked : array
      An image in which the boundaries between labels are superimposed on the
      original image.
    """
    if isinstance(seg_func, str):
        if seg_func not in SEG_FUNCS.keys():
            raise ValueError(
                '{} is not a recognized segmentation function name.'.format(
                    seg_func))
        else:
            seg_func = SEG_FUNCS[seg_func]

    if not callable(seg_func):
        raise ValueError('seg_func must be a string or callable function.')

    labels = seg_func(img, **kwargs)

    if with_boundaries:
        marked = mark_boundaries(img, labels, color=(1, 0, 0))
        return labels, marked
    else:
        return segment_mask
