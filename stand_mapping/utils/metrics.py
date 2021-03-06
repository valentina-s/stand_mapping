import numpy as np


def batchify(targets, predictions, nodata, score_func,
             aggregate=True, *args, **kwargs):
    """
    Applies a scoring function to each sample image in a batch.

    Parameters
    ----------
    targets : array-like, shape (batch_size, height, width)
      observed, ground-truth target images
    predictions : array-like, shape (batch_size, height, width)
      predicted images
    nodata : array-like, shape (batch_size, height, width)
      nodata masks indicating areas where differences between targets and
      predictions will be ignored
    score_func : callable
      scoring function that will be called on each sample, should expect
      targets, predictions and nodata as arguments, with optional args and
      kwargs to follow.
    aggregate : bool
      whether to return the average of each metrics across the batch (default),
      or to return each of the metrics for each of the samples in the batch
    """
    results = []
    for targ, pred, msk in zip(targets, predictions, nodata):
        # unpack the batch
        # X, Y, Z = batch
        results.append(score_func(targ, pred, msk, *args, **kwargs))

    if aggregate:
        results = tuple(np.array(results).mean(axis=0))

    return results
