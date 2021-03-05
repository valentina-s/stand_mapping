import numpy as np

def batchify(targets, predictions, nodata, score_func,
             aggregate=True, *args, **kwargs):
    """
    Applies a scoring function to each sample in a batch.

    Parameters
    ----------
    score_func : callable
      ...
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

# ious
# maskf1
# boundaryf1
