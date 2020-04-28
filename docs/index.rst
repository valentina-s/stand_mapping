Stand Mapping
=============
This package is designed to facilitate the segmentation and labeling of
distinct forest conditions by applying computer vision approaches on various
types of imagery. It is built around the core use case of designing
reproducible workflows for delineating forest stands using aerial imagery and
other raster information (e.g., lidar-derived canopy height, biomass estimates,
etc.).

The term "stand" is generally used to refer a forested area ranging in size
from a few acres to a few hundred acres that can be distinguished from
adjacent forested areas based on species composition, tree size, density,
spatial arrangement, productivity, and/or management history and accessibility.

In the context of object detection and delineation, the concept of a "stand"
can also be generalized to refer to any region of interest that can be
distinguished from adjacent regions using available input data. Any objects on
the landscape that that can be represented with polygons/masks can be the
"targets" our segmentation workflows aim to support. Examples include
identifying areas with forest composition or structure that provides suitable
habitat for a particular wildlife species, or areas that have been affected by
natural of human disturbances with varying levels of intensity such as
harvests, pathogens, or fire.

We are currently working on two main segmentation approaches:

- Multi-stage segmentation pipelines linking together image preprocessing,
  over-segmentation, region-merging, and boundary post-processing using tools
  primarily drawn from from :code:`scikit-image`
- Instance segmentation using a region-based convolutional neural network,
  Mask-RCNN_, adapted from the `PyTorch implementation`_

A significant part of this effort involves the construction of a
:doc:`bencmarking dataset<benchmarking-dataset>` that includes several layers
of features as well as the targets which include bounding boxes and masks
distinguishing major land cover types (water, field, forest, impervious) and
the distinct instances of each cover type.


Contents
========

.. toctree::
   :maxdepth: 2

   getting-started
   benchmarking-dataset


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _Mask-RCNN: https://arxiv.org/abs/1703.06870
.. _`PyTorch implementation`: https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn
.. _`training dataset`: :doc:`training-dataset`
