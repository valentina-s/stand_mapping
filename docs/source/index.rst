Stand Mapping
=============
This package is designed to facilitate segmentation and labeling of distinct
forest conditions (commonly referred to as "stands") by applying computer
vision approaches to various imagery sources. The core use case for this
software is for forest analysts to fit and apply reproducible stand-delineation
workflows using aerial imagery and other raster information (e.g.,
lidar-derived canopy height, biomass estimates, etc.). Apart from the
production of models for stand delineation that can be tuned using ground-truth
data and re-applied in other areas or as new imagery becomes available over
time, the outputs of these workflows will generally be georeferenced polygons
and/or rasters with each distinct stand indicated with a different label.

.. figure:: images/stands_on_aerial_image.png
   :width: 800
   :alt: Forest stand boundaries overlaid on an aerial image.

   Since widespread adoption of Geographic Information Systems, forest stands
   are now typically hand-delineated by foresters using high-resolution
   aerial imagery with supporting geospatial layers including property
   boundaries, roads, hydrographic features, and topography.

.. figure:: images/stands_on_lidar_chm.png
   :width: 800
   :alt: Forest stand boundaries overlaid on lidar-derived canopy height.

   The increasing availability of lidar data is enabling better interpretation
   of forest structure and stocking with unprecedented spatial resolution.
   Lidar-derived canopy height maps are particularly helpful for distinguishing
   between different forest types which may be less apparent from aerial
   imagery.

Pixels vs. Stands
#################
There has been a significant amount of research and product development related
to generating pixel-scale predictions of forest attributes. Our project departs
from this approach by recognizing the importance of aggregating the landscape
into practical management units which still remain the basis for most forest
conservation and management planning purposes.

The fundamental premise of this project is that human-drawn stand boundaries
provide a good place to start for teaching machines to recognize and attempt
to replicate how human managers delineate forest conditions for practical uses.
Our goal is to help refine and improve this process by providing additional
useful data but generating outputs in formats that remain central to the work
of forest owners and managers to understand what forest conditions exist and
how best they can steward them for a variety of objectives.

Generalization
##############

The term "stand" is generally used to refer a forested area ranging in size
from a few acres to more than one hundred acres that can be distinguished from
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

Stand Delineation Approach
##########################

We are currently working on two main segmentation approaches:

- Multi-stage segmentation pipelines linking together image preprocessing,
  over-segmentation, region-merging, and boundary post-processing using tools
  primarily drawn from from :code:`scikit-image`
- Instance segmentation using a region-based convolutional neural network,
  Mask-RCNN_, adapted from the `PyTorch implementation`_

A significant part of this effort involves the construction of a
:doc:`benchmarking dataset<benchmarking-dataset>` that includes several layers
of features as well as the targets which include bounding boxes and masks
distinguishing major land cover types (water, field, forest, impervious) and
the distinct instances of each cover type.


Contents
========

.. toctree::
   :maxdepth: 2

   getting-started
   benchmarking-dataset
   dependencies


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _Mask-RCNN: https://arxiv.org/abs/1703.06870
.. _`PyTorch implementation`: https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn
.. _`training dataset`: :doc:`training-dataset`
