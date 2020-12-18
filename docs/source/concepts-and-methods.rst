Concepts & Methods
==================
The technical tasks we are focused on in forest mapping involve delineating and
classifying the conditions within forest stands, which are considered as two
distinct steps.

The term "stand" is generally used to refer a forested area ranging in size
from a few acres to more than one hundred acres that can be distinguished from
adjacent forested areas based on species composition, tree size, density,
spatial arrangement, productivity, and/or management history and accessibility.

Generalizing the Stand
######################
In the context of object detection and delineation, the concept of a "stand"
can also be generalized to refer to any region of interest that can be
distinguished from adjacent regions using available input data. Any objects on
the landscape that that can be represented with polygons/masks can be the
"targets" of segmentation workflows like the ones we adopt. Examples include
identifying areas with forest composition or structure that provides suitable
habitat for a particular wildlife species, or areas that have been affected by
natural of human disturbances with varying levels of intensity such as
harvests, pathogens, or fire.

Pixels vs. Stands
#################
There has been a significant amount of research and product development related
to generating pixel-scale predictions of forest attributes. Our project departs
from this approach by recognizing the importance of aggregating the landscape
into practical management units which still remain the basis for most forest
conservation and management planning purposes.

The fundamental premise of the stand mapping aspect of this project is that
human-drawn stand boundaries provide a good place to start for teaching
machines to recognize and attempt to replicate how human managers delineate
forest conditions for practical uses.

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
distinguishing major land cover types (water, field, forest, barren,
impervious) and the distinct instances of each cover type.

.. _Mask-RCNN: https://arxiv.org/abs/1703.06870
.. _`PyTorch implementation`: https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn
.. _`training dataset`: :doc:`training-dataset`
