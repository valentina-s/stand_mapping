Stand Mapping
=============
This webpage documents an open-source data science and software engineering
effort to produce new tools that segment and label forest conditions using
computer vision approaches that can be applied with a variety of imagery and
other data sources.

.. figure:: images/stands_on_aerial_image.png
   :width: 800
   :alt: Forest stand boundaries overlaid on an aerial image.

   Since widespread adoption of Geographic Information Systems, forest stands
   are typically hand-delineated by foresters using high-resolution
   aerial imagery with supporting geospatial layers including property
   boundaries, roads, hydrographic features, and topography.

Motivation
##########
This software is being developed as part of a broader project with the goal of
increasing landowner engagement and adoption of stewardship and conservation
activities in forests across Oregon and Washington.
Most pathways to adopting new practices involve assessing current and
historical forest conditions and preparing plans to implement and follow-up on
new activities. This typically involves mapping and characterizing forest
stands and gathering publicly-available information in the forms of maps and
tables. This work and information can be costly, time-consuming, or difficult
to access, limiting the adoption of new stewardship practices by landowners
who might otherwise be inclined to do so.
We are investigating whether and how a web-based forest mapping and planning
toolkit could reduce these barriers.

.. figure:: images/example_tile.png
   :width: 800
   :alt: Examples of data layers used for training forest models.

   We are utilizing a stack of data layers to help train new algorithms
   to delineate land cover "instances" such as forest stands. Our models are
   based entirely upon free and publicly-available datasets including
   high-resolution aerial imagery, leaf-on and leaf-off satellite imagery,
   roads, hydrologic features, parcel boundaries, disturbance history, and
   terrain and land cover datasets.


Contents
========

.. toctree::
   :maxdepth: 1

   concepts-and-methods
   benchmarking-dataset
   getting-started
   dependencies
   sare-project
   funding-sources



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
