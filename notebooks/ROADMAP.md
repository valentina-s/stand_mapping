Roadmap for Stand Segmentation
==============================

## Overview
There are two general approaches we will pursue for automating the delineation of forest stands (or contiguous forested Areas of Interest (AOIs) in general).

Both workflows will ingest a multiband array or stack of rasters including aerial imagery (e.g., NAIP) and one or more lidar-derived raster layers (e.g., canopy height model). They will output a raster layer where each pixel is assigned to a distinct contiguous region (e.g., a "forest stand"). This is commonly known as *instance segmentation* in computer vision applications.


### Option A. Two-Stage: Superpixel & Merge
A similar example to this workflow can be found [here](http://emapr.ceoas.oregonstate.edu/pages/education/how_to/image_segmentation/how_to_spatial_segmentation.html).

1. Forested scenes are initially segmented using [watershed segmentation](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.watershed) of a canopy height raster, producing superpixels representing Tree-Associated Objects (TAOs).  Each pixel within a TAO/superpixel is assigned the average value of pixels within that TAO/superpixel. TAOs/superpixels now appear internally homogeneous.

2. TAOs/superpixels are merged into larger regions through on or more steps. Several options for clustering may be considered, including graph-based methods in `skimage` such as [hierarchical merging](https://scikit-image.org/docs/dev/api/skimage.future.graph.html#skimage.future.graph.merge_hierarchical), [Felzenswalb's segmentation](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb), and [normalized cuts](https://scikit-image.org/docs/dev/api/skimage.future.graph.html#skimage.future.graph.ncut) as well as a variety of clustering approaches provided by [`sklearn`](https://scikit-learn.org/stable/modules/clustering.html#clustering).

### Option B. Single-Stage: Neural Network Implementation of Mask-RCNN  
Mask R-CNN is a region-based convolutional neural network designed for object instance segmentation ([He et al., 2018](https://arxiv.org/pdf/1703.06870)). Open-source implementations are available in [PyTorch] (https://github.com/facebookresearch/detectron2) and [Keras + TensorFlow](https://github.com/matterport/Mask_RCNN), among other packages.

![](https://www.learnopencv.com/wp-content/uploads/2019/06/mask-rcnn-1024x477.jpg)

Mask-RCNN embeds object detection, semantic segmentation, and instance segmentation into a single framework. To train this model, a large set of training data will be prepared, comprised of thousands of image "chips" for both input data and labeled targets.  

* Input data is formatted as a multi-channel array (e.g., 1024 rows and 1024 columns with red, green, blue, near-infrared, canopy height, etc...). These data may be stored natively in geo-referenced formats (e.g., GeoTiff, but will likely be preprocessed and fed to the network as non-spatial images/arrays).

* The targets/labels the model learns from are rasterized forest stand delineations. An example stand delineation from the Oregon Garden is shown here. ![](https://oregonforests.org/sites/default/files/inline-images/OregonGarden_StandMap_Public.jpg)  Stand delineations will be converted from vector to raster format such that each polygon in an image "chip" is assigned a unique integer.

## Data Provenance, Storage, and Processing
### Data Sources
Forest stand delineations have been gathered from the US Forest Service, Bureau of Land Management, Washington Department of Natural Resources (DNR), and Oregon Department of Forestry and cover a diverse spectrum of forest conditions across the Pacific Northwest. These forest stands are typically saved in shapefile format.

Historical aerial imagery (NAIP) will be acquired from any of a number of publicly-available sources ([such as this one](https://nrcs.app.box.com/v/naip)).

Raw lidar point cloud data have been acquired from [NOAA Digital Coast](https://coast.noaa.gov/htdata/lidar1_z/), the Washington DNR [Lidar Portal](https://lidarportal.dnr.wa.gov/), and the [Puget Sound Lidar Consortium](https://pugetsoundlidar.ess.washington.edu/). These point clouds are processed through a [pipeline of command-line tools](https://github.com/Ecotrust/pyFIRS) now deployed on Microsoft Azure cloud instances supported by a Microsoft AI For Earth Grant. This pipeline is producing dozens of lidar-derived raster tiles covering millions of acres across Oregon and Washington.

### Data Storage
Lidar-derived rasters are currently being stored on an Azure FileShare. To facilitate collaboration with the project team and use of Google Colab cloud computing instances, working copies of lidar-derived rasters, stand boundaries, and aerial imagery will be stored in a Shared Google Drive.

### Cleaning Stand Delineations
Many stand delineations contain artifacts of regulatory constraints and human error in hand-drawn boundaries between forest types  traditionally developed from aerial imagery. These boundaries may not offer high-accuracy delineations between forest types that can be seen in aerial imagery or from lidar-derived data and could produce noisy edges that reduce the effectiveness of the stand boundaries for training the Mask-RCNN model or for use in quantifying goodness-of-it of newly-proposed delineations.

To adjust raw stand delineations, a series of pre-processing steps may be taken, including:
* Eroding or inwardly-buffering stand boundaries followed by [random walker segmentation](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_random_walker_segmentation.html) where the core areas of stands are used as seeds for edges between stands to be redrawn more precisely.
* Calculating gradients and utilizing edge-detection on the imagery + lidar followed by a method such as [active contours/snakes](https://scikit-image.org/docs/dev/api/skimage.segmentation.html?highlight=image%20segmentation#skimage.segmentation.morphological_chan_vese) that adjust initial stand boundaries to better snap to edges between forest types.

### Preparing Training Chips
Depending on the formats required to quickly feed image data to the Mask-RCNN model, geospatial data commonly in raster formats as GeoTiff may be processed into non-spatial formats such as png or other format if necessary. Raw aerial and lidar imagery that is often 1m or higher resolution may be resampled to coarser resolution to allow for a balance between image size/computational cost and geographic extent. The `Rasterio` and `GDAL` packages will be employed to produce these images. For each image/lidar chip, a corresponding target/label chip will be prepared using `Rasterio` and/or `GDAL` to convert the stand polygons from vector format (e.g., shapefile) into raster/image format ([example code using `GeoPandas` and `Rasterio` here](https://gis.stackexchange.com/a/151861/122267)).

## Executing Segmentation
The two-staged approach ("option A") will be implemented using `skimage` and can probably be implemented using the JupyterHub provided for this class. Fitting the Mask-RCNN model, however, will require a much larger computational and memory capacity, including important gains in processing efficiency with access to a GPU. We will attempt to implement the Mask-RCNN workflow using Google Colab.  

## Evaluating Performance
The performance of automated stand delineation approaches will be quantified using a scoring function such as Intersection over Union (IoU) compared to ground-truth stand delineations. If time allows, we may explore options to tune hyperparameters of models used in the two-stage segmentation workflow using `sklearn` cross-validation or `skopt` Bayesian search routines.

The automated stand delineations will also be shown to colleagues in the UW School of Environmental and Forest Sciences for qualitative characterizations and preference ratings.
