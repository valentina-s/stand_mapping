Making a Stand
==============================

Applying computer vision approaches to instance segmentation to delineate forest
stands and patches using aerial/satellite imagery and lidar data.

Documentation is hosted at [Read the Docs](https://stand-mapping.readthedocs.io/en/latest/).

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- You're reading it
    ├── data
    │   ├── external       <- Data from third party sources
    │   ├── raw            <- Data ready for processing (e.g., unzipped)
    │   ├── interim        <- Intermediate data that has been transformed
    │   └── processed      <- The final, canonical data sets for modeling
    │
    ├── docs               <- Documentation using Sphinx and Read the Docs
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         underscore, then a short `-` delimited description, e.g.
    │                         `01_initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yml    <- Packages needed to reproduce the computing environment using
    │                         conda package manager
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so stand_mapping can be imported
    └── stand_mapping      <- Source code for use in this project
        ├── __init__.py    <- Makes stand_mapping a Python module
        │
        ├── data           <- Scripts to download or generate data
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │                     predictions
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
