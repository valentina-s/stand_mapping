Dependencies
============

This project is under active development. Hence, the dependencies may change over time.


Documentation Requirements
::

  # Sphinx 1.7.8 has a bug when building on cached envs
  # https://github.com/sphinx-doc/sphinx/issues/5361
  sphinx>=1.3,!=1.7.8
  numpydoc>=0.9
  sphinx-gallery
  sphinx-copybutton
  pytest-runner
  scikit-learn
  matplotlib>=3.0.1
  dask[array]>=0.9.0


Runtime Requirements
::

  numpy>=1.18.1
  scipy>=1.4.1
  matplotlib>=3.1.3
  skimage>=0.16.2
  tensorflow>=2.1.0
  imageio>=2.3.0
  pandas>=1.0.1
