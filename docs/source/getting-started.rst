Getting started
===============

This project is under active development. Source code is not yet hosted on a
package index such as PyPI or conda-forge. In the meantime, the best way to get
up-and-running is to first clone this repo_ to your local machine then navigate
into it.
::

  git clone https://github.com/d-diaz/stand_mapping.git
  cd stand_mapping

For a simple install, you can just do
::

  python setup.py install

If you want to make any changes to the source code and have them updated while
you work, consider any of the following methods.
::

  python setup.py develop

If you want to use the :code:`pip` package manager
::

  pip install . -e

If you use the :code:`conda` package manager and have the :code:`conda-build`
package installed, you can also use
::

  conda develop .


Once that is done, you should be able to import any functions or modules from
:code:`stand_mapping`...
::

  >>> from stand_mapping.data.fetch import naip_from_tnm
  >>> XMIN, YMIN = 555750, 5266200
  >>> WIDTH, HEIGHT = 500, 500
  >>> BBOX = (XMIN, YMIN, XMIN+WIDTH, YMIN+HEIGHT)
  >>> img = get_naip_from_tnm(BBOX, res=1, inSR=6339)
  >>> img.shape
  (500,500,4)

.. _repo: https://github.com/d-diaz/stand_mapping
