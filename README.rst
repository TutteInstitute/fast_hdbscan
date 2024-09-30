.. -*- mode: rst -*-

.. image:: doc/hdbscan_logo.png
  :width: 600
  :alt: HDBSCAN logo
  :align: center

======================
Fast Multicore HDBSCAN
======================

The ``fast_hdbscan`` library provides a simple implementation of the HDBSCAN clustering algorithm designed specifically
for high performance on multicore machine with low dimensional data (2D to about 20D). The algorithm runs in parallel and can make
effective use of as many cores as you wish to throw at a problem. It is thus ideal for large SMP systems, and even
modern multicore laptops.

This library provides a
re-implementation of a subset of the HDBSCAN algorithm that is compatible with the
`hdbscan <https://github.com/scikit-learn-contrib/hdbscan>`_ library for data that is Euclidean and
low dimensional. The primary advantages of this library over the standard ``hdbscan`` library are:


 * this library can easily use all available cores to speed up computation;
 * this library has much faster implementations of tree condensing and cluster extraction;
 * this library is much simpler and more approachable for extending or using components from;
 * this library is built on numba and has less issues with binaries and compilation.

This library does not support all the features and input formats available in the hdbscan library.

-----------
Basic Usage
-----------

The ``fast_hdbscan`` library follows the ``hdbscan`` library in using the sklearn API. You can use the ``fast_hdbscan``
class ``HDBSCAN`` exactly as you would that of the ``hdbscan`` library with the caveat that ``fast_hdbscan`` only
supports a subset of the parameters and options of ``hdbscan``. Nonetheless, if you have low-dimensional
Euclidean data (e.g. the output of UMAP), you can use this library as a straightforward drop in replacement for
``hdbscan``:

.. code:: python

    import fast_hdbscan
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(1000)

    clusterer = fast_hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = clusterer.fit_predict(data)

------------
Installation
------------
fast_hdbscan requires:

 * numba
 * numpy
 * scikit-learn

fast_hdbscan can be installed via pip:

.. code:: bash

    pip install fast_hdbscan

To manually install this package:

.. code:: bash

    wget https://github.com/TutteInstitute/fast_hdbscan/archive/main.zip
    unzip main.zip
    rm main.zip
    cd fast_hdbscan-main
    python setup.py install

----------
References
----------

The algorithm used here is an adaptation of the algorithms described in the papers:

    McInnes L, Healy J. *Accelerated Hierarchical Density Based Clustering*
    In: 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp 33-42.
    2017 `[pdf] <http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8215642>`_

    R. Campello, D. Moulavi, and J. Sander, *Density-Based Clustering Based on
    Hierarchical Density Estimates*
    In: Advances in Knowledge Discovery and Data Mining, Springer, pp 160-172.
    2013

-------
License
-------

fast_hdbscan is BSD (2-clause) licensed. See the LICENSE file for details.

------------
Contributing
------------

Contributions are more than welcome! If you have ideas for features of projects please get in touch. Everything from
code to notebooks to examples and documentation are all *equally valuable* so please don't feel you can't contribute.
To contribute please `fork the project <https://github.com/TutteInstitute/fast_hdbscan/issues#fork-destination-box>`_ make your
changes and submit a pull request. We will do our best to work through any issues with you and get your code merged in.
