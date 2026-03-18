.. fast_hdbscan documentation master file, created by
   sphinx-quickstart on Wed Feb 22 18:39:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: hdbscan_logo.png
  :width: 600
  :alt: HDBSCAN logo
  :align: center

======================
Fast Multicore HDBSCAN
======================

The ``fast_hdbscan`` library provides an implementation of the HDBSCAN clustering algorithm designed specifically
for high performance on multicore machine. The algorithm runs in parallel and can make
effective use of as many cores as you wish to throw at a problem. It is thus ideal for large SMP systems, and even
modern multicore laptops.

This library provides a
re-implementation of a subset of the HDBSCAN algorithm that is compatible with the
`hdbscan <https://github.com/scikit-learn-contrib/hdbscan>`_ library. There are specific optimizationsfor data that is Euclidean and
low dimensional, other distance metrics and high dimensional data fallback to alternative parallel approaches that are faster than the
hdbscan library, but not necessarily as performant as the highly optimized low-dimensional Euclidean case. 
The primary advantages of this library over the standard ``hdbscan`` library are:


 * this library can easily use all available cores to speed up computation;
 * this library has much faster implementations of tree condensing and cluster extraction;
 * this library is much simpler and more approachable for extending or using components from;
 * this library is built on numba and has less issues with binaries and compilation.
 * this library provides features such as semi-supervision, linking constraints, sample weights, and branch detection from FLASC, and an implemntation of PLSCAN.

This library does not support *all* the features and input formats available in the hdbscan library, but covers the most common use cases. 

This library does support a number of research extensions to HDBSCAN including branch detection
from `FLASC <https://peerj.com/articles/cs-2792/>`_ and the semi-supervised clustering methods, 
as well as support for sample weights.

As a bonus this library also provides an easy to use implementation of the
`PLSCAN <https://arxiv.org/abs/2512.16558>`_ algorithm for automated cluster 
resolution selection and layered clustering.

-----------
Basic Usage
-----------

The ``fast_hdbscan`` library follows the ``hdbscan`` library in using the sklearn API. You can use the ``fast_hdbscan``
class ``HDBSCAN`` exactly as you wuld that of the ``hdbscan`` library with the caveat that ``fast_hdbscan`` only
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

if you need more than just Euclidean distance, or support for high dimensional data, you will also need:

 * pynndescent

fast_hdbscan can be installed via pip:

.. code:: bash

    pip install fast_hdbscan


----------
User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   basic_usage
   plscan
   metrics_and_constraints
   benchmarks
   comparable_clusterings
   detecting_branches
   for_developers


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

The branch-detection functionality is adapted from:

    D.M. Bot, J. Peeters, J. Liesenborgs, J. Aerts. 
    *FLASC: a flare-sensitive clustering algorithm.*
    In: PeerJ Computer Science, Volume 11, e2792, 2025.
    https://doi.org/10.7717/peerj-cs.2792.

The PLSCAN functionality is adapted from:

    D.M. Bot, L. McInnes, J. Aerts.
    *Persistent Multiscale Density-based Clustering.*
    In: arXiv preprint arXiv:2512.16558, 2025.
    https://arxiv.org/abs/2512.16558.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
