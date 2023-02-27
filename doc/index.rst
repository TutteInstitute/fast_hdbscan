.. fast_hdbscan documentation master file, created by
   sphinx-quickstart on Wed Feb 22 18:39:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fast_hdbscan's documentation!
========================================

The fast_hdbscan library provides a simple implementation of the HDBSCAN clustering algorithm designed specifically
for high performance on multicore machine with low dimensional data. The algorithm runs in parallel and can make
effective use of as many cores as you wish to throw at a problem. It is thus ideal for large SMP systems, and even
modern multicore laptops.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   basic_usage
   benchmarks
   comparable_clusterings



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
