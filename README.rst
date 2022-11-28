Benchmark repository for Sparse Logistic Regression
===================================================

|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms. This benchmark tests algorithms to solve the following problem:


$$\\min_w \\sum_i \\log(1 + \\exp(-y_i x_i^\\top w)) + \\lambda \\lVert w\\rVert_1$$

where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features, and

$$y \\in \\mathbb{R}^n, X = [x_1^\\top, \\dots, x_n^\\top]^\\top \\in \\mathbb{R}^{n \\times p}$$


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_logreg_l1
   $ benchopt run benchmark_logreg_l1

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

   $ benchopt run benchmark_logreg_l1 -s sklearn -d boston --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.


.. |Build Status| image:: https://github.com/benchopt/benchmark_logreg_l1/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_logreg_l1/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
