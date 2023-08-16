# scqp_bench
This repository provides benchmark experiments for the SCQP solver. SCQP is a first-order splitting method for convex quadratic programs. The QP solver is implemented in numpy (for dense QPs) and scipy (for sparse QPs) and invokes a basic implementation of the ADMM algorithm.

## Core Dependencies:
To use the ADMM solver you will need to install [numpy](https://numpy.org),  [Scipy](https://scipy.org) and [qdldl](https://github.com/osqp/qdldl-python).
Please see requirements.txt for full build details.

## Runtime Experiments:
The [demo](demo) directory contains simple demos for using the ADMM solver. Benchmarking runtime experiments are available in the [experiments](experiments) directory. In general, for small scale problems or for large scale sparse problems, OSQP is more efficient. However, for certain large scale dense problems, SCQP can provide improvement in computational efficiency. 

### Random Equality Constrained QP:
Dense QP                   |  Sparse QP   
:-------------------------:|:-------------------------:
![qp_eq_dense](/images/qp_eq_dense.png)  |  ![qp_eq_sparse](/images/qp_eq_sparse.png)

### Random Inequality Constrained QP:

Dense QP: m = n            |  Dense QP: m = 5n        | Dense QP: m = 10n           
:-------------------------:|:-------------------------:|:-------------------------:
![qp_ineq_dense_1](/images/qp_ineq_dense_1.png)  |  ![qp_ineq_dense_5](/images/qp_ineq_dense_5.png)|  ![qp_ineq_dense_10](/images/qp_ineq_dense_10.png)

Sparse QP: m = n           |  Sparse QP: m = 5n        | Sparse QP: m = 10n           
:-------------------------:|:-------------------------:|:-------------------------:
![qp_ineq_sparse_1](/images/qp_ineq_sparse_1.png)  |  ![qp_ineq_sparse_5](/images/qp_ineq_sparse_5.png)|  ![qp_ineq_sparse_10](/images/qp_ineq_sparse_10.png)

### Random LP:
Dense QP                   |  Sparse QP   
:-------------------------:|:-------------------------:
![lp_dense](/images/lp_dense.png)  |  ![lp_sparse](/images/lp_sparse.png)

Computational performance of SCQP vs OSQP for randomly generated QPs and LPS. Stopping tolerance = 1e-5.

### Warm Start Lasso:
Dense Lasso                |  Sparse Lasso   
:-------------------------:|:-------------------------:
![lp_dense](/images/lasso_dense.png)  |  ![lp_sparse](/images/lasso_sparse.png)

Computational performance of SCQP vs OSQP for randomly generated Lasso path building problems. Stopping tolerance = 1e-5.

### Warm Start Portfolio Optimization:
Dense Portfolio            |  Sparse Portfolio  
:-------------------------:|:-------------------------:
![lp_dense](/images/portfolio_dense.png)  |  ![lp_sparse](/images/portfolio_sparse.png)

Computational performance of SCQP vs OSQP for randomly generated portfolio optimization problems. Covariance matrix and return estimate update at each iteration. Stopping tolerance = 1e-5.

### Warm Start Portfolio Optimization:
Dense Portfolio (mu only)  |  Sparse Portfolio (mu only)
:-------------------------:|:-------------------------:
![lp_dense](/images/portfolio_dense_mu.png)  |  ![lp_sparse](/images/portfolio_sparse_mu.png)

Computational performance of SCQP vs OSQP for randomly generated portfolio optimization problems. Only the return estimate (mu) updates at each iteration. Stopping tolerance = 1e-5.
