import matplotlib.pyplot as plt
import os
from plot.utils import plot_profile


# --- preambles:
out_dir = os.path.expanduser('~/workspace/scqp_bench/images/')
# --- Random Equality QP
plot_profile(folder='~/experiments/scqp/random_qp/eq_dense_high_accurate/')
plt.savefig(out_dir+'qp_eq_dense.png')
plot_profile(folder='~/experiments/scqp/random_qp/eq_sparse_high_accurate/')
plt.savefig(out_dir+'qp_eq_sparse.png')

# --- Random Inequality QP
plot_profile(folder='~/experiments/scqp/random_qp/ineq_dense_high_accurate/', token='m_1.csv')
plt.savefig(out_dir+'qp_ineq_dense_1.png')
plot_profile(folder='~/experiments/scqp/random_qp/ineq_dense_high_accurate/', token='m_5.csv')
plt.savefig(out_dir+'qp_ineq_dense_5.png')
plot_profile(folder='~/experiments/scqp/random_qp/ineq_dense_high_accurate/', token='m_10.csv')
plt.savefig(out_dir+'qp_ineq_dense_10.png')

plot_profile(folder='~/experiments/scqp/random_qp/ineq_sparse_high_accurate/', token='m_1.csv')
plt.savefig(out_dir+'qp_ineq_sparse_1.png')
plot_profile(folder='~/experiments/scqp/random_qp/ineq_sparse_high_accurate/', token='m_5.csv')
plt.savefig(out_dir+'qp_ineq_sparse_5.png')
plot_profile(folder='~/experiments/scqp/random_qp/ineq_sparse_high_accurate/', token='m_10.csv')
plt.savefig(out_dir+'qp_ineq_sparse_10.png')


# --- Random LP
plot_profile(folder='~/experiments/scqp/random_lp/dense_high_accurate/')
plt.savefig(out_dir+'lp_dense.png')
plot_profile(folder='~/experiments/scqp/random_lp/sparse_high_accurate/')
plt.savefig(out_dir+'lp_sparse.png')

# --- Random Lasso
plot_profile(folder='~/experiments/scqp/random_lasso/dense/')
plt.savefig(out_dir+'lasso_dense.png')
plot_profile(folder='~/experiments/scqp/random_lasso/sparse/')
plt.savefig(out_dir+'lasso_sparse.png')


# --- Random portfolio
plot_profile(folder='~/experiments/scqp/random_portfolio/dense/')
plt.savefig(out_dir+'portfolio_dense.png')
plot_profile(folder='~/experiments/scqp/random_portfolio/dense_mu_only/')
plt.savefig(out_dir+'portfolio_dense_mu.png')
plot_profile(folder='~/experiments/scqp/random_portfolio/sparse/')
plt.savefig(out_dir+'portfolio_sparse.png')
plot_profile(folder='~/experiments/scqp/random_portfolio/sparse_mu_only/')
plt.savefig(out_dir+'portfolio_sparse_mu.png')