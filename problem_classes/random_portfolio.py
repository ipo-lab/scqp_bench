import numpy as np
import scipy.sparse as spa
from problem_classes.random_qp import generate_random_A


class SparseRandomPortfolio(object):
    '''
    Portfolio QP example
    '''
    def __init__(self, k, seed=1, n=None, n_factor=5, is_dense=False):
        '''
        Generate problem in QP format and CVXPY format
        '''
        # Set random seed
        np.random.seed(seed)

        self.is_dense = is_dense
        self.k = int(k)               # Number of factors
        if n is None:                 # Number of assets
            self.n = int(k * n_factor)
        else:
            self.n = int(n)

        # Generate data
        self.F = spa.random(self.n, self.k, density=0.5,
                            data_rvs=np.random.randn, format='csc')
        self.D = spa.diags(np.random.rand(self.n) *
                           np.sqrt(self.k), format='csc')
        self.mu = np.random.randn(self.n)
        self.gamma = 1.0

        self.osqp_problem = self._generate_osqp_problem()
        self.scqp_problem = self._generate_scqp_problem()

    @staticmethod
    def name():
        return 'Portfolio'

    def _generate_osqp_problem(self):
        '''
        Generate QP problem
        '''

        # Construct the problem
        #       minimize	x' D x + y' I y - (1/gamma) * mu' x
        #       subject to  1' x = 1
        #                   F' x = y
        #                   0 <= x <= 1
        P = spa.block_diag((2 * self.D, 2 * spa.eye(self.k)), format='csc')
        q = np.append(- self.mu / self.gamma, np.zeros(self.k))
        A = spa.vstack([
                spa.hstack([spa.csc_matrix(np.ones((1, self.n))),
                           spa.csc_matrix((1, self.k))]),
                spa.hstack([self.F.T, -spa.eye(self.k)]),
                spa.hstack((spa.eye(self.n), spa.csc_matrix((self.n, self.k))))
            ]).tocsc()
        l = np.hstack([1., np.zeros(self.k), np.zeros(self.n)])
        u = np.hstack([1., np.zeros(self.k), np.ones(self.n)])

        # Constraints without bounds
        A_nobounds = spa.vstack([
                spa.hstack([spa.csc_matrix(np.ones((1, self.n))),
                            spa.csc_matrix((1, self.k))]),
                spa.hstack([self.F.T, -spa.eye(self.k)]),
                ]).tocsc()
        l_nobounds = np.hstack([1., np.zeros(self.k)])
        u_nobounds = np.hstack([1., np.zeros(self.k)])
        bounds_idx = np.arange(self.n)

        # Separate bounds
        lx = np.hstack([np.zeros(self.n), -np.inf * np.ones(self.k)])
        ux = np.hstack([np.ones(self.n), np.inf * np.ones(self.k)])

        problem = {}
        problem['P'] = P
        problem['q'] = q
        problem['A'] = A
        problem['l'] = l
        problem['u'] = u
        problem['m'] = A.shape[0]
        problem['n'] = A.shape[1]

        problem['A_nobounds'] = A_nobounds
        problem['l_nobounds'] = l_nobounds
        problem['u_nobounds'] = u_nobounds
        problem['bounds_idx'] = bounds_idx
        problem['lx'] = lx
        problem['ux'] = ux

        return problem

    def _generate_scqp_problem(self):
        '''
        Generate QP dual problem
        '''
        if not hasattr(self, 'osqp_problem'):
            problem_0 = self._generate_scqp_problem()
        else:
            problem_0 = self.osqp_problem

        problem = {}
        problem['Q'] = problem_0['P']
        problem['p'] = problem_0['q']
        problem['A'] = problem_0['A']
        problem['lb'] = problem_0['l']
        problem['ub'] = problem_0['u']

        return problem

    def update_parameters(self, mu, F=None, D=None):
        """
        Update problem parameters with new mu, F, D
        """

        # Update internal parameters
        self.mu = mu
        if F is not None:
            if F.shape == self.F.shape and \
                    all(F.indptr == self.F.indptr) and \
                    all(F.indices == self.F.indices):
                # Check if F has same sparsity pattern as self.D
                self.F = F
            else:
                raise ValueError("F sparsity pattern changed")
        if D is not None:
            if D.shape == self.D.shape and \
                    all(D.indptr == self.D.indptr) and \
                    all(D.indices == self.D.indices):
                # Check if D has same sparsity pattern as self.D
                self.D = D
            else:
                raise ValueError("D sparsity pattern changed")

        # Update parameters in QP problem
        if F is None and D is None:
            # Update only q
            self.osqp_problem['q'] = np.append(- self.mu / self.gamma, np.zeros(self.k))
            # Update parameter in SCQP
            self.scqp_problem['p'] = self.osqp_problem['q']
        else:
            # Generate problem from scratch
            self.osqp_problem = self._generate_osqp_problem()
            self.scqp_problem = self._generate_scqp_problem()

        return None

    def update_problem(self, alpha=0.1, update_mu_only=False):
        new_mu = alpha * np.random.randn(self.n) + (1 - alpha) * self.mu
        if update_mu_only:
            new_F = None
            new_D = None
        else:
            new_F = self.F.copy()
            new_F.data = alpha * np.random.randn(self.F.nnz) + (1 - alpha) * self.F.data
            new_D = self.D.copy()
            new_D.data = alpha * np.random.rand(self.n) * np.sqrt(self.k) + (1 - alpha) * self.D.data
        self.update_parameters(mu=new_mu, F=new_F, D=new_D)


class DenseRandomPortfolio(object):
    '''
    Random QP example
    '''
    def __init__(self, n, seed=0):
        '''
        Generate problem in QP format and CVXPY format
        '''
        # Set random seed
        np.random.seed(seed)

        # Generate problem data
        self.n = int(n)

        Q, p, A, lb, ub = generate_dense_portfolio_data(n=n, seed=seed)

        self.P = Q
        self.q = p
        self.A = A
        self.l = lb
        self.u = ub

        self.osqp_problem = self._generate_osqp_problem()
        self.scqp_problem = self._generate_scqp_problem()

    @staticmethod
    def name():
        return 'Random QP'

    def _generate_osqp_problem(self):
        '''
        Generate OSQP problem
        '''
        problem = {}
        problem['P'] = spa.csc_matrix(self.P)
        problem['q'] = self.q
        problem['A'] = spa.csc_matrix(self.A)
        problem['l'] = self.l
        problem['u'] = self.u

        return problem

    def _generate_scqp_problem(self):
        '''
        Generate SCQP problem
        '''
        problem = {}
        problem['Q'] = self.P
        problem['p'] = self.q
        problem['A'] = self.A
        problem['lb'] = self.l
        problem['ub'] = self.u

        return problem

    def update_problem(self, alpha=0.1, update_mu_only=False):
        Q_new, p_new, _, _, _ = generate_dense_portfolio_data(n=self.n)

        self.q = alpha * p_new + (1 - alpha) * self.q
        if not update_mu_only:
            self.P = alpha * Q_new + (1-alpha) * self.P

        self.osqp_problem = self._generate_osqp_problem()
        self.scqp_problem = self._generate_scqp_problem()

        return None


def generate_dense_portfolio_data(n=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # --- prep covariance data:
    Sigma = np.random.normal(size=(2*n, n))
    Q = np.matmul(Sigma.T, Sigma) / Sigma.shape[0]

    # --- mu:
    p = np.random.normal(size=n)

    # --- constraints:
    A0 = np.sign(generate_random_A(n, n, prob=0.5)) #np.eye(n)
    lb0 = -np.random.uniform(size=(A0.shape[0])) #np.zeros(n)
    ub0 = np.random.uniform(size=(A0.shape[0])) #np.zeros(n)
    A = np.concatenate((np.ones((1, n)), A0))
    lb = np.concatenate((-np.ones(1), lb0))
    ub = np.concatenate((np.ones(1), ub0))

    return Q, p, A, lb, ub