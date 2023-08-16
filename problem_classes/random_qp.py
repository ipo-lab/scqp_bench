import numpy as np
import scipy.sparse as spa
from copy import deepcopy


class DenseRandomQP(object):
    '''
    Dense Random QP:
    '''
    def __init__(self, n, m_factor=1, density=0.85, is_eq=False, seed=0):
        # Set random seed
        np.random.seed(seed)
        m = round(n * m_factor)

        # Generate problem data
        self.n = int(n)
        self.m = m

        Q, p, A, lb, ub = generate_random_qp(n=n, m=m, prob=density, seed=seed, is_eq=is_eq)

        self.Q = Q
        self.p = p
        self.A = A
        self.lb = lb
        self.ub = ub

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
        problem['P'] = spa.csc_matrix(self.Q)
        problem['q'] = self.p
        problem['A'] = spa.csc_matrix(self.A)
        problem['l'] = self.lb
        problem['u'] = self.ub

        return problem

    def _generate_scqp_problem(self):
        '''
        Generate SCQP problem
        '''
        problem = {}
        problem['Q'] = self.Q
        problem['p'] = self.p
        problem['A'] = self.A
        problem['lb'] = self.lb
        problem['ub'] = self.ub

        return problem


class SparseRandomQP(object):
    '''
    Sparse Random QP:
    '''
    def __init__(self, n, m_factor=1, density=0.15, is_eq=False, seed=0):
        # Set random seed
        np.random.seed(seed)
        m = round(n * m_factor)

        # Generate problem data
        self.n = int(n)
        self.m = m
        self.density = density

        Q, p, A, lb, ub = generate_random_qp(n=n, m=m, prob=density, seed=seed, is_eq=is_eq)

        self.Q = Q
        self.p = p
        self.A = A
        self.lb = lb
        self.ub = ub

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
        problem['P'] = spa.csc_matrix(self.Q)
        problem['q'] = self.p
        problem['A'] = spa.csc_matrix(self.A)
        problem['l'] = self.lb
        problem['u'] = self.ub

        return problem

    def _generate_scqp_problem(self):
        '''
        Generate SCQP problem
        '''
        problem = {}
        if self.density < 0.10:
            problem['Q'] = spa.csc_matrix(self.Q)
            problem['A'] = spa.csc_matrix(self.A)
        else:
            problem['Q'] = self.Q
            problem['A'] = self.A
        problem['p'] = self.p
        problem['lb'] = self.lb
        problem['ub'] = self.ub

        return problem


def generate_random_qp(n, m, prob, seed, is_eq=False):
    np.random.seed(seed)
    M = np.random.normal(size=(n, n))
    M = M * np.random.binomial(1, prob, size=(n, n))
    # --- Q:
    Q = np.dot(M.T, M) + 10 ** -2 * np.eye(n)
    # --- p:
    p = np.random.normal(size=(n, 1))
    # --- A:
    A = generate_random_A(n=n, m=m, prob=prob)
    # --- ub:
    ub = np.random.uniform(size=(A.shape[0], 1))
    # --- lb:
    if is_eq:
        lb = deepcopy(ub)
    else:
        lb = -np.random.uniform(size=(A.shape[0], 1))
    return Q, p, A, lb, ub


def generate_random_A(n, m, prob):
    # --- A:
    A = np.zeros((m, n))
    for i in range(m):
        a = np.random.normal(size=(1, n))
        b = np.zeros(1)
        while b.sum() == 0:
            b = np.random.binomial(1, prob, size=(1, n))
        a = a*b
        A[i, :] = a

    return A


def sparse_spd(n, density=0.15):
    M = spa.random(n, n, density=density, data_rvs=np.random.randn, format='csc')
    M = spa.triu(M)
    M = M + M.T - spa.diags(M.diagonal())
    M = M.T
    d = np.linalg.eigvals(M.toarray()).min()
    d = np.abs(d) + 10**-2
    M = M + spa.diags(np.repeat(d, n))
    return M

