import numpy as np
import scipy.sparse as spa
from problem_classes.random_qp import generate_random_A


class DenseRandomLP(object):
    '''
    Dense Random QP:
    '''
    def __init__(self, n, m_factor=1, density=0.85, seed=0):
        # Set random seed
        np.random.seed(seed)
        m = round(n * m_factor)

        # Generate problem data
        self.n = int(n)
        self.m = m

        self.Q = np.zeros((n, n))
        self.p = np.random.normal(size=(n, 1))
        self.A = generate_random_A(n=n, m=m, prob=density)
        self.ub = np.random.uniform(size=(self.A.shape[0], 1))
        self.lb = -np.random.uniform(size=(self.A.shape[0], 1))

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
        problem['Q'] = None
        problem['p'] = self.p
        problem['A'] = self.A
        problem['lb'] = self.lb
        problem['ub'] = self.ub

        return problem


class SparseRandomLP(object):
    '''
    Sparse Random LP:
    '''
    def __init__(self, n, m_factor=1, density=0.15, seed=0):
        # Set random seed
        np.random.seed(seed)
        m = round(n * m_factor)

        # Generate problem data
        self.n = int(n)
        self.m = m

        self.Q = np.zeros((n, n))
        self.p = np.random.normal(size=(n, 1))
        A0 = generate_random_A(n=n, m=n, prob=density)
        self.A = np.eye(n) + A0
        self.ub = np.random.uniform(size=(self.A.shape[0], 1))
        self.lb = -np.random.uniform(size=(self.A.shape[0], 1))

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
        problem['Q'] = None
        problem['p'] = self.p
        problem['A'] = spa.csc_matrix(self.A)
        problem['lb'] = self.lb
        problem['ub'] = self.ub

        return problem



