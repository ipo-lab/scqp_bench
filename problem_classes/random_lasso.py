import numpy as np
import scipy.sparse as spa


class DenseRandomLasso(object):
    '''
    Dense Lasso
    '''
    def __init__(self, n, m_factor=100, density=0.85, seed=1):
        '''
        Generate problem in OSQP format and SCQP format
        '''
        # Set random seed
        np.random.seed(seed)

        self.n = int(n)               # Number of features
        self.m = int(self.n * m_factor)    # Number of data-points

        self.Ad = spa.random(self.m, self.n, density=density,
                             data_rvs=np.random.randn)
        self.A = self.Ad.toarray()
        self.ATA = self.A.T.dot(self.A)
        self.ATA_inv = np.linalg.inv(self.ATA)

        self.x_true = np.multiply((np.random.rand(self.n) >
                                   0.5).astype(float),
                                  np.random.randn(self.n)) / np.sqrt(self.n)
        self.bd = self.Ad.dot(self.x_true) + np.random.randn(self.m)
        self.lambda_max = np.linalg.norm(self.Ad.T.dot(self.bd), np.inf)
        self.lambda_param = (1./5.) * self.lambda_max

        self.osqp_problem = self._generate_osqp_problem()
        self.scqp_problem = self._generate_scqp_problem()

    @staticmethod
    def name():
        return 'Lasso'

    def _generate_scqp_problem(self):
        '''
        Generate QP dual problem
        '''

        problem = {}
        problem['Q'] = self.ATA_inv
        problem['p'] = - self.ATA_inv.dot(self.A.T.dot(self.bd))
        problem['A'] = np.eye(self.n)
        problem['lb'] = -np.ones(self.n)*self.lambda_param
        problem['ub'] = np.ones(self.n)*self.lambda_param

        return problem

    def _generate_osqp_problem(self):
        problem_0 = self._generate_scqp_problem()
        problem = {}
        problem['P'] = spa.csc_matrix(problem_0['Q'])
        problem['q'] = problem_0['p']
        problem['A'] = spa.csc_matrix(problem_0['A'])
        problem['l'] = problem_0['lb']
        problem['u'] = problem_0['ub']
        return problem

    def update_lambda(self, lambda_new):
        """
        Update lambda value in inner problems
        """
        # Update internal lambda parameter
        self.lambda_param = lambda_new

        # Update lb and ub in dual problem
        self.scqp_problem['lb'] = -np.ones(self.n) * self.lambda_param
        self.scqp_problem['ub'] = np.ones(self.n) * self.lambda_param
        self.osqp_problem['l'] = -np.ones(self.n) * self.lambda_param
        self.osqp_problem['u'] = np.ones(self.n) * self.lambda_param

    def dual_to_primal(self, v):
        x = self.A.T.dot(self.bd) - v
        x = self.ATA_inv.dot(x)
        return x


class SparseRandomLasso(object):
    '''
    Lasso QP example
    '''
    def __init__(self, n, m_factor=100, density=0.15, seed=1):
        '''
        Generate problem in OSQP format and SCQP format
        '''
        # Set random seed
        np.random.seed(seed)

        self.n = int(n)               # Number of features
        self.m = int(self.n * m_factor)    # Number of data-points

        self.Ad = spa.random(self.m, self.n, density=density,
                             data_rvs=np.random.randn)

        self.x_true = np.multiply((np.random.rand(self.n) >
                                   0.5).astype(float),
                                  np.random.randn(self.n)) / np.sqrt(self.n)
        self.bd = self.Ad.dot(self.x_true) + np.random.randn(self.m)
        self.lambda_max = np.linalg.norm(self.Ad.T.dot(self.bd), np.inf)
        self.lambda_param = (1./5.) * self.lambda_max

        self.osqp_problem = self._generate_osqp_problem()
        self.scqp_problem = self._generate_scqp_problem()

    @staticmethod
    def name():
        return 'Lasso'

    def _generate_osqp_problem(self):
        '''
        Generate QP problem
        '''

        # Construct the problem
        #       minimize	y' * y + lambda * 1' * t
        #       subject to  y = Ax - b
        #                   -t <= x <= t
        P = spa.block_diag((spa.csc_matrix((self.n, self.n)),
                            2*spa.eye(self.m),
                            spa.csc_matrix((self.n, self.n))), format='csc')
        q = np.append(np.zeros(self.m + self.n),
                      self.lambda_param * np.ones(self.n))
        In = spa.eye(self.n)
        Onm = spa.csc_matrix((self.n, self.m))
        A = spa.vstack([spa.hstack([self.Ad, -spa.eye(self.m),
                                    spa.csc_matrix((self.m, self.n))]),
                        spa.hstack([In, Onm, -In]),
                        spa.hstack([In, Onm, In])]).tocsc()
        l = np.hstack([self.bd, -np.inf * np.ones(self.n), np.zeros(self.n)])
        u = np.hstack([self.bd, np.zeros(self.n), np.inf * np.ones(self.n)])

        problem = {}
        problem['P'] = P
        problem['q'] = q
        problem['A'] = A
        problem['l'] = l
        problem['u'] = u
        problem['m'] = A.shape[0]
        problem['n'] = A.shape[1]

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

    def update_lambda(self, lambda_new):
        """
        Update lambda value in inner problems
        """
        # Update internal lambda parameter
        self.lambda_param = lambda_new

        # Update q in OSQP problem
        self.osqp_problem['q'] = np.append(np.zeros(self.m + self.n), self.lambda_param * np.ones(self.n))
        self.scqp_problem['p'] = self.osqp_problem['q']
