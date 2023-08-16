import numpy as np
import pandas as pd
from scqp.control import scqp_control
from scqp.scqp import SCQP
from experiments.utils import dir_create
from problem_classes.random_lasso import DenseRandomLasso, SparseRandomLasso
import time as time
import osqp
import os


# --- problem control:
eps_abs = eps_rel = 10**-3
check_solved = 10
max_iters = 1_000

# --- problem size control:
is_dense = False
n_sims = 10
n_lams = 100
n_list = [10, 25, 50, 100, 250, 500, 750, 1000]

# --- dir_pathectory:
if is_dense:
    folder = 'dense'
else:
    folder = 'sparse'

dir_path = os.path.expanduser('~/experiments/scqp/random_lasso/') + folder+'/'
dir_create(dir_path)

for i in range(len(n_list)):
    comp_times_list = []
    iter_count_list = []
    is_solved_list = []
    n = n_list[i]
    for sim in range(n_sims):
        #  --- data storage:
        model_names = ["OSQP", "SCQP"]
        comp_times = np.zeros((n_lams, 2))
        comp_times = pd.DataFrame(comp_times, columns=model_names)

        iter_count = np.zeros((n_lams, 2))
        iter_count = pd.DataFrame(iter_count, columns=model_names)

        is_solved = np.zeros((n_lams, 2))
        is_solved = pd.DataFrame(is_solved, columns=model_names)

        # --- create problem:
        if is_dense:
            problem = DenseRandomLasso(n=n, seed=sim)
        else:
            problem = SparseRandomLasso(n=n, seed=sim)

        lambda_array = np.logspace(np.log10(0.01 * problem.lambda_max), np.log10(problem.lambda_param), n_lams)[::-1]
        # --- create solvers:
        # --- Create SCQP:
        control_scqp = scqp_control(eps_abs=eps_abs, eps_rel=eps_rel, verbose=True, check_solved=check_solved,
                                    warm_start=True, max_iters=max_iters, cache_factor=True)
        model_scqp = SCQP(Q=problem.scqp_problem['Q'], p=problem.scqp_problem['p'],
                          A=problem.scqp_problem['A'], lb=problem.scqp_problem['lb'],
                          ub=problem.scqp_problem['ub'], control=control_scqp)

        # --- Create OSQP:
        control_osqp = {"eps_abs": eps_abs, "eps_rel": eps_rel, "scaled_termination": False,
                        "check_termination": check_solved, "max_iter": max_iters, "warm_start": True}
        model_osqp = osqp.OSQP()

        model_osqp.setup(P=problem.osqp_problem['P'], q=problem.osqp_problem['q'],
                         A=problem.osqp_problem['A'], l=problem.osqp_problem['l'],
                         u=problem.osqp_problem['u'], **control_osqp)

        # --- clock solve time OSQP:
        start = time.time()
        sol_osqp = model_osqp.solve()
        time_osqp = time.time() - start
        iter_count_osqp = sol_osqp.info.iter
        is_solved_osqp = int(sol_osqp.info.status == 'solved')
        print('computation time: {:f}'.format(time_osqp))

        # --- clock solve time:
        start = time.time()
        sol_scqp = model_scqp.solve()
        time_scqp = time.time() - start
        iter_count_scqp = model_scqp.sol.get('iter')
        is_solved_scqp = model_scqp.sol.get('status')
        print('computation time: {:f}'.format(time_scqp))

        for k in range(n_lams):
            print('lam iteration: {:d}'.format(k))
            lambda_new = lambda_array[k]

            # --- update problem values:
            problem.update_lambda(lambda_new=lambda_new)
            if is_dense:
                model_osqp.update(l=problem.osqp_problem['l'], u=problem.osqp_problem['u'])
                model_scqp.update(lb=problem.scqp_problem['lb'], ub=problem.scqp_problem['ub'])
            else:
                model_osqp.update(q=problem.osqp_problem['q'])
                model_scqp.update(p=problem.scqp_problem['p'])
            # --- clock solve time OSQP:
            start = time.time()
            sol_osqp = model_osqp.solve()
            time_osqp = time.time() - start
            iter_count_osqp = sol_osqp.info.iter
            is_solved_osqp = int(sol_osqp.info.status == 'solved')
            print('computation time: {:f}'.format(time_osqp))

            # --- clock solve time:
            start = time.time()
            sol_scqp = model_scqp.solve()
            time_scqp = time.time() - start
            iter_count_scqp = model_scqp.sol.get('iter')
            is_solved_scqp = model_scqp.sol.get('status')
            print('computation time: {:f}'.format(time_scqp))

            # --- store data:
            comp_times["OSQP"][k] = time_osqp
            comp_times["SCQP"][k] = time_scqp
            iter_count["OSQP"][k] = iter_count_osqp
            iter_count["SCQP"][k] = iter_count_scqp
            is_solved["OSQP"][k] = is_solved_osqp
            is_solved["SCQP"][k] = is_solved_scqp

        comp_times_list.append(comp_times)
        iter_count_list.append(iter_count)
        is_solved_list.append(is_solved)

    # --- write data:
    comp_times = pd.concat(comp_times_list)
    iter_count = pd.concat(iter_count_list)
    is_solved = pd.concat(is_solved_list)
    file_comp_time = dir_path + 'comp_time_n_{:d}'.format(problem.n) + '.csv'
    file_iter_count = dir_path + 'iter_count_n_{:d}'.format(problem.n) + '.csv'
    file_is_solved = dir_path + 'solved_status_n_{:d}'.format(problem.n) + '.csv'
    comp_times.to_csv(file_comp_time, index=False)
    iter_count.to_csv(file_iter_count, index=False)
    is_solved.to_csv(file_is_solved, index=False)
