import numpy as np
import pandas as pd
from scqp.control import scqp_control
from scqp.scqp import SCQP
from experiments.utils import dir_create
from problem_classes.random_lp import DenseRandomLP, SparseRandomLP
import time as time
import osqp
import os

# --- problem control:
is_dense = True
eps_abs = eps_rel = 10 ** -5

# --- solver params:
check_solved = 5
max_iters = 100_000

# --- problem size control:
n_sims = 10
n_list = [10, 25, 50, 100, 250, 500, 750, 1000]
m_factor = 1

# --- directory:
if is_dense:
    folder = 'dense'
else:
    folder = 'sparse'

if eps_abs < 10 ** -4:
    folder = folder + '_high_accurate'
else:
    folder = folder + '_low_accurate'
dir_path = os.path.expanduser('~/experiments/scqp/random_lp/') + folder + '/'
dir_create(dir_path)

for n in n_list:
    model_names = ["OSQP", "SCQP"]
    comp_times = np.zeros((n_sims, 2))
    comp_times = pd.DataFrame(comp_times, columns=model_names)

    iter_count = np.zeros((n_sims, 2))
    iter_count = pd.DataFrame(iter_count, columns=model_names)

    is_solved = np.zeros((n_sims, 2))
    is_solved = pd.DataFrame(is_solved, columns=model_names)

    for i in range(n_sims):
        # --- generate data:
        if is_dense:
            problem = DenseRandomLP(n=n, m_factor=m_factor, seed=i)
        else:
            problem = SparseRandomLP(n=n, m_factor=m_factor, seed=i)

        # --- Create SCQP:
        control_scqp = scqp_control(eps_abs=eps_abs, eps_rel=eps_rel, verbose=True, check_solved=check_solved,
                                    warm_start=False, max_iters=max_iters, sigma=1e-12)
        model_scqp = SCQP(Q=problem.scqp_problem['Q'], p=problem.scqp_problem['p'],
                          A=problem.scqp_problem['A'], lb=problem.scqp_problem['lb'],
                          ub=problem.scqp_problem['ub'], control=control_scqp)

        # --- Create OSQP:
        control_osqp = {"eps_abs": eps_abs, "eps_rel": eps_rel, "scaled_termination": False,
                        "check_termination": check_solved,
                        "max_iter": max_iters}
        model_osqp = osqp.OSQP()
        model_osqp.setup(P=problem.osqp_problem['P'], q=problem.osqp_problem['q'], A=problem.osqp_problem['A'],
                         l=problem.osqp_problem['l'], u=problem.osqp_problem['u'], **control_osqp)

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
        comp_times["OSQP"][i] = time_osqp
        comp_times["SCQP"][i] = time_scqp
        iter_count["OSQP"][i] = iter_count_osqp
        iter_count["SCQP"][i] = iter_count_scqp
        is_solved["OSQP"][i] = is_solved_osqp
        is_solved["SCQP"][i] = is_solved_scqp

    # --- write data:
    file_comp_time = dir_path + 'comp_time_n_{:d}'.format(n) + '_m_{:d}'.format(round(m_factor)) + '.csv'
    file_iter_count = dir_path + 'iter_count_n_{:d}'.format(n) + '_m_{:d}'.format(round(m_factor)) + '.csv'
    file_is_solved = dir_path + 'solved_status_n_{:d}'.format(n) + '_m_{:d}'.format(round(m_factor)) + '.csv'
    comp_times.to_csv(file_comp_time, index=False)
    iter_count.to_csv(file_iter_count, index=False)
    is_solved.to_csv(file_is_solved, index=False)


