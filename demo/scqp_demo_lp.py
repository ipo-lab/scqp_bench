from scqp.control import scqp_control
from scqp.scqp import SCQP
from problem_classes.random_lp import DenseRandomLP, SparseRandomLP
import time as time
import osqp
import matplotlib.pyplot as plt

# --- problem setup:
n = 100
is_eq = False
seed = 0
# --- solver setup:
eps_abs = eps_rel = 10**-5
max_iters = 10_000
check_solved = 10

# --- generate data:
#problem = DenseRandomLP(n=n, seed=seed)
problem = SparseRandomLP(n=n, seed=seed)
# --- Create SCQP object:
control_scqp = scqp_control(eps_abs=eps_abs, eps_rel=eps_rel, verbose=True, check_solved=check_solved,
                            warm_start=False, max_iters=max_iters)
model_scqp = SCQP(Q=problem.scqp_problem['Q'], p=problem.scqp_problem['p'],
                  A=problem.scqp_problem['A'], lb=problem.scqp_problem['lb'],
                  ub=problem.scqp_problem['ub'], control=control_scqp)

# --- Create OSQP object:
control_osqp = {"eps_abs": eps_abs,"eps_rel": eps_rel, "scaled_termination":False, "check_termination":check_solved,
                "max_iter": max_iters}
model_osqp = osqp.OSQP()
model_osqp.setup(P=problem.osqp_problem['P'], q=problem.osqp_problem['q'], A=problem.osqp_problem['A'],
                 l=problem.osqp_problem['l'], u=problem.osqp_problem['u'], **control_osqp)

# --- unpack problem data:
Q = problem.Q
p = problem.p
A = problem.A
lb = problem.lb
ub = problem.ub

# --- clock solve time OSQP:
start = time.time()
sol_osqp = model_osqp.solve()
time_osqp = time.time() - start
x_osqp = sol_osqp.x
y_osqp = sol_osqp.y
z_osqp = A.dot(x_osqp)
iter_count_osqp = sol_osqp.info.iter
is_solved_osqp = int(sol_osqp.info.status == 'solved')
print('computation time: {:f}'.format(time_osqp))

# --- clock solve time:
start = time.time()
sol_scqp = model_scqp.solve()
time_scqp = time.time() - start
x_scqp = model_scqp.sol.get('x')
y_scqp = model_scqp.sol.get('y')
z_scqp = model_scqp.sol.get('z')
iter_count_scqp = model_scqp.sol.get('iter')
is_solved_scqp = model_scqp.sol.get('status')
print('computation time: {:f}'.format(time_scqp))

# --- primal solution:
plt.plot(x_osqp)
plt.plot(x_scqp)

# --- dual solution
plt.plot(y_osqp)
plt.plot(y_scqp)

# --- objective value
obj_osqp = model_scqp.compute_objective(x=x_osqp, Q=Q, p=p)
obj_scqp = model_scqp.compute_objective(x=x_scqp, Q=Q, p=p)
print('OSQP objective: {:f}'.format(obj_osqp))
print('SCQP objective: {:f}'.format(obj_scqp))

# --- primal residual:
primal_tol_osqp = model_scqp.compute_primal_tol(x=x_osqp, z=z_osqp, A=A)
primal_tol_scqp = model_scqp.compute_primal_tol(x=x_scqp, z=z_scqp, A=A)
primal_error_osqp = model_scqp.compute_primal_error(x=x_osqp, A=A, lb=lb, ub=ub)
primal_error_scqp = model_scqp.compute_primal_error(x=x_scqp, A=A, lb=lb, ub=ub)

print('OSQP primal tolerance: {:f}'.format(primal_tol_osqp))
print('OSQP primal error: {:f}'.format(primal_error_osqp))

print('SCQP primal tolerance: {:f}'.format(primal_tol_scqp))
print('SCQP primal error: {:f}'.format(primal_error_scqp))
# --- dual_residual
dual_tol_osqp = model_scqp.compute_dual_tol(x=x_osqp, y=y_osqp,  Q=Q, p=p, A=A)
dual_tol_scqp = model_scqp.compute_dual_tol(x=x_scqp, y=y_scqp,  Q=Q, p=p, A=A)
dual_error_osqp = model_scqp.compute_dual_error(x=x_osqp, y=y_osqp,  Q=Q, p=p, A=A)
dual_error_scqp = model_scqp.compute_dual_error(x=x_scqp, y=y_scqp,  Q=Q, p=p, A=A)

print('OSQP dual tolerance: {:f}'.format(dual_tol_osqp))
print('OSQP dual error: {:f}'.format(dual_error_osqp))

print('SCQP dual tolerance: {:f}'.format(dual_tol_scqp))
print('SCQP dual error: {:f}'.format(dual_error_scqp))

