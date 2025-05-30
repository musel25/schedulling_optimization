#!/usr/bin/env python

import logging, pyomo.common.log
logging.getLogger('pyomo.solvers').setLevel(logging.ERROR)

"""
Re-entrant flow-shop MILP solved with HiGHS.
Automatically tries:
  1) highspy  (solver_io='python') – fastest, no temp files
  2) highs    (command-line binary)
Falls back to a helpful error if neither is available.
"""

import itertools, time, pyomo.environ as pyo, shutil, importlib

# ---------------------------------------------------------------------------
# Data (same as before) ------------------------------------------------------
jobs, stages = ['J1','J2','J3','J4','J5'], [1,2,3,4,5]
p = {('J1',1):6, ('J1',2):5, ('J1',3):3, ('J1',4):9, ('J1',5):3,
     ('J2',1):7, ('J2',2):6, ('J2',3):2, ('J2',4):8, ('J2',5):2,
     ('J3',1):5, ('J3',2):7, ('J3',3):4, ('J3',4):7, ('J3',5):4,
     ('J4',1):6, ('J4',2):5, ('J4',3):3, ('J4',4):6, ('J4',5):3,
     ('J5',1):8, ('J5',2):6, ('J5',3):2, ('J5',4):9, ('J5',5):2}
phys = {1:1, 2:2, 3:3, 4:4, 5:3}

# ---------------------------------------------------------------------------
# Model ----------------------------------------------------------------------
m = pyo.ConcreteModel()
m.JK   = [(j,k) for j in jobs for k in stages]
m.S    = pyo.Var(m.JK, domain=pyo.NonNegativeReals)
m.Cmax = pyo.Var(domain=pyo.NonNegativeReals)

pairs = []
for mach in set(phys.values()):
    ops = [(j,k) for (j,k) in m.JK if phys[k]==mach]
    pairs += list(itertools.combinations(ops,2))
m.y = pyo.Var(pairs, domain=pyo.Binary)
M = sum(p.values())

def preced(mod,j,k):
    if k==5: return pyo.Constraint.Skip
    return mod.S[j,k+1] >= mod.S[j,k]+p[(j,k)]
m.preced = pyo.Constraint(jobs, stages, rule=preced)

def cap1(mod,j,k,h,l):
    if phys[k]!=phys[l]: return pyo.Constraint.Skip
    return mod.S[j,k]+p[(j,k)] <= mod.S[h,l] + M*(1-mod.y[(j,k),(h,l)])
def cap2(mod,j,k,h,l):
    if phys[k]!=phys[l]: return pyo.Constraint.Skip
    return mod.S[h,l]+p[(h,l)] <= mod.S[j,k] + M*   mod.y[(j,k),(h,l)]
m.cap1 = pyo.Constraint(pairs, rule=cap1)
m.cap2 = pyo.Constraint(pairs, rule=cap2)

m.cmax_def = pyo.Constraint(jobs, rule=lambda mod,j: mod.S[j,5]+p[(j,5)] <= mod.Cmax)
m.obj = pyo.Objective(expr=m.Cmax, sense=pyo.minimize)

# ---------------------------------------------------------------------------
# Solver picker --------------------------------------------------------------
def get_highs_solver():
    """Return a ready-to-use Pyomo solver object for HiGHS."""
    # 1) try python interface (highspy)
    try:
        importlib.import_module("highspy")
        s = pyo.SolverFactory("highs", solver_io="python")
        if s.available(exception_flag=False):
            print("--> using HiGHS via highspy (in-process)")
            return s
    except ModuleNotFoundError:
        pass
    # 2) try command-line executable
    if shutil.which("highs"):
        s = pyo.SolverFactory("highs")        # shell
        if s.available(exception_flag=False):
            print("--> using HiGHS executable")
            return s
    # 3) nothing found
    raise RuntimeError(
        "HiGHS not found.\n"
        "Install with EITHER:\n"
        "  conda install -c conda-forge highs      # command-line binary\n"
        "OR\n"
        "  pip install highspy                     # Python bindings\n")

solver = get_highs_solver()

# ---------------------------------------------------------------------------
# Solve ----------------------------------------------------------------------
t0 = time.perf_counter()
res = solver.solve(m, tee=False)
print(f"\nStatus        : {res.solver.termination_condition}")
print(f"Optimal Cmax  : {pyo.value(m.Cmax):g}")
print(f"Solve time    : {time.perf_counter()-t0:.3f} s")
