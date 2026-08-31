"""Prepare a gurobipy-free copy of the ASSENT dataset so it can be loaded in the GPU venv
(where gurobipy's DLL is blocked by Windows Application Control). The only gurobipy dependency
when loading the data is that the pickled labels are `optimization_utils.Solution` objects; we
convert each to a plain types.SimpleNamespace with the same attributes (x, tau, s, y_tx, y_rx).

Run in the CPU/global env (where gurobipy works):
  python experiments/prep_data.py
"""
import os, sys, types, pickle
from types import SimpleNamespace
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import paths  # noqa: F401
import numpy as np

# Stub out src.utils.optimization_utils so unpickling the dataset's Solution labels does NOT import
# gurobipy (whose DLL is blocked by Windows Application Control). The Solution/ProblemParams classes
# themselves need no gurobipy; pickle reconstructs them via __new__ + __dict__, so bare classes work.
_opt = types.ModuleType("src.utils.optimization_utils")
class Solution:  # noqa: E701
    pass
class ProblemParams:  # noqa: E701
    pass
_opt.Solution = Solution
_opt.ProblemParams = ProblemParams
sys.modules["src.utils.optimization_utils"] = _opt

os.chdir(paths.UPSTREAM)
from src.learning_v4.assent_data_utils_v4 import data_loader

res = data_loader(nsamps=10000, CONSOLE_RUN=True)
res["solution"] = [SimpleNamespace(x=np.asarray(s.x), tau=np.asarray(s.tau), s=np.asarray(s.s),
                                   y_tx=np.asarray(s.y_tx), y_rx=np.asarray(s.y_rx))
                   for s in res["solution"]]
out = os.path.join(paths.RESULTS, "assent_data_plain.pkl")
pickle.dump(res, open(out, "wb"), protocol=4)
print(f"saved {out}: {len(res['solution'])} samples, gurobipy-free")
