import pyomo
from pathlib import Path

from pycge.examples.splcge_model_def import SplModelDef
from pycge.pycge import PyCGE
from pyomo.environ import check_optimal_termination, value
from pyomo.opt import SolverFactory


cge = PyCGE(SplModelDef())
cge.model_data(str(Path(r"swindon\sam_input")))

import sys
import shutil
from pyomo.environ import SolverFactory

print("IPOPT on PATH:", shutil.which("ipopt"))
print("Pyomo sees IPOPT:", SolverFactory("ipopt").available(exception_flag=False))

cge.model_calibrate("ipopt")

# Fix the price of capital at its initialized value of 1.
cge.model_instance("pf", "CAP")
cge.model_calibrate("ipopt")
if not check_optimal_termination(cge.base_results):
    raise RuntimeError("Benchmark solve did not terminate optimally")

# Start a scenario from the calibrated benchmark.
cge.model_sim()
labour = value(cge.base.FF["LAB"])
cge.model_modify_sim("FF", "LAB", 1.10 * labour)
cge.model_solve("ipopt")
#if not check_optimal_termination(cge.sim_results):
    #raise RuntimeError("Scenario solve did not terminate optimally")

# Z is production output in SplModelDef.
base_output = value(cge.base.Z["BRD"])
scenario_output = value(cge.sim.Z["BRD"])
print(f"Bread output: {base_output:.2f} → {scenario_output:.2f}")
print(f"Change: {100 * (scenario_output / base_output - 1):.2f}%")