from math import isclose
import pyomo.environ as pyo
from pycge.examples.splcge_model_def import SplModelDef
from pycge.pycge import PyCGE

# Build the benchmark and fix the capital price as the numeraire.
cge = PyCGE(SplModelDef())
file_name = r"swindon\sam_input"
cge.model_data(file_name)
cge.model_instance("pf", "CAP")

base = cge.base
print(base.i)
sector = next(iter(base.i))
base.eqpx[sector].deactivate()

# Solve the benchmark.
cge.model_calibrate("ipopt")

# Increase the labour endowment by 10% and solve the scenario.
cge.model_sim()
labour = pyo.value(base.FF["LAB"])
cge.model_modify_sim("FF", "LAB", 1.10 * labour)
cge.model_solve("ipopt")

# Verify that the omitted goods market clears in both solutions.
for label, model in [("Benchmark", base), ("Scenario", cge.sim)]:
    demand = pyo.value(model.X[sector])
    supply = pyo.value(model.Z[sector])
    print(f"{label} {sector} market residual: {demand - supply:.3g}")


base_output = pyo.value(base.Z[sector])
scenario_output = pyo.value(cge.sim.Z[sector])
print(f"{sector} output: {base_output:.2f} → {scenario_output:.2f}")
if base_output != 0:
    print(f"Change: {100 * (scenario_output / base_output - 1):.2f}%")