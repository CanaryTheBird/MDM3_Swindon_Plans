# Imports from the existing CGE model
import os
import sys
from scipy.integrate import solve_ivp

import pyomo.environ as pyo
from pycge.examples.stdcge_model_def import StdModelDef

# Locate the project folder
project_folder = r"C:\Users\supre\OneDrive\Documents\mdm3\MDM3_Swindon_Plans"

sys.path.insert(0, project_folder)
os.chdir(project_folder)

from swindon.Create_SAM_folder import sam_to_std

# Make IPOPT available to Pyomo
os.environ["PATH"] = (
    r"C:\Users\supre\miniconda3\Library\bin"
    + os.pathsep
    + os.environ["PATH"]
)

# Load the existing CGE model

# Load the new SAM and create the standard CGE model
sam, goods, netted, data = sam_to_std()

base = StdModelDef().model().create_instance(data)

# Set capital price as the numeraire
base.pf["CAP"].fix(1.0)

# Apply the same variable-domain adjustments as Rocco's model
for name in ["Sg", "Tz", "Tm", "Xg", "X", "Xv"]:
    for var in base.find_component(name).values():
        if var.value is not None and var.value <= 0:
            if var.value < 0:
                var.domain = pyo.Reals
                var.setlb(None)
            else:
                var.setlb(0)

# Apply the Walras Law adjustment
omitted_good = goods[0]
base.eqpqd[omitted_good].deactivate()

# Configure IPOPT
solver = pyo.SolverFactory("ipopt")
solver.set_executable(
    r"C:\Users\supre\miniconda3\Library\bin\ipopt.exe",
    validate=True
)

# Read initial factor endowments
initial_labour = pyo.value(base.FF["LAB"])
initial_capital = pyo.value(base.FF["CAP"])

# Solve the benchmark equilibrium
results = solver.solve(base)
base.solutions.load_from(results)

print("Benchmark LMAN output:", pyo.value(base.Z["LMAN"]))
print("Benchmark total labour:", sum(pyo.value(base.F["LAB", i]) for i in base.i))
print("Benchmark total capital:", sum(pyo.value(base.F["CAP", i]) for i in base.i))

print("\nCGE benchmark solved successfully.")
print("Initial labour:", initial_labour)
print("Initial capital:", initial_capital)


# SOLOW-SWAN CODE
# Load existing Solow-Swan equations and initial conditions

# SOLOW-SWAN CODE
# Initial conditions and parameters from the existing Solow-Swan model

K0 = 12503623082
L0 = 155927
Y0 = 12700000000

g = 0.002
n = 0.005
s = 0.18
alpha = 0.35
delta = 0.08

# Calibrate initial productivity to match the 2026 output
A0 = (
    Y0 / (K0**alpha * L0**(1-alpha))
)**(1 / (1-alpha))


# SOLOW-SWAN CODE
# Differential equations describing changes in K, L and A

def solow_swan_odes(t, stocks, g, n, s, alpha, delta):

    K, L, A = stocks

    Y = K**alpha * (A*L)**(1-alpha)

    dKdt = s*Y - delta*K
    dLdt = n*L
    dAdt = g*A

    return [dKdt, dLdt, dAdt]

# Store the initial economic stocks for 2026

K = K0
L = L0
A = A0


# Prepare a list to store annual results

# Run the annual Solow-Swan projections

annual_results = []

for year in range(2026, 2032):

    # SOLOW-SWAN CODE
    # Calculate output using the current economic stocks

    Y = K**alpha * (A*L)**(1-alpha)

    # [NEW INTEGRATION CODE]
    # Save this year's results

    annual_results.append({
        "Year": year,
        "Capital": K,
        "Labour": L,
        "Productivity": A,
        "Output": Y
    })

    # SOLOW-SWAN CODE
    # Calculate the next year's stocks using the ODE
    if year < 2031:

        solution = solve_ivp(
            solow_swan_odes,
            [year, year + 1],
            [K, L, A],
            args=(g, n, s, alpha, delta)
        )

        if not solution.success:
            raise RuntimeError("Solow-Swan calculation failed")

        K, L, A = solution.y[:, -1]

# Display annual results
print("\nSwindon Dynamic Economic Model")
print("-" * 75)
print(f"{'Year':<8}{'Capital':>18}{'Labour':>15}{'Output':>20}")

for result in annual_results:
    print(
        f"{result['Year']:<8}"
        f"{result['Capital']:>18,.0f}"
        f"{result['Labour']:>15,.0f}"
        f"{result['Output']:>20,.0f}"
    )

# Connect Solow-Swan projections to the CGE model

cge_annual_results = []
cge_sector_results = []

for year_result in annual_results:

    year = year_result["Year"]

    # Calculate proportional changes in labour and capital
    labour_ratio = year_result["Labour"] / L0
    capital_ratio = year_result["Capital"] / K0

    # Update the CGE factor endowments
    base.FF["LAB"].set_value(initial_labour * labour_ratio)
    base.FF["CAP"].set_value(initial_capital * capital_ratio)

    # The 2026 benchmark has already been solved.
    # Solve again for every subsequent year.
    if year > 2026:
        results = solver.solve(base)
        base.solutions.load_from(results)

        if results.solver.termination_condition not in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal
        ):
            raise RuntimeError(
                f"CGE failed to solve for {year}: "
                f"{results.solver.termination_condition}"
            )

    # Record the CGE results for this year
    labour_used = sum(
        pyo.value(base.F["LAB", i]) for i in base.i
    )
    capital_used = sum(
        pyo.value(base.F["CAP", i]) for i in base.i
    )

    cge_annual_results.append({
        "Year": year,
        "Labour endowment": pyo.value(base.FF["LAB"]),
        "Capital endowment": pyo.value(base.FF["CAP"]),
        "Labour used": labour_used,
        "Capital used": capital_used,
        "LMAN output": pyo.value(base.Z["LMAN"]),
        "Total output": sum(
            pyo.value(base.Z[i]) for i in base.i
        )
    })

    # Store output for every sector
    for i in goods:
        cge_sector_results.append({
            "Year": year,
            "Sector": i,
            "Output": pyo.value(base.Z[i])
        })

# Display annual CGE results
print("\nAnnual Dynamic CGE Results")
print("-" * 105)

print(
    f"{'Year':<8}"
    f"{'Labour supply':>17}"
    f"{'Capital supply':>17}"
    f"{'Labour used':>17}"
    f"{'Capital used':>17}"
    f"{'LMAN output':>15}"
)

for result in cge_annual_results:
    print(
        f"{result['Year']:<8}"
        f"{result['Labour endowment']:>17,.3f}"
        f"{result['Capital endowment']:>17,.3f}"
        f"{result['Labour used']:>17,.3f}"
        f"{result['Capital used']:>17,.3f}"
        f"{result['LMAN output']:>15,.3f}"
    )


# Display output by sector for each year
print("\nSectoral Output Projections")
print("-" * 75)

print(f"{'Year':<8}", end="")

for i in goods:
    print(f"{i:>10}", end="")

print()

for year in range(2026, 2032):
    print(f"{year:<8}", end="")

    for i in goods:
        output = next(
            r["Output"] for r in cge_sector_results
            if r["Year"] == year and r["Sector"] == i
        )
        print(f"{output:>10.2f}", end="")

    print()
