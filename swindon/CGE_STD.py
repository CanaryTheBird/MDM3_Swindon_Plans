import pandas as pd  
from Create_SAM_folder import sam_to_std

sector = "LMAN"  # name of the sector to receive the extra investment in the scenario
mode = "add_external"  # "reallocate" or "add_external"
ammount = 10.0  # Investment at benchmark prices, in SAM units



import pyomo.environ as pyo
from pycge.examples.stdcge_model_def import StdModelDef

sam, goods, netted, data = sam_to_std()
base = StdModelDef().model().create_instance(data)
base.pf["CAP"].fix(1.0)
#sg = goverement saving, tg = government dissaving, tm = production subsidies,
#xg = government consumption, x = private consumption, xv = investment
#defined in github library pycge/examples/stdcge_model_def.py
#enables solver to work with negative numbers for these variables, which is necessary for the Swindon SAM.
for name in ["Sg", "Tz", "Tm", "Xg", "X", "Xv"]:
    for var in base.find_component(name).values():
        if var.value is not None and var.value <= 0:
            if var.value < 0:
                #changed domain, to be in the real numbers, inc negatives
                var.domain = pyo.Reals
                #remove lower bound
                var.setlb(None)
            else:
                #sets lower bound to 0
                var.setlb(0)

#With a fixed numeraire, one market equation is redundant. this is due to walras LAW
#need to double check this, not 100% sure if this is correct, but it seems to work.
omitted_good = goods[0]
#eqpqd = equation for the market of good i, defined in github library pycge/examples/stdcge_model_def.py
base.eqpqd[omitted_good].deactivate()
solver = pyo.SolverFactory("ipopt")


#function to find the equalibrium solution for the model, given a model and a label to identify the solution.
def solve(model, label):
    results = solver.solve(model)
    model.solutions.load_from(results)

#solves the base model with no changes to the SAM
solve(base, "Benchmark")

# sets the expected invest for each sector in the model
for i in goods:
    expected = sam.at[i, "INV"]

# Investment shares measure the composition of the savings-funded budget.
weights = sam.loc[goods, "INV"].copy()
total = weights.sum()

#how to handle a reallocation of investment across sectors
if mode == "reallocate":
    #takes the investment share of all other sectors
    other = weights.index != sector
    other_total = weights[other].sum()

    #proportionally redistruibuted investement from other sectors
    #this can also be done by hand
    weights.loc[other] *= (other_total - ammount) / other_total
    weights.loc[sector] += ammount

shares = (weights / total).to_dict()

#clones the base model
scenario = base.clone()
#deactivates the investment market equation, as it will be replaced by a new investment equation that accounts for the changes in the SAM.
scenario.eqXv.deactivate()


def additional_quantity(i):
    #adds the extra investment to the sector if mode is add_external
    return ammount if mode == "add_external" and i == sector else 0.0

def investment_rule(model, i):
    #Sp = private saving, Sg = goverement saving, Sf = external saving, epsilon = exchange rate, pq = price of good i
    #defined in github library pycge/examples/stdcge_model_def.py
    savings = model.Sp + model.Sg + model.epsilon * model.Sf
    return model.Xv[i] == shares[i] * savings / model.pq[i] + additional_quantity(i)

#defines the new investemtn equation as we deactivated the original one
scenario.scenario_investment = pyo.Constraint(scenario.i, rule=investment_rule)
if mode == "add_external":
    #extra funding covers the extra order at its scenario price.
    #Convert local expenditure to external units for the trade balance.
    #eqepsilon = external exchange rate, defined in github library pycge/examples/stdcge_model_def.py
    scenario.eqepsilon.deactivate()
    transfer = scenario.pq[sector] * ammount / scenario.epsilon
    #creates a new external balance equation that accounts for the extra investment in the scenario.
    scenario.scenario_external_balance = pyo.Constraint(
        expr=sum(scenario.pWm[i] * scenario.M[i] for i in goods)
        == sum(scenario.pWe[i] * scenario.E[i] for i in goods) + scenario.Sf + transfer
    )

#solves the scenario model with the changes to the SAM
solve(scenario, "Scenario")

#if any sectors had their gross trade netted, print them out
if (netted > 0).any():
    print("Gross trade netted for:", ", ".join(netted[netted > 0].index))

#output the results of the scenario and the benchmark
results = pd.DataFrame({
    "Investment normal": [pyo.value(base.Xv[i]) for i in goods],
    "Investment changed": [pyo.value(scenario.Xv[i]) for i in goods],
    "Output normal": [pyo.value(base.Z[i]) for i in goods],
    "Output changed": [pyo.value(scenario.Z[i]) for i in goods],
}, index=goods)
results["Output change (%)"] = 100 * (results["Output changed"] / results["Output normal"] - 1)
print(results.round(3).to_string())


