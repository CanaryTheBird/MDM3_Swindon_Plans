import pandas as pd
import pyomo.environ as pyo
from pycge.examples.stdcge_model_def import StdModelDef
import matplotlib.pyplot as plt
from Create_SAM_folder import sam_to_std

def investment_settings(sam, goods, mode, amount):
    #Calculate investment expenditure shares for the chosen scenario.
    weights = sam.loc[goods, "INV"].copy()
    total = weights.sum()

    if mode == "reallocate":
        for sector, change in amount.items():
            weights.loc[sector] += change
        extra_investment = {}
    else:
        extra_investment = amount.copy()

    return (weights / total).to_dict(), extra_investment


def endowment(model, factor):
    #Labour supply adjusts, while other factor endowments stay fixed.
    if factor == "LAB":
        return model.LS
    return model.FF[factor]


def factor_income(model):
    #Household factor income includes earnings from the supplied labour.
    return sum(model.pf[h] * endowment(model, h) for h in model.h)

def base_model(mode, ammount, epsilon_L):
    sam, goods, netted, data = sam_to_std()
    goods = list(goods)
    shares, extra_investment = investment_settings(sam, goods, mode, amount)

    base = StdModelDef().model().create_instance(data)
    # Capital services are the price reference for this model.
    base.pf["CAP"].fix(1.0)

    # Sg is government saving, Tz production taxes and Tm import tariffs.
    # Xg is government demand, X intermediate inputs and Xv investment demand.
    # Permit subsidies, government dissaving and zero benchmark transactions.
    for name in ["Sg", "Tz", "Tm", "Xg", "X", "Xv"]:
        for var in base.find_component(name).values():
            if var.value is not None and var.value <= 0:
                if var.value < 0:
                    var.domain = pyo.Reals
                    var.setlb(None)
                else:
                    var.setlb(0)

    # With a price reference fixed, omit one redundant goods market equation.
    omitted_good = goods[0]
    base.eqpqd[omitted_good].deactivate()

    # LS uses the model's benchmark labour quantity units. Calibration at a
    # unit wage gives the same numeric value as benchmark labour payments.
    # Conversion to a number of workers requires employment data.
    base.LS = pyo.Var(
        initialize=pyo.value(base.FF["LAB"]), within=pyo.PositiveReals
    )
    base.eps = pyo.Param(initialize=epsilon_L, mutable=True)

    # Cobb-Douglas household demand implies this consumption price index.
    # Benchmark commodity prices equal 1, so its benchmark value is 1.
    base.consumer_price_index = pyo.Expression(
        rule=lambda m: pyo.prod(m.pq[i] ** m.alpha[i] for i in m.i)

    )
    base.real_wage = pyo.Expression(
        expr=base.pf["LAB"] / base.consumer_price_index
    )

    # Replace household budget equations so they use the supplied labour.
    base.eqTd.deactivate()
    base.eqSp.deactivate()
    base.eqXp.deactivate()
    base.eqpf["LAB"].deactivate()

    base.eqTd_LS = pyo.Constraint(expr=base.Td == base.taud * factor_income(base))
    base.eqSp_LS = pyo.Constraint(expr=base.Sp == base.ssp * factor_income(base))
    base.eqXp_LS = pyo.Constraint(
        base.i,
        rule=lambda m, i: m.Xp[i]
        == m.alpha[i] * (factor_income(m) - m.Sp - m.Td) / m.pq[i],
    )
    base.eqpf_LS = pyo.Constraint(
        expr=sum(base.F["LAB", i] for i in base.i) == base.LS
    )
    base.eqLS = pyo.Constraint(
        expr=base.LS == base.FF["LAB"] * base.real_wage ** base.eps
    )

    solver = pyo.SolverFactory("ipopt")
    solver.solve(base)

    return base, shares, extra_investment, goods, netted, data

def scenario_model(ammount, mode, epsilon_L):    
    # Create a scenario model with the same data and equations as the base.
    base, shares, extra_investment, goods, netted, data = base_model(mode, ammount, epsilon_L)
    scenario = base.clone()
    scenario.eqXv.deactivate()
    for i, share in shares.items():
        if share == 0:
            scenario.Xv[i].setlb(0)

    def investment_rule(model, i):
        # Ordinary investment uses savings. Extra orders receive outside funds.
        savings = model.Sp + model.Sg + model.epsilon * model.Sf
        return model.Xv[i] == (
            shares[i] * savings / model.pq[i] + extra_investment.get(i, 0.0)
        )

    scenario.scenario_investment = pyo.Constraint(
        scenario.i, rule=investment_rule
    )
    if mode == "add_external":
        # eqepsilon is the external payments balance. Price the additional
        # orders in the scenario and convert their cost to external units.
        scenario.eqepsilon.deactivate()
        transfer = sum(
            scenario.pq[i] * value for i, value in extra_investment.items()
        ) / scenario.epsilon
        scenario.scenario_external_balance = pyo.Constraint(
            expr=sum(scenario.pWm[i] * scenario.M[i] for i in goods)
            == sum(scenario.pWe[i] * scenario.E[i] for i in goods)
            + scenario.Sf + transfer
        )

    solver = pyo.SolverFactory("ipopt")
    solver.solve(scenario)

    return scenario, netted, goods, base

def jobs_change(base, scenario, goods):
    # Calculate the change in jobs for each industry.
    # Replace these examples with actual jobs data for every sector.
    employment = {
        "PRIM": 1000,
        "LMAN": 2000,
        "MATL": 500,
        "C21": 1500,
        "ENG": 800,
        "C27": 1200,
        "UTIL": 600,
        "CONS": 400,
        "TRAD": 300,
        "TRAN": 700,
        "HOSP": 900,
        "INFO": 500,
        "FIN": 400,
        "REAL": 300,
        "PROF": 600,
        "M72": 200,
        "ADMN": 100,
        "PUBL": 800,
        "OTHR": 500,

        # Add the remaining sectors.
    }

    jobs = pd.DataFrame(index=goods)
    jobs["Jobs before"] = pd.Series(employment)

    jobs["Labour growth"] = [
        pyo.value(scenario.F["LAB", i])
        / pyo.value(base.F["LAB", i])
        - 1
        for i in goods
    ]
    
    jobs["Jobs after"] = jobs["Jobs before"] * (1 + jobs["Labour growth"])

    jobs["Jobs change"] = jobs["Jobs after"] - jobs["Jobs before"]

    return jobs

def print_market_residuals(label, model, netted, goods, base, scenario): 
    if (netted > 0).any():
        print("Gross trade netted for:", ", ".join(netted[netted > 0].index))

    results = pd.DataFrame({
        "Investment normal": [pyo.value(base.Xv[i]) for i in goods],
        "Investment changed": [pyo.value(scenario.Xv[i]) for i in goods],
        "Output normal": [pyo.value(base.Z[i]) for i in goods],
        "Output changed": [pyo.value(scenario.Z[i]) for i in goods],
    }, index=goods)
    results["Output change (%)"] = 100 * (
        results["Output changed"] / results["Output normal"] - 1
    )
    print(results.round(3).to_string())

    print(f"Benchmark labour: {pyo.value(base.LS):.6f}")
    for label, before, after in [
        ("Labour supply", base.LS, scenario.LS),
        ("Wage", base.pf["LAB"], scenario.pf["LAB"]),
        ("Consumer prices", base.consumer_price_index, scenario.consumer_price_index),
        ("Real wage", base.real_wage, scenario.real_wage),
    ]:
        change = 100 * (pyo.value(after) / pyo.value(before) - 1)
        print(f"{label} change: {change:.6f}%")


#these functions define the CGE model


mode = "reallocate"  # "reallocate" or "add_external"

# For reallocation, positive amounts receive investment budget and negative
# amounts give it up. The changes must sum to zero. REAL is an example donor.
# For add_external, use only nonnegative amounts

# Assumed elasticity of labour supply with respect to the real wage.
# The supplied value and its cited source have not been independently verified.
epsilon_L = 0.6

list_of_ammounts = [{"LMAN": 15.0, "CONS": 0.0, "REAL": -15.0},
                    {"LMAN": 10.0, "CONS": 5.0, "REAL": -15.0},
                    {"LMAN": 5.0, "CONS": 10.0, "REAL": -15.0},
                    {"LMAN": 0.0, "CONS": 15.0, "REAL": -15.0},]
# Run the base and scenario models, then print the results.
for amount in list_of_ammounts:
    scenario, netted, goods, base = scenario_model(amount, mode, epsilon_L)
    jobs = jobs_change(base, scenario, goods)
    print(jobs[["Jobs before", "Jobs after"]].round(0).astype(int).to_string())
    print(jobs["Jobs change"].sum().round(0).astype(int))
