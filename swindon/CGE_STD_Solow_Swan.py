import pandas as pd
import pyomo.environ as pyo
from pycge.examples.stdcge_model_def import StdModelDef
import matplotlib.pyplot as plt
from Create_SAM_folder import sam_to_std
from math import exp
import numpy as np

from scipy.integrate import solve_ivp #zach code

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

def base_model(mode, amount, epsilon_L):
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

def scenario_model(amount, mode, epsilon_L):    
    # Create a scenario model with the same data and equations as the base.
    base, shares, extra_investment, goods, netted, data = base_model(mode, amount, epsilon_L)
    scenario = base.clone()
    scenario.eqXv.deactivate()
    for i, share in shares.items():
        if share == 0:
            scenario.Xv[i].setlb(0)

    def investment_rule(model, i):
        # Ordinary investment uses savings. Extra orders receive outside funds.
        savings = model.Sp + model.Sg + model.epsilon * model.Sf
        return model.Xv[i] == (shares[i] * savings / model.pq[i] + extra_investment.get(i, 0.0))

    scenario.scenario_investment = pyo.Constraint(scenario.i, rule=investment_rule)
    if mode == "add_external":
        # eqepsilon is the external payments balance. Price the additional
        # orders in the scenario and convert their cost to external units.
        scenario.eqepsilon.deactivate()
        transfer = sum(scenario.pq[i] * value for i, value in extra_investment.items()) / scenario.epsilon
        scenario.scenario_external_balance = pyo.Constraint(expr=sum(scenario.pWm[i] * scenario.M[i] for i in goods)
                                                            == sum(scenario.pWe[i] * scenario.E[i] for i in goods)
                                                            + scenario.Sf + transfer)

    solver = pyo.SolverFactory("ipopt")
    solver.solve(scenario)

    return scenario, netted, goods, base

def jobs_change(base, scenario, goods):
    # Calculate the change in jobs for each industry.
    # Replace these examples with actual jobs data for every sector.
    employment = {
        "PRIM": 330,
        "LMAN": 1330,
        "MATL": 1100,
        "C21": 1000,
        "ENG": 1425,
        "C27": 900,
        "UTIL": 2750,
        "CONS": 4500,
        "TRAD": 17000,
        "TRAN": 13260,
        "HOSP": 8000,
        "INFO": 3265,
        "FIN": 9750,
        "REAL": 1250,
        "PROF": 6550,
        "M72": 3000,
        "ADMN": 8725,
        "PUBL": 24500,
        "OTHR": 5175,
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


# Solow-Swan aggregate economic projection - zach function
def solow_swan_path(
    initial_capital=12_503_623_082,
    initial_labour=155_927,
    initial_output=12_700_000_000,
    g=0.002,
    n=0.005,
    s=0.18,
    alpha=0.35,
    depreciation=0.08,
    years=5,
):
    # Calculate initial technology from the production function.
    A0 = (
        initial_output
        / (initial_capital**alpha * initial_labour**(1 - alpha))
    ) ** (1 / (1 - alpha))

    # [SOLOW-SWAN CODE] Differential equations for K, L and A.
    def equations(t, y):
        K, L, A = y

        output = K**alpha * (A * L)**(1 - alpha)

        dKdt = s * output - depreciation * K
        dLdt = n * L
        dAdt = g * A

        return [dKdt, dLdt, dAdt]

    # Solve the system at annual intervals.
    years_to_evaluate = np.arange(0, years + 1)

    solution = solve_ivp(
        equations,
        [0, years],
        [initial_capital, initial_labour, A0],
        t_eval=years_to_evaluate,
    )

    if not solution.success:
        raise RuntimeError("Solow-Swan ODE solution failed.")

    # Extract the projected variables.
    K = solution.y[0]
    L = solution.y[1]
    A = solution.y[2]

    # Calculate output and capital per effective worker.
    Y = K**alpha * (A * L)**(1 - alpha)
    k = K / (A * L)

    results = pd.DataFrame({
        "Capital stock (£)": K,
        "Labour": L,
        "Technology": A,
        "Output (£)": Y,
        "Capital per effective worker": k,
    })

    results.index = 2026 + solution.t.astype(int)
    results.index.name = "Year"

    return results



def project_cge(
    base,
    goods,
    solver,
    *,
    scenario=None,
    shock_year=2027,
    initial_capital_stock=12_503_623_082,
    benchmark_investment_pounds=0.18 * 12_700_000_000,
):
    # [CGE CODE] Create the baseline and optional scenario models.
    normal = base.clone()
    shocked = scenario.clone() if scenario is not None else None

    # Record the original 2026 CGE factor endowments.
    capital_endowment_2026 = pyo.value(base.FF["CAP"])
    labour_endowment_2026 = pyo.value(base.FF["LAB"])

    # [NEW INTEGRATION CODE] Generate the aggregate Solow-Swan path.
    solow_results = solow_swan_path(
        initial_capital=initial_capital_stock,
        years=5,
    )

    # Keep the original investment conversion for reporting purposes.
    prices_2026 = {i: pyo.value(base.pq[i]) for i in goods}

    def investment_at_base_prices(model):
        return sum(
            prices_2026[i] * pyo.value(model.Xv[i])
            for i in goods
        )

    investment_scale = (
        benchmark_investment_pounds / investment_at_base_prices(base)
    )

    annual = []
    sectors = []

    # Project the CGE equilibrium from 2026 to 2031.
    for year in range(2026, 2032):

        apply_shock = shocked is not None and year == shock_year
        model = shocked if apply_shock else normal

        # Retrieve the corresponding Solow-Swan projections.
        solow_year = solow_results.loc[year]

        capital_stock = solow_year["Capital stock (£)"]
        labour_stock = solow_year["Labour"]
        technology = solow_year["Technology"]
        solow_output = solow_year["Output (£)"]

        # [NEW INTEGRATION CODE]
        # Scale CGE factor endowments using Solow-Swan growth.
        if year > 2026 or apply_shock:

            model.FF["CAP"].set_value(
                capital_endowment_2026
                * capital_stock / initial_capital_stock
            )

            model.FF["LAB"].set_value(
                labour_endowment_2026
                * labour_stock / solow_results.loc[2026, "Labour"]
            )

            # Solve the CGE equilibrium for this year.
            result = solver.solve(model, load_solutions=False)
            pyo.assert_optimal_termination(result)
            model.solutions.load_from(result)

        # [CGE CODE] Calculate investment demand.
        # This is reported but does not determine aggregate capital.
        investment = (
            investment_scale * investment_at_base_prices(model)
        )

        annual.append({
            "Year": year,
            "Capital stock (£)": capital_stock,
            "Labour stock": labour_stock,
            "Technology": technology,
            "Solow-Swan output (£)": solow_output,
            "CGE investment demand (£)": investment,
            "Labour supply (model units)": pyo.value(model.LS),
        })

        # [CGE CODE] Calculate sector-level employment and output.
        jobs = jobs_change(base, model, goods)

        for i in goods:
            sectors.append({
                "Year": year,
                "Sector": i,
                "Output": pyo.value(model.Z[i]),
                "Labour (model units)": pyo.value(model.F["LAB", i]),
                "Estimated jobs": jobs.loc[i, "Jobs after"],
                "Capital income": pyo.value(
                    model.pf["CAP"] * model.F["CAP", i]
                ),
            })

    return (
        pd.DataFrame(annual).set_index("Year"),
        pd.DataFrame(sectors).set_index(["Year", "Sector"]),
    )




def plot_jobs(investment_results):

    ranking = (
        pd.DataFrame(investment_results)
        .dropna(subset=["Net jobs in 2031"])
        .sort_values("Net jobs in 2031", ascending=False)
    )

    ranked = (
        ranking.dropna(subset=["Net jobs in 2031"])
        .sort_values("Net jobs in 2031", ascending=False)
        .reset_index(drop=True)
    )
    ranked["Rank"] = ranked.index + 1

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(ranked["Rank"], ranked["Net jobs in 2031"], color="steelblue")
    ax.axhline(0, color="grey", linewidth=1)

    best = ranked.iloc[0]
    ax.scatter(best["Rank"], best["Net jobs in 2031"], color="darkorange", zorder=3)
    ax.annotate(
        f"Best tested: {best['Net jobs in 2031']:+.1f} jobs",
        (best["Rank"], best["Net jobs in 2031"]),
        xytext=(10, 8),
        textcoords="offset points",
    )

    ax.set_xlabel("Allocation rank")
    ax.set_ylabel("Net estimated jobs vs baseline")
    ax.set_title("£15m investment reallocation rankings for 2031")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.tight_layout()
    plt.plot()
    plt.show()


def plot_money(investment_captial_results):

    print('hi 2')
    score = "Capital income change in 2031"

    ranking = (
        pd.DataFrame(investment_captial_results)
        .sort_values(score, ascending=False)
        .reset_index(drop=True)
    )

    ranked = (
        ranking.dropna(subset=[score])
        .sort_values(score, ascending=False)
        .reset_index(drop=True)
    )
    ranked["Rank"] = ranked.index + 1

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(ranked["Rank"], ranked[score], color="steelblue")
    ax.axhline(0, color="grey", linewidth=1)

    best = ranked.iloc[0]
    ax.scatter(best["Rank"], best[score], color="darkorange", zorder=3)
    ax.annotate(
        f"Best tested: {best[score]:+.3g}",
        (best["Rank"], best[score]),
        xytext=(10, 8),
        textcoords="offset points",
    )

    ax.set_xlabel("Allocation rank")
    ax.set_ylabel("Change in capital income vs baseline (£m)")
    ax.set_title("£15m investment reallocation rankings for 2031")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.tight_layout()
    plt.plot()
    plt.show()

    print("Best allocation:", best["Allocation"])



##################################################################################################################################################

#these functions define the CGE model


mode = "reallocate"  # "reallocate" or "add_external"

# For reallocation, positive amounts receive investment budget and negative
# amounts give it up. The changes must sum to zero. REAL is an example donor.
# For add_external, use only nonnegative amounts

# Assumed elasticity of labour supply with respect to the real wage.
# The supplied value and its cited source have not been independently verified.
epsilon_L = 0.6
sam, goods, netted, data = sam_to_std()

#This is the total budget and where its coming from
budget = 15.0
#in this case its coming from REAL
donor = "REAL"
recipients = [sector for sector in goods if sector != donor]

# Test giving the entire budget to each sector in turn.
list_of_ammounts = [
    {sector: budget, donor: -budget}
    for sector in recipients
]

# Test an equal split across all eligible sectors.
list_of_ammounts.append({
    **{sector: budget / len(recipients) for sector in recipients},
    donor: -budget,
})

# Test a range of mixed allocations, including concentrated splits.
rng = np.random.default_rng(42)

#creates some noise
for weights in rng.dirichlet(np.full(len(recipients), 0.2), size=100):
    allocation = dict(zip(recipients, budget * weights))
    allocation[donor] = -budget
    list_of_ammounts.append(allocation)


investment_captial_results = []
job_results = []
# Run the base and scenario models, then print the results.
base, _, _, goods, _, _ = base_model(mode, {}, epsilon_L)
#runs the base model over time frame
base_annul,base_sectors = project_cge(base,goods,pyo.SolverFactory("ipopt"))

for amount in list_of_ammounts:
    #runs the simulation
    scenario, netted, goods, base = scenario_model(amount, mode, epsilon_L)

    #find the dynamic of cge of the simulated year this is called the shocked year
    shocked_annul,shocked_sectors = project_cge(scenario,goods,pyo.SolverFactory("ipopt"))

    #number of jobs changed by sector
    jobs_change_by_sector = (shocked_sectors["Estimated jobs"] - base_sectors["Estimated jobs"]).unstack("Sector").sort_index()

    net_jobs_change = jobs_change_by_sector.sum(axis=1)

    net_jobs_by_year = jobs_change_by_sector.sum(axis=1, skipna=False)

    #shows the difference in number of jobs from the base model 
    job_results.append({"Allocation": amount.copy(),"Net jobs in 2031": net_jobs_by_year.loc[2031],})

    #capital change
    capital_income_change = (shocked_sectors["Capital income"] - base_sectors["Capital income"])

    investment_captial_results.append({"Allocation": amount.copy(),"Capital income change in 2031": (capital_income_change.xs(2031, level="Year").sum(skipna=False)),})


# Zach Code - Display the simulation results

#print("\nBASELINE ANNUAL RESULTS")
#print(base_annul)

#print("\nBASELINE SECTOR RESULTS")
#print(base_sectors)

#print("\nINVESTMENT SCENARIO JOB RESULTS")
#print(pd.DataFrame(job_results).to_string(index=False))

#print("\nINVESTMENT SCENARIO CAPITAL INCOME RESULTS")
#print(pd.DataFrame(investment_captial_results).to_string(index=False))

# [NEW CODE] Generate investment scenario graphs
plot_jobs(job_results)
plot_money(investment_captial_results)