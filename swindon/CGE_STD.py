import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pyomo.environ as pyo
from pycge.examples.stdcge_model_def import StdModelDef
import matplotlib.pyplot as plt
from Create_SAM_folder import sam_to_std
from math import exp
import numpy as np

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


def project_cge(base,goods,solver,*,scenario=None,shock_year=2027,initial_capital_stock=12_503_623_082,
                benchmark_investment_pounds=0.18 * 12_700_000_000,depreciation=0.08,labour_growth=0.005,):
    normal = base.clone()
    shocked = scenario.clone() if scenario is not None else None

    capital_endowment_2026 = pyo.value(base.FF["CAP"])
    labour_endowment_2026 = pyo.value(base.FF["LAB"])
    prices_2026 = {i: pyo.value(base.pq[i]) for i in goods}

    def investment_at_base_prices(model):
        return sum(
            prices_2026[i] * pyo.value(model.Xv[i]) for i in goods
        )

    investment_scale = (
        benchmark_investment_pounds / investment_at_base_prices(base)
    )

    capital_stock = initial_capital_stock
    annual = []
    sectors = []

    for year in range(2026, 2032):
        apply_shock = shocked is not None and year == shock_year
        model = shocked if apply_shock else normal

        # The baseline 2026 model already has its benchmark solution.
        if year > 2026 or apply_shock:
            model.FF["CAP"].set_value(
                capital_endowment_2026 * capital_stock / initial_capital_stock
            )
            model.FF["LAB"].set_value(
                labour_endowment_2026 * exp(labour_growth * (year - 2026))
            )

            result = solver.solve(model, load_solutions=False)
            pyo.assert_optimal_termination(result)
            model.solutions.load_from(result)

        investment = investment_scale * investment_at_base_prices(model)
        next_capital_stock = (1 - depreciation) * capital_stock + investment

        annual.append({
            "Year": year,
            "Capital stock (£)": capital_stock,
            "Real investment (£)": investment,
            "Next year's capital stock (£)": next_capital_stock,
            "Labour supply (model units)": pyo.value(model.LS),
        })

        jobs = jobs_change(base, model, goods)
        changed_in_jobs = []
        for i in goods:
            sectors.append({
                "Year": year,
                "Sector": i,
                "Output": pyo.value(model.Z[i]),
                "Labour (model units)": pyo.value(model.F["LAB", i]),
                "Estimated jobs": jobs.loc[i, "Jobs after"],
                "Capital income": pyo.value(model.pf["CAP"] * model.F["CAP", i]),
            })
        


        capital_stock = next_capital_stock

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

def run_one_allocation(amount, mode, epsilon_L, base_sectors):
    """
    Run one investment allocation in a separate process.

    Each worker builds its own Pyomo models and IPOPT solver. This avoids
    trying to share Pyomo model objects between processes.
    """
    scenario, netted, goods, baseline_model = scenario_model(
        amount, mode, epsilon_L
    )

    solver = pyo.SolverFactory("ipopt")
    shocked_annual, shocked_sectors = project_cge(
        scenario, goods, solver
    )

    jobs_change_by_sector = (
        shocked_sectors["Estimated jobs"]
        - base_sectors["Estimated jobs"]
    ).unstack("Sector").sort_index()

    net_jobs_by_year = jobs_change_by_sector.sum(
        axis=1, skipna=False
    )

    capital_income_change = (
        shocked_sectors["Capital income"]
        - base_sectors["Capital income"]
    )

    job_result = {
        "Allocation": amount.copy(),
        "Net jobs in 2031": net_jobs_by_year.loc[2031],
    }

    capital_result = {
        "Allocation": amount.copy(),
        "Capital income change in 2031":
            capital_income_change.xs(2031, level="Year").sum(skipna=False),
    }

    return job_result, capital_result


def main():
    mode = "reallocate"  # "reallocate" or "add_external"

    # Assumed elasticity of labour supply with respect to the real wage.
    epsilon_L = 0.6

    sam, goods, netted, data = sam_to_std()
    goods = list(goods)

    # Total investment budget and its source.
    budget = 15.0
    donor = "REAL"
    recipients = [sector for sector in goods if sector != donor]

    # Test giving the entire budget to each sector in turn.
    list_of_amounts = [
        {sector: budget, donor: -budget}
        for sector in recipients
    ]

    # Test an equal split across all eligible sectors.
    list_of_amounts.append({
        **{sector: budget / len(recipients) for sector in recipients},
        donor: -budget,
    })

    # Test 100 random mixed allocations.
    rng = np.random.default_rng(42)
    for weights in rng.dirichlet(
        np.full(len(recipients), 0.2), size=100
    ):
        allocation = dict(zip(recipients, budget * weights))
        allocation[donor] = -budget
        list_of_amounts.append(allocation)

    print(f"Testing {len(list_of_amounts)} investment allocations.")

    # Solve the baseline once, before starting the parallel scenarios.
    base, _, _, goods, _, _ = base_model(mode, {}, epsilon_L)
    baseline_solver = pyo.SolverFactory("ipopt")
    base_annual, base_sectors = project_cge(
        base, goods, baseline_solver
    )

    # Choose a sensible number of worker processes.
    # Start with 4; increase only if your RAM and CPU can handle it.
    max_workers = min(4, os.cpu_count() or 1, len(list_of_amounts))
    print(f"Running scenarios using {max_workers} worker processes.")

    job_results = []
    investment_capital_results = []

    # Windows requires multiprocessing code to be protected by
    # if __name__ == "__main__": (the call at the bottom does this).
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_one_allocation,
                amount,
                mode,
                epsilon_L,
                base_sectors,
            ): index
            for index, amount in enumerate(list_of_amounts, start=1)
        }

        for future in as_completed(futures):
            scenario_number = futures[future]
            try:
                job_result, capital_result = future.result()
                job_results.append(job_result)
                investment_capital_results.append(capital_result)
                print(
                    f"Finished scenario {scenario_number}/"
                    f"{len(list_of_amounts)}"
                )
            except Exception as error:
                print(
                    f"Scenario {scenario_number} failed: {error}"
                )
                raise

    # Keep results in their original scenario order.
    job_results.sort(
        key=lambda row: list_of_amounts.index(row["Allocation"])
    )
    investment_capital_results.sort(
        key=lambda row: list_of_amounts.index(row["Allocation"])
    )

    print("\nAll scenarios completed.")
    plot_jobs(job_results)
    plot_money(investment_capital_results)


if __name__ == "__main__":
    main()
