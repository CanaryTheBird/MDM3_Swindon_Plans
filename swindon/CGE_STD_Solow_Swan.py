# Set before importing numerical libraries, including in Windows workers.
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
import time
import hashlib
import re
from pathlib import Path
from scipy.special import ndtr
import pandas as pd
import pyomo.environ as pyo
from pycge.examples.stdcge_model_def import StdModelDef
import matplotlib.pyplot as plt
from Create_SAM_folder import sam_to_std
from math import exp
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    result = solver.solve(base)
    if not pyo.check_optimal_termination(result):
        raise RuntimeError(f"Baseline calibration failed: {result.solver.termination_condition}")

    return base, shares, extra_investment, goods, netted, data

def scenario_model(amount, mode, epsilon_L, cached=None):    
    # Create a scenario model with the same data and equations as the base.
    if cached is None:
        base, shares, extra_investment, goods, netted, data = base_model(mode, amount, epsilon_L)
    else:
        base, sam, goods, netted = cached
        shares, extra_investment = investment_settings(sam, goods, mode, amount)
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
    result = solver.solve(scenario)
    if not pyo.check_optimal_termination(result):
        raise RuntimeError(f"Scenario calibration failed for {amount}: {result.solver.termination_condition}")

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



def make_solver():
    solver = pyo.SolverFactory("ipopt")
    solver.options["max_iter"] = 100
    solver.options["tol"] = 1e-7
    solver.options["bound_relax_factor"] = 0.0
    return solver


def solve_gradually(model, solver, target_cap, target_lab, target_tech):
    params = [model.FF["CAP"], model.FF["LAB"]]
    targets = [target_cap, target_lab]
    if hasattr(model, "tech"):
        params.append(model.tech)
        targets.append(target_tech)
    starts = [pyo.value(p) for p in params]
    variables = list(model.component_data_objects(pyo.Var))
    progress, step = 0.0, 1.0
    min_step = 1e-4
    original_limit = solver.options.get("max_iter", 100)
    last_error = None
    try:
        while progress < 1.0:
            fraction = min(1.0, progress + step)
            saved = [v.value for v in variables]
            success = False
            # Retry difficult small steps with a larger iteration budget.
            limits = (100, 500, 2000) if step <= 0.125 else (100,)
            for limit in limits:
                for var, val in zip(variables, saved):
                    var.set_value(val, skip_validation=True)
                for p, start, target in zip(params, starts, targets):
                    p.set_value(start + fraction * (target - start))
                solver.options["max_iter"] = limit
                try:
                    result = solver.solve(model, load_solutions=False)
                    last_error = result.solver.termination_condition
                    if pyo.check_optimal_termination(result):
                        model.solutions.load_from(result)
                        success = True
                        break
                except Exception as exc:
                    last_error = exc
            if success:
                progress = fraction
                step = min(1.0, step * 1.5)
            else:
                for var, val in zip(variables, saved):
                    var.set_value(val, skip_validation=True)
                for p, start, target in zip(params, starts, targets):
                    p.set_value(start + progress * (target - start))
                step *= 0.5
                if step < min_step:
                    raise RuntimeError(
                        f"Adaptive continuation failed at {progress:.4f}: {last_error}"
                    )
    finally:
        solver.options["max_iter"] = original_limit


def project_cge(
    base, goods, solver, *, scenario=None, shock_year=2027,
    recurring=False, initial_capital_stock=12_503_623_082,
    benchmark_investment_pounds=0.18 * 12_700_000_000,
    initial_labour=155_927, initial_output=12_700_000_000,
    n=0.005, g=0.002, alpha=0.35, depreciation=0.08,
    enable_technology=True, investment_scale_override=None,
):
    # Always use the original calibrated economy as the reference.
    normal = base.clone()
    shocked = scenario.clone() if scenario is not None else None
    cap0 = pyo.value(base.FF["CAP"])
    lab0 = pyo.value(base.FF["LAB"])
    prices = {i: pyo.value(base.pq[i]) for i in goods}

    def real_investment(model):
        return sum(prices[i] * pyo.value(model.Xv[i]) for i in goods)

    # Benchmark conversion from SAM investment units to pounds.
    investment_scale = (benchmark_investment_pounds / real_investment(base)
                        if investment_scale_override is None else investment_scale_override)

    # Update the existing Cobb-Douglas constraint, preserving eqY and eqF.
    # PyCGE's calibrated b parameter is immutable, so use a mutable multiplier.
    if enable_technology:
        for model in [normal, shocked]:
            if model is None:
                continue
            model.tech = pyo.Param(initialize=1.0, mutable=True)
            for i in goods:
                model.eqpy[i].set_value(
                    model.Y[i] == model.b[i]
                    * model.tech ** model.beta["LAB", i]
                    * pyo.prod(model.F[h, i] ** model.beta[h, i]
                               for h in model.h)
                )

    K = initial_capital_stock
    L = initial_labour
    A0 = (initial_output / (K**alpha * L**(1-alpha)))**(1/(1-alpha))
    A = A0
    annual, sectors = [], []

    for year in range(2026, 2032):
        apply_shock = shocked is not None and (
            year == shock_year or (recurring and year >= shock_year)
        )
        model = shocked if apply_shock else normal
        target_cap = cap0 * K / initial_capital_stock
        target_lab = lab0 * L / initial_labour
        target_tech = A / A0

        # The unshocked 2026 benchmark has already been solved.
        if year != 2026 or apply_shock:
            try:
                solve_gradually(model, solver, target_cap, target_lab,
                                target_tech)
            except Exception as exc:
                raise RuntimeError(
                    f"CGE failed in {year}, shock={apply_shock}, "
                    f"K={K:.6g}, L={L:.6g}, technology={target_tech:.6g}: {exc}"
                ) from exc

        investment = investment_scale * real_investment(model)
        # Include productivity in constant-price value added via Y quantities.
        real_gva = sum(pyo.value(base.py[i]) * pyo.value(model.Y[i])
                       for i in goods)
        annual.append({
            "Year": year,
            "Capital stock (£)": K,
            "Labour stock": L,
            "Technology": A,
            "Solow-Swan output (£)": K**alpha * (A*L)**(1-alpha),
            "CGE real GVA (model units)": real_gva,
            "CGE investment demand (£)": investment,
            "Labour supply (model units)": pyo.value(model.LS),
            "Real wage": pyo.value(model.real_wage),
            "Investment scale": investment_scale,
        })
        jobs = jobs_change(base, model, goods)
        for i in goods:
            sectors.append({
                "Year": year, "Sector": i,
                "Output": pyo.value(model.Z[i]),
                "Labour (model units)": pyo.value(model.F["LAB", i]),
                "Estimated jobs": jobs.loc[i, "Jobs after"],
                "Capital income": pyo.value(model.pf["CAP"] * model.F["CAP", i]),
                "Real capital income": pyo.value(model.pf["CAP"] * model.F["CAP", i]
                                                 / model.consumer_price_index),
            })

        # Annual discrete Solow-Swan update using realised CGE investment.
        K = (1 - depreciation) * K + investment
        L *= exp(n)
        A *= exp(g)

    return (pd.DataFrame(annual).set_index("Year"),
            pd.DataFrame(sectors).set_index(["Year", "Sector"]))


def main():
    mode = "reallocate"
    epsilon_L = 0.6
    budget = 15.0
    recipients = ["MATL", "ENG", "C27", "TRAN", "INFO", "FIN", "M72"]
    # Replace these weights with a suspicious allocation divided by 15.
    weights = np.ones(len(recipients)) / len(recipients)
    # Also test all seven single-sector allocations if desired.
    test_single_sectors = False
    sam, _, _, _ = sam_to_std()
    base, _, _, goods, netted, _ = base_model(mode, {}, epsilon_L)
    cached = (base, sam, goods, netted)
    donors = [i for i in goods if i not in recipients]
    donor_inv = sam.loc[donors, "INV"].astype(float)
    if (donor_inv < 0).any() or donor_inv.sum() < budget:
        raise ValueError("Invalid donor investment pool")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Recipient weights must be nonnegative and sum to one")
    donor_changes = (-budget * donor_inv / donor_inv.sum()).to_dict()
    tests = {"zero": {}, "selected": {**dict(zip(recipients, budget * weights)), **donor_changes}}
    if test_single_sectors:
        for i in recipients:
            tests[i] = {i: budget, **donor_changes}
    print("Benchmark SAM investment:", sam.loc[goods, "INV"].sum())
    print("Benchmark model investment:", sum(pyo.value(base.pq[i]*base.Xv[i]) for i in goods))
    annual_rows, sector_rows = [], []
    for label, scale in [("original", None), ("million_units", 1_000_000)]:
        try:
            ba, bs = project_cge(base, goods, make_solver(), investment_scale_override=scale)
        except Exception as exc:
            print(label, "baseline failed:", exc, flush=True)
            continue
        for name, amount in tests.items():
            try:
                scenario, _, _, _ = scenario_model(amount, mode, epsilon_L, cached=cached)
                sa, ss = project_cge(base, goods, make_solver(), scenario=scenario,
                                     recurring=False, investment_scale_override=scale)
                if not bs.index.equals(ss.index):
                    raise ValueError("Baseline and scenario indexes differ")
                comp = pd.DataFrame({"Baseline jobs": bs["Estimated jobs"],
                                     "Scenario jobs": ss["Estimated jobs"],
                                     "Baseline labour": bs["Labour (model units)"],
                                     "Scenario labour": ss["Labour (model units)"]})
                comp["Net jobs"] = comp["Scenario jobs"] - comp["Baseline jobs"]
                annual = comp.groupby(level="Year").sum()
                for col in ["Capital stock (£)", "CGE investment demand (£)", "Labour supply (model units)", "Real wage", "Investment scale"]:
                    annual["Baseline " + col] = ba[col]
                    annual["Scenario " + col] = sa[col]
                print("\n", label, name, "allocation:", amount, flush=True)
                print(annual[["Baseline jobs", "Scenario jobs", "Net jobs"]].to_string())
                print(comp.loc[2031].sort_values("Net jobs", ascending=False).to_string())
                if name == "zero" and annual["Net jobs"].abs().max() > 1:
                    print("WARNING: zero reallocation differs from baseline by more than one job")
                annual_rows.append(annual.reset_index().assign(Scaling=label, Test=name))
                sector_rows.append(comp.reset_index().assign(Scaling=label, Test=name))
                pd.concat(annual_rows, ignore_index=True).to_csv("diagnostic_annual.csv", index=False)
                pd.concat(sector_rows, ignore_index=True).to_csv("diagnostic_sectors.csv", index=False)
            except Exception as exc:
                print(label, name, "failed:", exc, flush=True)

if __name__ == "__main__":
    main()
