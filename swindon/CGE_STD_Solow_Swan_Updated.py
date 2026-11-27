"""Swindon recursive CGE with sector-specific installed capital (V6).

Policy amounts are in benchmark £m. The SAM's INV column identifies goods
purchased for investment, not the industry installing those assets. Without
an investment-by-destination matrix, assume neutral gross investment is
allocated by benchmark capital income shares, and apply the same signed
policy transfer to installed capital as to benchmark commodity expenditure.
Installed capital is immobile between industries; labour remains mobile.
This is a new model closure, not a correction to the old numerical results.
"""
# Set before importing numerical libraries, including in Windows workers.
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
import time
import hashlib
import re
from pathlib import Path
import pandas as pd
import pyomo.environ as pyo
from pycge.examples.stdcge_model_def import StdModelDef
import matplotlib.pyplot as plt
from Create_SAM_folder import sam_to_std
from math import exp
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from math import comb
import csv

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
                    var.domain = pyo.NonNegativeReals
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
    # No separate scenario equilibrium: pass signed £m transfers to the path.
    if mode != "reallocate":
        raise ValueError("This corrected version supports reallocation only.")
    if cached is None:
        sam, _, _, _ = sam_to_std()
        base, _, _, goods, netted, _ = base_model(mode, {}, epsilon_L)
    else:
        base, sam, goods, netted = cached
    # Return signed transfers in benchmark £m, not share changes.
    unknown = set(amount) - set(goods)
    if unknown:
        raise ValueError(f"Unknown investment sectors: {unknown}")
    transfer = {i: float(amount.get(i, 0.0)) for i in goods}
    if not np.isfinite(list(transfer.values())).all() or abs(sum(transfer.values())) > 1e-8:
        raise ValueError("Reallocation must be finite and budget-neutral.")
    return transfer, netted, goods, base


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


def solve_gradually(model, solver, target_cap, target_lab, target_tech, target_shares):
    # Update all sectoral capital supplies along the same continuation path.
    params = [model.FF["CAP"], model.FF["LAB"]]
    targets = [sum(target_cap.values()), target_lab]
    for i in model.i:
        params.append(model.capital_supply[i])
        targets.append(target_cap[i])
    if hasattr(model, "tech"):
        params.append(model.tech)
        targets.append(target_tech)
    for i in model.i:
        params.append(model.investment_share[i])
        targets.append(target_shares[i])
    starts = [pyo.value(p) for p in params]
    variables = list(model.component_data_objects(pyo.Var))
    progress, step = 0.0, 0.25
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
                step = min(0.25, step * 1.5)
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


def installed_investment(total_real_investment, capital_weights, transfer_pounds):
    """Allocate a single pool of investment; policy shifts its destination.

    capital_weights and transfer_pounds are arrays in the same sector order.
    This is an assumed destination mapping, not data identified by the SAM.
    """
    weights = np.asarray(capital_weights, dtype=float)
    transfer = np.asarray(transfer_pounds, dtype=float)
    if (weights.shape != transfer.shape or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1.0)
            or not np.isfinite(transfer).all() or abs(transfer.sum()) > 0.01
            or not np.isfinite(total_real_investment) or total_real_investment <= 0):
        raise ValueError("Invalid installed investment inputs or nonzero net transfer.")
    result = total_real_investment * weights + transfer
    if (result < 0).any():
        raise ValueError("Insufficient installed investment in a donor sector.")
    if not np.isclose(result.sum(), total_real_investment, rtol=1e-12, atol=0.01):
        raise RuntimeError("Installed investment does not balance.")
    return result


def project_cge(
    base, goods, solver, *, scenario=None, shock_year=2027,
    recurring=False, initial_capital_stock=12_503_623_082,
    investment_scale=1_000_000,
    initial_labour=155_927, initial_output=12_700_000_000,
    n=0.005, g=0.002, alpha=0.35, depreciation=0.08,
    enable_technology=True,
):
    # One model per trajectory; capital stocks are carried forward by industry.
    goods = list(goods)
    if len(goods) != len(set(goods)):
        raise ValueError(f"Duplicate sectors in goods: {goods}")
    model = base.clone()
    lab0 = pyo.value(base.FF["LAB"])
    prices = {i: pyo.value(base.pq[i]) for i in goods}
    savings0 = pyo.value(base.Sp + base.Sg + base.epsilon * base.Sf)
    if savings0 <= 0:
        raise ValueError("Nonpositive benchmark savings.")
    original = {i: prices[i] * pyo.value(base.Xv[i]) / savings0 for i in goods}
    if abs(sum(original.values()) - 1) > 1e-6:
        raise ValueError("Benchmark investment does not match savings.")
    # The scenario is a dictionary of signed benchmark £m transfers.
    scenario = scenario or {}
    if set(scenario) - set(goods):
        raise ValueError("Unknown sector in policy transfer.")
    transfer = np.array([scenario.get(i, 0.0) for i in goods], dtype=float)
    if not np.isfinite(transfer).all() or abs(transfer.sum()) > 1e-8:
        raise ValueError("Policy transfers must be finite and budget-neutral.")
    # Maintain the V5 commodity-demand channel: benchmark expenditure shares.
    benchmark_inv = sum(pyo.value(base.Xv0[i]) for i in goods)
    if benchmark_inv <= 0:
        raise ValueError("Nonpositive SAM investment total.")
    changed = {i: original[i] + transfer[j] / benchmark_inv for j, i in enumerate(goods)}
    if min(changed.values()) < -1e-10 or abs(sum(changed.values())-1) > 1e-6:
        raise ValueError("Invalid investment shares after reallocation.")
    model.investment_share = pyo.Param(model.i, initialize=original, mutable=True)
    model.eqXv.deactivate()
    model.scenario_investment = pyo.Constraint(model.i, rule=lambda m,i:
        m.pq[i]*m.Xv[i] == m.investment_share[i]*(m.Sp+m.Sg+m.epsilon*m.Sf))
    for i in goods:
        if original[i] >= 0:
            model.Xv[i].domain = pyo.NonNegativeReals
            model.Xv[i].setlb(0)

    # Benchmark capital services are measured at the common benchmark rental
    # price. Use their shares as a proxy for the shares of physical capital.
    cap0 = np.array([pyo.value(base.F["CAP", i]) for i in goods])
    if (cap0 <= 0).any() or not np.isfinite(cap0).all():
        raise ValueError("All sectoral benchmark capital services must be positive.")
    weights = cap0 / cap0.sum()
    K0 = initial_capital_stock * weights
    K = K0.copy()
    weight_by_sector = dict(zip(goods, weights))
    model.capital_supply = pyo.Param(model.i, mutable=True,
                                    initialize=dict(zip(goods, cap0)))
    model.capital_rent = pyo.Var(model.i, domain=pyo.PositiveReals,
                                 initialize=pyo.value(base.pf["CAP"]))
    # Fix each industry's capital services to its installed stock. Its rental
    # price is endogenous, so firms still satisfy their factor demand equations.
    model.eqpf["CAP"].deactivate()
    for i in goods:
        model.eqF["CAP", i].set_value(
            model.F["CAP", i] == model.beta["CAP", i]*model.py[i]*model.Y[i]/model.capital_rent[i])
    model.sector_capital_market = pyo.Constraint(model.i, rule=lambda m,i:
        m.F["CAP", i] == m.capital_supply[i])
    # Retain the previous capital price reference as a fixed-weight rental index.
    model.capital_numeraire = pyo.Constraint(expr=sum(
        weight_by_sector[i]*model.capital_rent[i] for i in goods) == model.pf["CAP"])
    # Household income now uses each industry's own rental rate, not one wage
    # for all capital. Update all three income-dependent household equations.
    model.sector_factor_income = pyo.Expression(expr=
        model.pf["LAB"]*model.LS + sum(model.capital_rent[i]*model.F["CAP", i] for i in goods)
        + sum(model.pf[h]*model.FF[h] for h in model.h if h not in ("CAP", "LAB")))
    model.eqTd_LS.set_value(model.Td == model.taud*model.sector_factor_income)
    model.eqSp_LS.set_value(model.Sp == model.ssp*model.sector_factor_income)
    for i in goods:
        model.eqXp_LS[i].set_value(model.Xp[i] == model.alpha[i]*(
            model.sector_factor_income-model.Sp-model.Td)/model.pq[i])

    if enable_technology:
        model.tech = pyo.Param(initialize=1.0, mutable=True)
        for i in goods:
            model.eqpy[i].set_value(model.Y[i] == model.b[i]
                * model.tech ** model.beta["LAB", i]
                * pyo.prod(model.F[h,i] ** model.beta[h,i] for h in model.h))

    L = initial_labour
    A0 = (initial_output / (initial_capital_stock**alpha * L**(1-alpha)))**(1/(1-alpha))
    A = A0
    annual, sectors = [], []
    for year in range(2026, 2032):
        apply_shock = bool(np.any(transfer)) and (
            year == shock_year or (recurring and year >= shock_year))
        target_shares = changed if apply_shock else original
        target_cap = dict(zip(goods, cap0 * K / K0))
        target_lab = lab0 * L / initial_labour
        target_tech = A / A0
        # The calibrated 2026 solution also satisfies the new sector capital
        # equations by construction. No independent shocked model is created.
        if year != 2026 or apply_shock:
            try:
                solve_gradually(model, solver, target_cap, target_lab,
                                target_tech, target_shares)
            except Exception as exc:
                raise RuntimeError(f"CGE failed in {year}, shock={apply_shock}, "
                    f"K={K.sum():.6g}, L={L:.6g}: {exc}") from exc
        # Keep the existing market-residual check, including the omitted good.
        for con in list(model.component_data_objects(pyo.Constraint, active=True)) + [model.eqpqd[goods[0]]]:
            body = pyo.value(con.body)
            for bound, sign in [(con.lower, -1), (con.upper, 1)]:
                if bound is not None:
                    rhs = pyo.value(bound)
                    if not np.isfinite(body) or sign*(body-rhs) > 1e-3 + 1e-6 * max(1, abs(rhs), abs(body)):
                        raise RuntimeError(f"Market/constraint residual in {year}: {con.name}: {body-rhs}")
        commodity_investment = investment_scale * np.array([
            prices[i] * pyo.value(model.Xv[i]) for i in goods])
        investment = float(commodity_investment.sum())
        # The same investment pool is allocated to productive destinations.
        # No £15m is added to aggregate capital: the transfer sums to zero.
        policy_pounds = transfer * investment_scale if apply_shock else np.zeros(len(goods))
        installed = installed_investment(investment, weights, policy_pounds)
        K_next = (1 - depreciation) * K + installed
        if (K_next <= 0).any() or not np.isfinite(K_next).all():
            raise ValueError(f"Invalid next-year sectoral capital stocks in {year}")
        if not np.isclose(K_next.sum(), (1-depreciation)*K.sum()+investment, rtol=1e-12, atol=0.01):
            raise RuntimeError("Aggregate capital accounting error.")
        real_gva = sum(pyo.value(base.py[i]) * pyo.value(model.Y[i]) for i in goods)
        annual.append({
            "Year": year,
            "Capital stock (£)": K.sum(),
            "Labour stock": L,
            "Technology": A,
            "Solow-Swan output (£)": K.sum()**alpha * (A*L)**(1-alpha),
            "CGE real GVA (model units)": real_gva,
            "CGE investment demand (£)": investment,
            "Installed investment (£)": installed.sum(),
            "Labour supply (model units)": pyo.value(model.LS),
        })
        jobs = jobs_change(base, model, goods)
        for j, i in enumerate(goods):
            income = pyo.value(model.capital_rent[i]*model.F["CAP", i])
            sectors.append({
                "Year": year, "Sector": i,
                "Output": pyo.value(model.Z[i]),
                "Labour (model units)": pyo.value(model.F["LAB", i]),
                "Estimated jobs": jobs.loc[i, "Jobs after"],
                "Capital stock (£)": K[j],
                "Commodity investment (£)": commodity_investment[j],
                "Installed investment (£)": installed[j],
                "Policy capital transfer (£)": policy_pounds[j],
                "Capital rental price": pyo.value(model.capital_rent[i]),
                "Capital income": income,
                "Real capital income": income / pyo.value(model.consumer_price_index),
            })
        K = K_next
        L *= exp(n)
        A *= exp(g)
    annual_frame = pd.DataFrame(annual).set_index("Year")
    sector_frame = pd.DataFrame(sectors).set_index(["Year", "Sector"])
    if not annual_frame.index.is_unique or not sector_frame.index.is_unique:
        raise ValueError("Duplicate year or year-sector entries in projection.")
    return annual_frame, sector_frame


def baseline_diagnostics(base_annual, base_sectors):
    """Export the baseline, labour-to-jobs conversion and sectoral changes.

    Labour quantities are benchmark factor units, not headcounts. These
    diagnostics do not change the calibrated model or its equations.
    """
    if not base_sectors.index.is_unique:
        raise ValueError("Duplicate year-sector keys in baseline diagnostics.")
    jobs = base_sectors["Estimated jobs"].unstack("Sector").sort_index()
    labour = base_sectors["Labour (model units)"].unstack("Sector").sort_index()
    if not jobs.index.equals(labour.index) or not jobs.columns.equals(labour.columns):
        raise ValueError("Employment and labour tables do not align.")
    if not np.isfinite(jobs.to_numpy()).all() or not np.isfinite(labour.to_numpy()).all():
        raise ValueError("Nonfinite baseline labour or jobs data.")
    if (labour.loc[2026] <= 0).any() or (jobs.loc[2026] <= 0).any():
        raise ValueError("Nonpositive benchmark labour or employment; check sector data.")

    jobs_change = jobs.subtract(jobs.loc[2026], axis=1)
    jobs_growth = (jobs.divide(jobs.loc[2026], axis=1) - 1) * 100
    labour_growth = (labour.divide(labour.loc[2026], axis=1) - 1) * 100
    comparison = base_annual.copy()
    comparison["Estimated jobs"] = jobs.sum(axis=1, skipna=False)
    comparison["CGE labour demand"] = labour.sum(axis=1, skipna=False)
    comparison["Jobs growth (%)"] = (comparison["Estimated jobs"] / jobs.loc[2026].sum() - 1) * 100
    comparison["Labour growth (%)"] = (comparison["CGE labour demand"] / labour.loc[2026].sum() - 1) * 100
    comparison["Labour market gap (model units)"] = (
        comparison["CGE labour demand"] - comparison["Labour supply (model units)"]
    )

    conversion_factor = jobs.loc[2026] / labour.loc[2026]
    sector_results = pd.DataFrame({
        "Jobs in 2026": jobs.loc[2026],
        "Jobs in 2031": jobs.loc[2031],
        "Jobs change": jobs_change.loc[2031],
        "Jobs growth (%)": jobs_growth.loc[2031],
        "Labour in 2026 (model units)": labour.loc[2026],
        "Labour in 2031 (model units)": labour.loc[2031],
        "Labour growth (%)": labour_growth.loc[2031],
        "Jobs per labour unit": conversion_factor,
        "Labour share 2026 (%)": labour.loc[2026] / labour.loc[2026].sum() * 100,
        "Labour share 2031 (%)": labour.loc[2031] / labour.loc[2031].sum() * 100,
    }).sort_values("Jobs change")
    conversion_results = sector_results[["Jobs in 2026", "Labour in 2026 (model units)",
                                         "Jobs per labour unit", "Jobs change"]].sort_values(
                                             "Jobs per labour unit", ascending=False)
    print("\nBaseline employment and labour demand:")
    print(comparison[["Estimated jobs", "CGE labour demand", "Jobs growth (%)",
                      "Labour growth (%)", "Labour market gap (model units)"]].round(3).to_string())
    print("\nBaseline employment changes by sector:")
    print(sector_results[["Jobs in 2026", "Jobs in 2031", "Jobs change",
                          "Jobs growth (%)", "Jobs per labour unit"]].round(3).to_string())
    comparison.to_csv("v6_baseline_employment_diagnostic.csv")
    sector_results.to_csv("v6_baseline_sector_diagnostic.csv")
    conversion_results.to_csv("v6_labour_conversion_diagnostic.csv")
    base_sectors.to_csv("v6_baseline_sector_annual.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(comparison.index, comparison["Jobs growth (%)"], marker="o", label="Estimated jobs")
    ax.plot(comparison.index, comparison["Labour growth (%)"], marker="s", label="CGE labour demand")
    ax.axhline(0, color="grey", linestyle="--")
    ax.set(xlabel="Year", ylabel="Change relative to 2026 (%)",
           title="Baseline employment and labour demand")
    ax.set_xticks(comparison.index)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("v6_baseline_employment_growth.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    changes = sector_results["Jobs change"]
    ax.barh(changes.index, changes.values,
            color=["seagreen" if x >= 0 else "indianred" for x in changes])
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="Estimated employment change", ylabel="Sector",
           title="Baseline sectoral employment change: 2026–2031")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig("v6_baseline_sector_employment.png", dpi=200)
    plt.close(fig)
    print("Baseline diagnostics and figures saved.", flush=True)
    return comparison, sector_results


def plot_jobs(investment_results):

    ranking = (
        pd.DataFrame(investment_results)
        .dropna(subset=["Cumulative job-years 2027-2031"])
        .sort_values("Cumulative job-years 2027-2031", ascending=False)
    )

    ranked = (
        ranking.dropna(subset=["Cumulative job-years 2027-2031"])
        .sort_values("Cumulative job-years 2027-2031", ascending=False)
        .reset_index(drop=True)
    )
    ranked["Rank"] = ranked.index + 1

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(ranked["Rank"], ranked["Cumulative job-years 2027-2031"], color="steelblue")
    ax.axhline(0, color="grey", linewidth=1)

    best = ranked.iloc[0]
    ax.scatter(best["Rank"], best["Cumulative job-years 2027-2031"], color="darkorange", zorder=3)
    ax.annotate(
        f"Best tested: {best['Cumulative job-years 2027-2031']:+.1f} job-years",
        (best["Rank"], best["Cumulative job-years 2027-2031"]),
        xytext=(10, 8),
        textcoords="offset points",
    )

    ax.set_xlabel("Allocation rank")
    ax.set_ylabel("Cumulative net job-years vs baseline")
    ax.set_title("£15m reallocation: cumulative employment, 2027–2031")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.tight_layout()
    plt.plot()
    plt.show()


def plot_money(investment_captial_results):

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
    ax.set_ylabel("Change in real capital income vs baseline (SAM units)")
    ax.set_title("£15m investment reallocation rankings for 2031")

    # Zoom in on the range of capital-income changes. -zach code
    y_min = ranked[score].min()
    y_max = ranked[score].max()
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()
    plt.plot()
    plt.show()

    print("Best allocation:", best["Allocation"])




##################################################################################################################################################

# Worker function: one investment allocation per process.
# Keep this function at module level so Windows multiprocessing can pickle it.
_worker_cache = None


def initialise_worker(mode, epsilon_L):
    global _worker_cache
    sam, _, _, _ = sam_to_std()
    base, _, _, goods, netted, _ = base_model(mode, {}, epsilon_L)
    _worker_cache = (base, sam, goods, netted)
    global _worker_baseline
    _, _worker_baseline = project_cge(base, goods, make_solver())


def run_one_allocation(amount, mode, epsilon_L, base_sectors, recipients, recurring=False):
    scenario, netted, goods, scenario_base = scenario_model(
        amount, mode, epsilon_L, cached=_worker_cache
    )
    def checked_table(frame, name):
        table = frame.reset_index()
        keys = ["Year", "Sector"]
        duplicates = table.loc[table.duplicated(keys, keep=False), keys]
        if not duplicates.empty:
            raise ValueError(f"{name} duplicate year-sector keys: {duplicates.to_dict('records')}")
        return table

    def compare(left, right, name):
        keys = ["Year", "Sector"]
        merged = checked_table(left, name).merge(
            checked_table(right, "Baseline"), on=keys, how="outer",
            suffixes=("_scenario", "_base"), validate="one_to_one", indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"{name} and baseline year-sector keys differ")
        return merged

    worker_check = compare(_worker_baseline, base_sectors, "Worker baseline")
    for column, atol in [("Estimated jobs", 0.05), ("Real capital income", 1e-4),
                         ("Capital stock (£)", 10.0)]:
        if not np.allclose(worker_check[f"{column}_scenario"], worker_check[f"{column}_base"],
                           atol=atol, rtol=1e-8):
            raise RuntimeError(f"Worker baseline differs from parent baseline: {column}")
    solver = make_solver()
    shocked_annual, shocked_sectors = project_cge(
        scenario_base, goods, solver, scenario=scenario,
        shock_year=2027, recurring=recurring,
    )
    comparison = compare(shocked_sectors, _worker_baseline, "Scenario")
    comparison["Jobs change"] = (comparison["Estimated jobs_scenario"]
                                 - comparison["Estimated jobs_base"])
    comparison["Capital change"] = (comparison["Real capital income_scenario"]
                                    - comparison["Real capital income_base"])
    if not np.isfinite(comparison[["Jobs change", "Capital change"]].to_numpy()).all():
        raise ValueError("Nonfinite sector results")
    net_jobs_by_year = comparison.groupby("Year")["Jobs change"].sum().sort_index()
    if net_jobs_by_year.index.tolist() != list(range(2026, 2032)):
        raise ValueError("Missing projection years in employment results.")
    total_capital_income_change = comparison.loc[
        comparison["Year"].eq(2031), "Capital change"
    ].sum()

    return (
        {
            "Allocation": amount.copy(),
            "Cumulative job-years 2027-2031": net_jobs_by_year.loc[2027:2031].sum(),
            "Net jobs in 2031": net_jobs_by_year.loc[2031],
            "Peak jobs 2027-2031": net_jobs_by_year.loc[2027:2031].max(),
            **{f"{i} capital stock change in 2031 (£)": float(
                comparison.loc[comparison["Year"].eq(2031) & comparison["Sector"].eq(i),
                               "Capital stock (£)_scenario"].iloc[0]
                - comparison.loc[comparison["Year"].eq(2031) & comparison["Sector"].eq(i),
                                 "Capital stock (£)_base"].iloc[0]) for i in recipients},
            **{f"Net jobs in {year}": net_jobs_by_year.loc[year]
               for year in range(2027, 2031)},
        },
        {
            "Allocation": amount.copy(),
            "Capital income change in 2031": total_capital_income_change,
        },
    )



# Exhaustive search settings. Runtime settings can change between sessions.
BUDGET = 15
INCREMENT = 1   # £m; integer increments that divide BUDGET
WORKERS = 16
MAX_RUNTIME_HOURS = 40  # Per session; None means no time limit.
MAX_ATTEMPTS_PER_SESSION = None  # Set to 12 for a short smoke test.
RANKINGS_EVERY = 250
RECURRING = False
EPSILON_L = 0.6
RECIPIENTS = ["MATL", "ENG", "C27", "TRAN", "INFO", "FIN", "M72"]
JOBS = "Cumulative job-years 2027-2031"
CAPITAL = "Capital income change in 2031"


def integer_allocations(total, dimensions):
    """Generate each weak composition exactly once, using integer arithmetic."""
    if dimensions == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for tail in integer_allocations(total - first, dimensions - 1):
            yield (first,) + tail


def make_grid(budget, increment, dimensions):
    if not isinstance(budget, int) or not isinstance(increment, int) or increment <= 0 or budget <= 0 or budget % increment:
        raise ValueError("Use positive integer £m increments that divide the budget.")
    # Test pure allocations first, then pairs, then increasingly mixed ones.
    grid = list(integer_allocations(budget // increment, dimensions))
    grid.sort(key=lambda a: (sum(v > 0 for v in a), tuple(-v for v in a)))
    return [tuple(v * increment for v in a) for a in grid]


def grid_worker_init(mode, epsilon_L, base_sectors, recipients, recurring):
    global _grid_worker_args
    initialise_worker(mode, epsilon_L)
    _grid_worker_args = (mode, epsilon_L, base_sectors, recipients, recurring)


def grid_worker(amount):
    return run_one_allocation(amount, *_grid_worker_args)


def atomic_csv(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    start = time.monotonic()
    deadline = float("inf") if MAX_RUNTIME_HOURS is None else start + MAX_RUNTIME_HOURS * 3600
    mode = "reallocate"
    recipients = RECIPIENTS.copy()
    sam, goods, _, _ = sam_to_std()
    goods = list(goods)
    if set(recipients) - set(goods):
        raise ValueError("Some recipient sectors are missing from the SAM.")
    donors = [s for s in goods if s not in recipients]
    donor_inv = sam.loc[donors, "INV"].astype(float)
    if (donor_inv < 0).any() or donor_inv.sum() < BUDGET:
        raise ValueError("Insufficient nonnegative benchmark donor investment.")
    donor_changes = (-BUDGET * donor_inv / donor_inv.sum()).to_dict()

    grid = make_grid(BUDGET, INCREMENT, len(recipients))
    total = len(grid)
    assert total == comb(BUDGET // INCREMENT + len(recipients) - 1, len(recipients) - 1)
    print(f"Exhaustive grid: {total:,} unique allocations", flush=True)
    prefix = f"grid_v6_{INCREMENT}m"
    checkpoint = Path(f"{prefix}_investment_results.csv")
    # Hash economic functions + SAM + grid definition and policy settings.
    # Runtime limits and worker count may be changed without invalidating results.
    source = Path(__file__).read_text(encoding="utf-8")
    model_source = source.split("# Exhaustive search settings.", 1)[0]
    signature = hashlib.sha256((model_source + sam.to_csv() + repr((
        BUDGET, INCREMENT, recipients, mode, EPSILON_L, RECURRING, "grid_order_v1"
    ))).encode()).hexdigest()
    columns = ["Scenario", "Run signature", *goods, JOBS, CAPITAL,
               "Net jobs in 2031", "Peak jobs 2027-2031",
               *[f"{i} capital stock change in 2031 (£)" for i in recipients],
               *[f"Net jobs in {y}" for y in range(2027, 2031)], "Status", "Error"]
    rows = []
    completed = set()
    if checkpoint.exists():
        saved = pd.read_csv(checkpoint)
        if list(saved.columns) != columns or not saved["Run signature"].eq(signature).all():
            raise ValueError("Checkpoint is incompatible with this model/grid. Do not mix runs.")
        if saved["Scenario"].duplicated().any():
            raise ValueError("Duplicate IDs in checkpoint; inspect before continuing.")
        for row in saved.to_dict("records"):
            idx = int(row["Scenario"])
            if idx != row["Scenario"] or not 1 <= idx <= total or row["Status"] not in ("Success", "Failed"):
                raise ValueError(f"Invalid checkpoint row: {idx}")
            if not np.allclose([row[s] for s in recipients], grid[idx-1], rtol=0, atol=1e-9):
                raise ValueError(f"Checkpoint allocation mismatch: {idx}")
            if not np.allclose([row[s] for s in donors], [donor_changes[s] for s in donors], rtol=0, atol=1e-9):
                raise ValueError(f"Checkpoint donor mismatch: {idx}")
            if row["Status"] == "Success" and not np.isfinite([row[c] for c in columns if c not in ("Scenario", "Run signature", "Status", "Error")]).all():
                raise ValueError(f"Incomplete successful checkpoint row: {idx}")
            completed.add(idx)
            rows.append(row)
        print(f"Resuming: {len(completed):,}/{total:,} already attempted.", flush=True)
    else:
        with checkpoint.open("x", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()
            f.flush()
            os.fsync(f.fileno())

    def export_rankings():
        frame = pd.DataFrame(rows, columns=columns)
        successful = frame.loc[frame["Status"].eq("Success")]
        atomic_csv(successful.sort_values(JOBS, ascending=False), Path(f"{prefix}_jobs_rankings.csv"))
        atomic_csv(successful.sort_values(CAPITAL, ascending=False), Path(f"{prefix}_capital_rankings.csv"))
        atomic_csv(frame.loc[frame["Status"].eq("Failed")], Path(f"{prefix}_failed_scenarios.csv"))
        # Exact non-dominated set among evaluated successful grid points.
        ordered = successful.sort_values([JOBS, CAPITAL], ascending=False)
        frontier = ordered.loc[ordered[CAPITAL] > ordered[CAPITAL].cummax().shift(fill_value=-np.inf)]
        atomic_csv(frontier.sort_values(JOBS), Path(f"{prefix}_pareto_allocations.csv"))
        if not successful.empty:
            for objective in (JOBS, CAPITAL):
                best = successful.loc[successful[objective].idxmax()]
                print(f"Best tested {objective}: {best[objective]:.8g} | "
                      f"{dict((s, float(best[s])) for s in recipients)}", flush=True)
        print(f"Coverage: {len(frame):,}/{total:,} attempted; {len(successful):,} successful; "
              f"{len(frame)-len(successful):,} failed.", flush=True)

    remaining = ((idx, a) for idx, a in enumerate(grid, 1) if idx not in completed)
    if len(completed) == total:
        export_rankings()
        print("All grid allocations have been attempted. Check failed scenarios before claiming an exhaustive optimum.")
        return

    # Same parent calibration and zero-shock validation as V6.
    print("Solving baseline model...", flush=True)
    base, _, _, goods, _, _ = base_model(mode, {}, EPSILON_L)
    base_annual, base_sectors = project_cge(base, goods, make_solver())
    # Do not overwrite the original V6 Bayesian baseline files.
    base_annual.to_csv(f"{prefix}_baseline_annual.csv")
    base_sectors.to_csv(f"{prefix}_baseline_sectors.csv")
    _, control = project_cge(base, goods, make_solver(), scenario={s: 0.0 for s in goods})
    error = (control["Estimated jobs"] - base_sectors["Estimated jobs"]).abs().max()
    print(f"Zero-shock maximum sector/year error: {error:.6g} jobs", flush=True)
    if (not np.isfinite(error) or error > 0.05
            or not np.allclose(control["Capital stock (£)"], base_sectors["Capital stock (£)"], atol=10.0, rtol=1e-8)
            or not np.allclose(control["Real capital income"], base_sectors["Real capital income"], atol=1e-4, rtol=1e-8)):
        raise RuntimeError("Zero-shock validation failed; grid search stopped.")

    submitted = 0
    session_done = 0
    workers = max(1, min(WORKERS, os.cpu_count() or 1))
    # Only keep a small number of tasks in flight. Stop submitting at deadline,
    # but save all in-flight results before exiting normally.
    try:
        with checkpoint.open("a", encoding="utf-8", newline="") as f, ProcessPoolExecutor(
                max_workers=workers, initializer=grid_worker_init,
                initargs=(mode, EPSILON_L, base_sectors, recipients, RECURRING)) as executor:
            writer = csv.DictWriter(f, fieldnames=columns)
            pending = {}

            def submit_next():
                nonlocal submitted
                if time.monotonic() >= deadline or (MAX_ATTEMPTS_PER_SESSION is not None and submitted >= MAX_ATTEMPTS_PER_SESSION):
                    return False
                item = next(remaining, None)
                if item is None:
                    return False
                idx, allocation = item
                amount = {**donor_changes, **dict(zip(recipients, allocation))}
                future = executor.submit(grid_worker, amount)
                pending[future] = (idx, amount)
                submitted += 1
                return True

            for _ in range(workers):
                if not submit_next():
                    break
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    idx, amount = pending.pop(future)
                    row = {"Scenario": idx, "Run signature": signature, **{s: amount.get(s, 0.0) for s in goods}}
                    try:
                        jobs, capital = future.result()
                        row.update({k: v for k, v in jobs.items() if k != "Allocation"})
                        row[CAPITAL] = capital[CAPITAL]
                        if not np.isfinite([row[c] for c in columns if c not in ("Scenario", "Run signature", "Status", "Error")]).all():
                            raise ValueError("Nonfinite scenario results")
                        row.update(Status="Success", Error="")
                    except BrokenProcessPool:
                        # Infrastructure failure is not an economic failure.
                        # Do not mark these allocations as attempted; resume later.
                        raise
                    except Exception as exc:
                        row.update(Status="Failed", Error=f"{type(exc).__name__}: {exc}".replace("\n", " ").replace("\r", " "))
                    writer.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())
                    rows.append(row)
                    completed.add(idx)
                    session_done += 1
                    elapsed = time.monotonic() - start
                    print(f"Scenario {idx}: {row['Status']} | {len(completed):,}/{total:,} | "
                          f"{elapsed/3600:.2f}h elapsed", flush=True)
                    if row["Status"] == "Failed":
                        print(row["Error"], flush=True)
                    if session_done % RANKINGS_EVERY == 0:
                        export_rankings()
                    submit_next()
    except KeyboardInterrupt:
        print("Interrupted. Already saved rows will be skipped on the next run.", flush=True)
    finally:
        export_rankings()
    if len(completed) < total:
        print("Grid incomplete. Run this same file again to resume from its CSV.")
    else:
        print("All grid allocations attempted. Any failures must be resolved to claim full successful coverage.")


if __name__ == "__main__":
    main()
