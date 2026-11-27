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
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
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


def main():
    start_time = time.monotonic()
    max_runtime_hours = 8
    deadline = start_time + max_runtime_hours * 3600
    mode = "reallocate"  # Reallocation only in this version.
    epsilon_L = 0.6

    sam, goods, netted, data = sam_to_std()

    recurring = False  # True: maintain the changed investment shares from 2027
    diagnostics_only = False  # True: produce baseline diagnostics and stop.
    max_evaluations = 500  # Test first; increase after checking the new baseline.
    candidate_count = 20000
    worker_limit = 4  # Test first; increase to 4 after validation.
    budget = 15.0
    recipients = ["MATL", "ENG", "C27", "TRAN", "INFO", "FIN", "M72"]
    missing = set(recipients) - set(goods)
    if missing:
        raise ValueError(f"Missing recipient sectors: {missing}")
    donors = [sector for sector in goods if sector not in recipients]

    # Fund the reallocation proportionally to benchmark investment.
    donor_changes = {}
    if mode == "reallocate":
        donor_investment = sam.loc[donors, "INV"].astype(float)
        if (donor_investment < 0).any():
            raise ValueError("Negative benchmark donor investment; check SAM.")
        donor_total = donor_investment.sum()
        if donor_total < budget or donor_total <= 0:
            raise ValueError("Insufficient benchmark donor investment.")
        donor_changes = (-budget * donor_investment / donor_total).to_dict()

    def allocation_from_weights(weights):
        return {**dict(zip(recipients, budget * np.asarray(weights))),
                **donor_changes}

    if diagnostics_only:
        print("Solving baseline model for diagnostics...")
        base, _, _, goods, _, _ = base_model(mode, {}, epsilon_L)
        base_annual, base_sectors = project_cge(base, goods, make_solver())
        baseline_diagnostics(base_annual, base_sectors)
        return

    # Separate checkpoint because the employment objective has changed.
    checkpoint = Path("bayesian_v6_investment_results.csv")
    # The sectoral closure changes the model, so old scores are incompatible.
    source = Path(__file__).read_text(encoding="utf-8")
    signature = hashlib.sha256(
        (sam.to_csv() + repr((mode, epsilon_L, recurring, budget, recipients))
         + re.sub(r"(?m)^    (max_evaluations|worker_limit|candidate_count|max_runtime_hours|diagnostics_only) = .*$", "", source)).encode()
    ).hexdigest()
    rows = []
    if checkpoint.exists():
        saved = pd.read_csv(checkpoint)
        if "Run signature" not in saved or not saved["Run signature"].eq(signature).all():
            raise ValueError("Checkpoint settings/code differ. Rename the old CSV.")
        saved["Run signature"] = signature
        rows = saved.to_dict("records")
        print(f"Resuming with {len(rows)} previous attempts.")

    def save_results():
        frame = pd.DataFrame(rows)
        tmp = checkpoint.with_suffix(".tmp")
        frame.sort_values("Scenario").to_csv(tmp, index=False)
        os.replace(tmp, checkpoint)
        good = frame[frame["Status"] == "Success"]
        for score, filename in [("Cumulative job-years 2027-2031", "bayesian_v6_jobs_rankings.csv"),
                                ("Capital income change in 2031", "bayesian_v6_capital_rankings.csv")]:
            good.sort_values(score, ascending=False).to_csv(filename, index=False)
        frame[frame["Status"] == "Failed"].to_csv("bayesian_v6_failed_scenarios.csv", index=False)

    rng = np.random.default_rng(42 + len(rows))
    d = len(recipients)
    initial = np.vstack([np.eye(d), np.full((1,d), 1/d),
                         np.random.default_rng(42).dirichlet(np.ones(d), size=4)])
    scores = ["Cumulative job-years 2027-2031", "Capital income change in 2031"]

    def propose(count):
        attempted = [np.array([r[i] for i in recipients], float)/budget for r in rows]
        chosen = []
        def unseen(w):
            return all(np.linalg.norm(w-v) > 1e-6 for v in attempted + chosen)
        for w in initial:
            if len(chosen) == count:
                return chosen
            if unseen(w):
                chosen.append(w)
        good = [r for r in rows if r["Status"] == "Success" and
                all(np.isfinite(float(r[k])) for k in scores)]
        if len(good) < 5:
            while len(chosen) < count:
                w = rng.dirichlet(np.ones(d))
                if unseen(w): chosen.append(w)
            return chosen
        X = np.array([[r[i]/budget for i in recipients] for r in good])
        pool = np.vstack([rng.dirichlet(np.full(d,a), size=candidate_count//3)
                          for a in (0.15, 1., 4.)])
        # Include local perturbations around each objective's best observed point.
        for score in scores:
            best = X[np.argmax([r[score] for r in good])]
            pool = np.vstack([pool, rng.dirichlet(0.1 + 30*best, size=1000)])
        acquisition = []
        for score in scores:
            y = np.array([r[score] for r in good], float)
            scale = max(np.std(y), 1e-10)
            y = (y-y.mean())/scale
            # BEGIN GP SEARCH SETTINGS
            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e3)
                ) * Matern(
                    length_scale=0.4,
                    length_scale_bounds=(0.01, 5.0),
                    nu=2.5
                ),
                alpha=1e-4,
                normalize_y=False,
                n_restarts_optimizer=2,
                random_state=42
            )
            # END GP SEARCH SETTINGS
            gp.fit(X,y)
            mu, sd = gp.predict(pool, return_std=True)
            improvement = mu-y.max()-0.01
            z = improvement/np.maximum(sd,1e-12)
            ei = improvement*ndtr(z)+sd*np.exp(-0.5*z*z)/np.sqrt(2*np.pi)
            acquisition.append(ei)
        while len(chosen) < count:
            # Alternate objectives; discourage near-duplicate parallel proposals.
            ei = acquisition[(len(rows)+len(chosen))%2].copy()
            for v in attempted + chosen:
                dist = np.linalg.norm(pool-v,axis=1)
                ei *= np.minimum(1.,dist/0.12)
                ei[dist<=1e-6] = -np.inf
            idx = int(np.argmax(ei))
            if not np.isfinite(ei[idx]):
                w = rng.dirichlet(np.ones(d))
                if unseen(w): chosen.append(w)
            else:
                chosen.append(pool[idx])
        return chosen

    if len(rows) < max_evaluations and time.monotonic() < deadline:
        print("Solving baseline model...")
        base, _, _, goods, _, _ = base_model(mode, {}, epsilon_L)
        base_annual, base_sectors = project_cge(base, goods, make_solver())
        baseline_diagnostics(base_annual, base_sectors)
        # Zero-shock validation before expensive optimisation.
        _, control = project_cge(base, goods, make_solver(), scenario={i: 0.0 for i in goods})
        error = (control["Estimated jobs"] - base_sectors["Estimated jobs"]).abs().max()
        print(f"Zero-shock maximum sector/year error: {error:.6g} jobs")
        if (not np.isfinite(error) or error > 0.05
                or not np.allclose(control["Capital stock (£)"], base_sectors["Capital stock (£)"], atol=10.0, rtol=1e-8)
                or not np.allclose(control["Real capital income"], base_sectors["Real capital income"], atol=1e-4, rtol=1e-8)):
            raise RuntimeError("Zero-shock validation failed; optimisation stopped.")
        max_workers = min(worker_limit, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers,
                                 initializer=initialise_worker,
                                 initargs=(mode, epsilon_L)) as executor:
            while len(rows) < max_evaluations and time.monotonic() < deadline:
                weights = propose(min(max_workers, max_evaluations-len(rows)))
                if time.monotonic() >= deadline:
                    break
                first = max([int(r["Scenario"]) for r in rows], default=0)+1
                futures = {}
                for idx,w in enumerate(weights, first):
                    if time.monotonic() >= deadline:
                        break
                    amount = allocation_from_weights(w)
                    future = executor.submit(run_one_allocation, amount, mode,
                                             epsilon_L, base_sectors, recipients, recurring)
                    futures[future] = (idx,amount)
                for future in as_completed(futures):
                    idx,amount = futures[future]
                    row = {"Scenario":idx, "Run signature":signature,
                           **{i:amount.get(i,0.) for i in goods}}
                    try:
                        jobs,capital = future.result()
                        row.update({scores[0]:jobs[scores[0]], scores[1]:capital[scores[1]]})
                        row.update({k: v for k, v in jobs.items() if k != "Allocation"})
                        if not all(np.isfinite(row[k]) for k in scores):
                            raise ValueError("Nonfinite objective")
                        row.update(Status="Success", Error="")
                    except Exception as exc:
                        print(f"Scenario {idx} error: {type(exc).__name__}: {exc}", flush=True)
                        row.update({scores[0]:np.nan,scores[1]:np.nan,
                                    "Status":"Failed","Error":str(exc)})
                    rows.append(row)
                    save_results()
                    elapsed = (time.monotonic() - start_time) / 3600
                    print(f"Scenario {idx}: {row['Status']} ({len(rows)}/{max_evaluations}) | {elapsed:.2f}h elapsed", flush=True)
    if time.monotonic() >= deadline:
        print("Time limit reached. No further scenarios will be submitted.")
    if not rows:
        return
    save_results()
    good = [r for r in rows if r["Status"] == "Success"]
    if good:
        results = [{"Allocation":{i:r[i] for i in goods},
                    **{k:r[k] for k in scores}} for r in good]
        for score in scores:
            best = max(results,key=lambda r:r[score])
            print(f"Best tested {score}: {best[score]}")
            print({i:best["Allocation"][i] for i in recipients})
        # Reproduce the two reported best allocations and export their paths.
        # If an old checkpoint is already complete, recreate the common base.
        base, _, _, goods, _, _ = base_model(mode, {}, epsilon_L)
        _, benchmark_sectors = project_cge(base, goods, make_solver())
        for objective, tag in zip(scores, ("jobs", "capital")):
            best = max(results, key=lambda r:r[objective])
            try:
                annual, sectors = project_cge(base, goods, make_solver(),
                    scenario=best["Allocation"], shock_year=2027, recurring=recurring)
                annual.to_csv(f"v6_best_{tag}_annual.csv")
                sectors.to_csv(f"v6_best_{tag}_sectors.csv")
                delta_jobs = (sectors["Estimated jobs"] - benchmark_sectors["Estimated jobs"]).groupby(level="Year").sum()
                delta_capital = (sectors["Real capital income"] - benchmark_sectors["Real capital income"]).xs(2031, level="Year").sum()
                rechecked = delta_jobs.loc[2027:2031].sum() if tag == "jobs" else delta_capital
                print(f"Best {tag} objective recheck: {rechecked:.8g} (saved {best[objective]:.8g})")
                if not np.isclose(rechecked, best[objective], rtol=1e-6, atol=0.05 if tag == "jobs" else 1e-4):
                    print(f"WARNING: best {tag} result was not reproduced; investigate before using it.")
            except Exception as exc:
                print(f"Best {tag} trajectory recheck failed: {exc}", flush=True)
        plot_jobs(results)
        plot_money(results)


if __name__ == "__main__":
    main()
