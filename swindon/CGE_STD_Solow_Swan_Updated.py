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
    enable_technology=True,
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
    investment_scale = benchmark_investment_pounds / real_investment(base)

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


def run_one_allocation(amount, mode, epsilon_L, base_sectors, recipients, recurring=False):
    scenario, netted, goods, scenario_base = scenario_model(
        amount, mode, epsilon_L, cached=_worker_cache
    )
    solver = make_solver()
    shocked_annual, shocked_sectors = project_cge(
        scenario_base, goods, solver, scenario=scenario,
        shock_year=2027, recurring=recurring,
    )

    jobs_change_by_sector = (
        shocked_sectors["Estimated jobs"] - base_sectors["Estimated jobs"]
    ).unstack("Sector").sort_index()
    net_jobs_by_year = jobs_change_by_sector.sum(axis=1, skipna=False)

    capital_income_change = (
        shocked_sectors["Real capital income"] - base_sectors["Real capital income"]
    )
    total_capital_income_change = (
        capital_income_change.xs(2031, level="Year")
        .sum(skipna=False)
    )

    return (
        {
            "Allocation": amount.copy(),
            "Net jobs in 2031": net_jobs_by_year.loc[2031],
        },
        {
            "Allocation": amount.copy(),
            "Capital income change in 2031": total_capital_income_change,
        },
    )


def main():
    start_time = time.monotonic()
    max_runtime_hours = 100
    deadline = start_time + max_runtime_hours * 3600
    mode = "reallocate"  # "reallocate" or "add_external"
    epsilon_L = 0.6

    sam, goods, netted, data = sam_to_std()

    recurring = False  # True: maintain the changed investment shares from 2027
    max_evaluations = 1000  # Total attempts, including saved results.
    candidate_count = 20000
    worker_limit = 4  # Benchmark 4 versus 8 on your machine.
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

    # Separate checkpoint from the old random-search results.
    checkpoint = Path("bayesian_investment_results.csv")
    signature = hashlib.sha256(
        (sam.to_csv() + repr((mode, epsilon_L, recurring, budget, recipients))
         + re.sub(r"(?m)^    (max_evaluations|worker_limit|candidate_count|max_runtime_hours) = .*$", "", Path(__file__).read_text(encoding="utf-8"))).encode()
    ).hexdigest()
    rows = []
    if checkpoint.exists():
        saved = pd.read_csv(checkpoint)
        # Verify old checkpoints against the exact original Bayesian source.
        import base64, zlib
        legacy_source = zlib.decompress(base64.b64decode('eJzdPe122zaW//0UXGX3lHRo2nLipNVUs8fNR5uNk/jEaZuOjpaHFiGJY4pkSMq24vXTzJvMk+39AECApByn093ZszqdWCKBi4uLi/sNzAPnTNTOuZjnpXCSVZGXdZItnGy9EmUyi1InTc7LqExE5TtJNkvXMb5OMufXJIvzq8q5yssLUVbBDnd28moHYDlZtBLYzB28e3Mavv35Tfjhp/cvjp+fDXxn8O70xdsfTo7P2s/fvD6xH+045mcA7158PH3f7vbDySsblDeijnkViOwyKfNsguhMnbEzGA4UosuoWsLk1M9S7MzLfOUUUY2PJS2cU/jJL6pZUmyCqhCzBMgiX2dxXcrXF6mIyixYROuqSqIsLMp8JqpKtfxRPj/lx+/FooQ/+Rd6B0DbTKQaypuohge+8yzPqjrK6tf0Ws2hiLI4qhz4r4j1s02+yhUd6NUmlzPdzBYiENfRqkhFFVR1DL/DVR6LNIzFXA15Vsdv8NlzMVcwV1FdpHkNZAqKDX4juGnNcJ+VArAMz47fhPM8jUWpIFXRKqzzEAbihgBmqd6J60JBB94rNggxK0ymwh6zPJuty1JkdTBf12ugoF4nptdpnqcvrsVsXeelDzDCWY6zq0W8Y65iktVAf8BSo5anlyJMLgvnwedotoSBYrGzQ1TILkVVr2DIsBI1bo7KhYn4ziLPY9gUSC8YaZWvs1ry3YNnUTpbpwRe98YZiixOEGunWkaIO26Ueimc2TKvRAa4iQy2Wh4QlCuRLJZ1BUwLwwVpPpvIEQev3v4ymAazvNi4HjWt8xpYcqy6BNV6BW/oVTInDJ0x8D6sSwpwAK/BSO8sxKESM6LXbBllC9q2PJ8gqcWqcr2RtQ/VKIgS95w6D8eys24prusyCo35j52bW3oL7CxGd7WTg8sJUstSAN1Amiii7POcvQD4KU5mtev5HUg7vIAC5RT+dom1fWceIc5qrU6i83wNJFgXRQpcF/91XdVA5Ktlkgonh9UpZYcGUOXA3ts48+RaxIGismyEdD45/sEgsESdBg9OznY6z16+nHDnqcSYf8FMgHkFY62w/SlfV2IJu0qNx42kaAaWQimCPOoQtyNz0cwSETspzTQwEUBGYSSK+WQ5dXa71Fp6xCJL5ApuuvQYzfOokuLCNTcBLERRJWmehScSa3O7ZLCHROw7cVRHzNpSJEhWpmbwIk2q2qUf/Jx3THeRoelX7VBmJ0QdehqizfUCnooXzFh+JSRhZ8JFVBmJB86zqEhwr1WivExA4DiAFVG5AG0pgKxzAdIJvvHWTipJMz0qEnrw7PgUNjDwjzsMDiRKD5wz0KsVIH0JMp2mVkWXMBnf+fAZwOfxelYnIMLr6BrHzWLnw0qJrxrExnxeBRLSxzakWKygg+98dFDylSsRJyydinXNsD5emrKK2ytwp9AjAXTW51USkyVggk4qxpPAfBZlDtZENluuovLCgbXKqojwlsiZtsFkcLZA/f3hM/27wn8/0pOP9M/lYGrLqcsIGZ7pOAf7g6R7niG3IkwvuIzStegILNid0JNfImEyUFhvoRth3Lz5fuwc2B07nb/va4EfbBLnqwiwG6OGDd6DrK22NgU+Tc9dxMHrtLHFY3/Phml+TUCHRl32Q9nkOzmuG860FPEabANYL95guDpg94lP6wgXh9cGW8PmDLEJTINaTg6mDe+KT8WneGI2mwaxwPW9BHZyNVInZw6IqYo2BvH/N5XBFCyHHBg6q5N646yzBLQJbC0yNYnHI7AoJCx861xFoJYWyaWEWSEDSRPV4ZWJekYoog2Ja8XHYDQB21a0iXIgGkA4B/Gez5URC2T6tE5QMwuwGvINMzhsf2P/wtx4hX+JSlcvUwJYglmYfBZjfEc4udQepDsphCloqCtYrCSjFqd5BT0uRcMnnkHnopKDnEZltHIN6Fq2glxb19F5KsYfyrXQpH+Wn5/vPc/XixQostTagjc0igtQBRVLJjCmQP4XRHDmH9hS4lpR6wdNTthlsIq4VNSsIrZJnaEPZpMDa2dQXu+xoUEyHkiUIXUPaRQ5vxfXBZrBgEJDy3INk0qj1XkcOasRNUPx566C4tMkASW166yCKC2WEf5CuZCQcgoSSQWDkmjyhMQ928cDw6wca9lMawX2xVbM5RCSSu9FkUZAuobU5+t4YWytCokETLvBLbFdIcv99SG2N5Tx6qzY+urj9ld6Sj0bVY8ZaqYmv6IEMVa7DVk+xGjW0Nc6WsdgJdgWCr7xWrh+AeRZoUFWVXEPiB/7IepFpGaJ389E4LqOgD8+FsAwugWMb7DRruO2zC7P2YMGgCj++RB7wBOSA/3OfgUy340ezRytLZYJvCaAlsG/PANP00WaitYwXx5DyygJRYsfmGFrR+zuamlj8jS5QqUc54x+vCTKbNxBUuRFPfCkCVmtU7S/uENAf3jllE2MihahzJZidhFC12QFg6MBkmS0OVwGYuhrmBNsk/dgqiUr8aIs89KdD34AoGkCamxm6Ih5BBZ6PHJuGEYg0TCgg3mADhd8ux3YfgRi6W+1KXuNVbZ5lYsm7V5l8bKJacjmGTiRIh6Thh/hwEo8k20JukcBYv1ImqFRbWQco7xuhEjEmo8WTJGXB0Fpi+OMrL3wlbNzxvcx53vcNzlWx8IHgIyebvmHWfAETJFPsvkMEFSST7vRIDUuu4KRNpwkD+47xqvr6gKBuc24YxfqET5egjho2WStoAEKIuVLJQb4B867MgZGLTem4U1mE9vTYBK9QFI5eRmzbTITYDA4OZjsCXj0czDopGlDOHEnIAh7HCC4HqqvC/1VriUIA/lq3u+n0sRw5i6Th0WkGmNftmJ9/LCzpAEoQBdofADejZIqimR6B1kc0JJpujUAIUneIqiWMDq2EcVxCGiAVxKlA5PK4pOac8JbSLXS5iFwUIqOXuCckhWEjQAcCY4oNUDJhQCeoZ2qWBA36oxMyxpfJCX8qmo0MfVQbOI2a9VwqESuy6b4IfdpTsIYdYfuJu0gaW1JhpamV9ZdDcnaqMGakXncLkp6fRTyoSTPFnVuI/brSnKKevLGstEMh/73fsZtWvwqWkO++KOHfNjAPpvDL7Usf4jCVKD/cKV5pvizqzSJOjcsT29/nwZVaDdqRMpqFMesK/+an1chRwVdqSZ0J16VkY6qqHgpR0NVFBIhEK4CY7Jgfa+B9TbKQVGmN/SpcFtzHJtVKdAfXRQCQPqNoMDsNjLYyUAMN2/s3Gh6Dk7fv3ozGDmPHh00FuXg5M3xW3g4tJ++Of5wgk+HB+bTZ4dDfHhgPXzx9kd8+PjwyGr5FB5+ZzX8+cMrhHn49MiC+e7tGTx9fGS1/fD++DlCfXrQfszIHj4xH//07uwUHn9rN3719uU7nO7hExOzl68Qwnc2Eu9fHNN0D62np+/fvYSnT46sp2+eHiJUe6zj528Q7LdPLSqc/vwDzbg1uXcffnoPj4+GT2XjW2ZCWlfYcHHwHBb3JfjIwiXnbGxsd2w0GfwHNuX81mDKfc4E5rPcZvG9HaODDAgvyvyqXlKXiUao8e21RDBs+UbK7DvtKEBfqz1naIW3GnlFj9lVaU8lmsP+JLT65gduzBAkVO9MrFlSN95pbWBqhL2+ESwhgO93dnYeOCD18qu9s6soc6LFohQL3MsChEe+SmYYvPyr4ODlnkPpFbBd6Dfb1Ng5rK4w6xXVS3ZoZMgjnHHAdTw8DI8OHoVPDh+FB98e+lYb9qPHw6Oj8LvDp/Y7sJaKdY3dYYOEB/w/brIYg31yIGFl9ENyWQU/ht/yd/IQ4fcj+S4WRYkpQEQf+8hmGxGV1Rja9Ag1iYtTi9kyg8kuNk103gjsKqKwaDo+gFXpBJjkfAxGc1uU2t0ljIETbPLs7gJn7PF0pGvtoQsIT/cd65XEf3L27uTdr3tnvx6/dZ69e/5i6jxP5hRfpMk0rgmy7mvfOSFD6DjQRrBu4YITsTHUFTb2nWOY32ZHP+SJwbPXzQzcY/jnxLNQb3rEr2PSptBGdt6zVgeev24an1BjfHjSPDymhwt4eNzAlcw9Qfg+dfSp5VQH6lFPsg24qcC2onBllqHCofA67Htp7BFXYIJDoCxAXgAUiiAqSSke+NwANuywsSXWhPu4SUkaHr6iaCMiJwrI1HjWYgm/xQpA+wOjeU3YjTu4qjCHaZgo/IJqPcN86502yMAQC8BAzeTYCgkGmtfI0ZnVakegtBAUmk8wwimJ+ZqJwuNvVGD6xH465KfH9tNDvXbNrpQ8Q9a7zOoUYMYJ4HG0xIUMCvPYv92DMbHdBbbDDcUNlLRE86qjsAx7Q6eV6nx24bh//5sHWu+1aXvQwsHDE1PHa3kCL45NtclTk3B+87sD9c0Uml5ILWsjHqiI7eHB4RO0hhVh6yCq6k2Burf2uj0CSvSAc/YbsNbA0hyy3Y7MeK6iCxGy+alc8Pta1dJozSmODdpqFV2HiVKRYID1NqrzlN+Lvae9DYDaWRyWIgVoHBek9iDvLSOYesk58H5dlFEMgiDdKJ+f2/iYogOvGLek/g67UX9H3SBnXmDMH7llovPDnDH0m4Qxx/SY1xkCdegfZKp28DKqorouFWoDHHRgCGYeOYgKrFSQiWFCrJEVPJZqYiLPdIQnjEpjABWcQS7QuuERGCG9vVW6lwfUOb0QDfcwP0dpULky5SJVFwgJqp8B+tai4JXxnaFcHnBfQvkcVvgxPcvLZJGgO5smmBQbt5acQhYN8/jIOjxUCjweChRo0AkDbUz0ctPQjYsFFE7O94iIHTGal5wFxfBMkmHa12/aP6RJ2D5qFV1SEG1yGTRePuZoG7JN7Q4skKHLS1A/wnqHvhIg7MSgwJMZOqEV+JQpDascJpgnrCUYoTB9Vp+cyAgsSEQ8HMRFf8dBax2kApCKwmVIcsyiBuAeUJyQm9kzw4nwGsBkGF433SmzvRTVwHafk8LVE/eZOt6dWdKQme8S1V91kRT4O4nZbOOMWd+Yhc8srLaPGpv5Vr6s1NtqCwqFgQD1gCXWHADaQ8LeY3BeF5U7hRrRrNPFYkjz0x92kCIgzaM4VPK8GhPr9AdLrG1wj4BBLxBgkq8MbbQ/LCM0xgFNAK1p1XVrz2aD4PJvbXZeiuii81Zcz0RRg52CfxIupINn/XhapIJWViPcJ23bSX20RBhrdumyBss1LUXo9y5InCN76v1lBP//9pWm2b32FRELJdRR550WYFp9bNlPXUN3KzPNB8dxVJCNBRujTrK1FYYDv+FG4T8KHs8xFtcwz+2gF67MYqAuS40tf6fIsJWfNFikoR3OwBUhMBykk3E8ZbfsNmE7Smdh5gSM1BDdhTEYhNLfBq9rXZZJtmAZ4rcd+JBM2143XlcPGDHrsEADjP1w5O4e932L938fz185+zoEYHr52zx8oAFskrBx4ondDYf/OL2KNpVO8yua6/grLDiHRDYqm6frdVjJZnm5ojrObkoLSU72gI47ydfEtirSa9Y2kfrV5gqswoG0pVu1KWRZKkvnfEsjLmCRlheVgIydm2TUbkp5Ca8Vy7pt0mKUe24W2SptZCbSVYk8DEf2m1GM5FR7GO1LmnUrut6H4i1nx2+sXBsmZTBNw7wm06pGKhIWDt2XO/gTPL32nDgBLnH5uYhVVFtcg4mLZXJWhc5MZ1PQHhRUVghtxKffZP73pS7C2zz78cU3lclN52xOixrNNqyElhVBVJeDbBipGiH4m9YJVp2UOnvc4We71o7z0lilx2zpKyac2mJRJuHSbhpafaTks1Vu41xsq3Qi7WbXOFmi1Vr97qgy3/mp2MgUrdQbvTKVG/8mc57869wsFjE/uybyu+rXuaijJsi7pWNTy8SO3GRJNSA2kOVWAK3p2zW57RaSCV9Tfr1HFuvoiS1IjeCjLUkxqmGEQE4w+LEnY4YUCdlvfusYzDE7ZBwd82XuhfxD8GYnMiqDc0F1gtPhyBgGGtCxeHRoSIcIK7RD4kIUhFIktus57QUmsOOxobNALjuu1lbUhR7/2WzUULP5xlxuDDy3MCJ5y1ul5S0j2akcAkTwLkWHtq+H0QuWA419lMlAbaOXsU5Gc7mVjqEhEF2/feB8AOGzzhTWFMJp6vWWeFwkBRkWb+CpkCGNuHH6YJZEnn+R0R+gnjFre9v1uiC/LyzyJfZ3OpEH9bm3pX5fO24+AMGrLDbg0Bukx60Uh+MbgxrwsN9kmw9ej29ej4InC2hyMr45kV8NW+LGmBC9BVMQ0O6xAT1WZuhXNKu0rZqGVNjuFsVrMMkrPj+g0xGXWO0Jk53Jc0Z7XCAqa23jGKOzSeT8pqp4E2GUN9Boi8tIli20jYQNae+uWkfxu9V925bPZ8mi4lE3VneOO46IgW2O+mLAlRqpoyHYxg690utt4Vd6aYS+cysaa0WRdzmGLMVmG0dgOySl8+Mvxw7TiG0WBKNo3NOnU8evhjZqwPonymdg2kO11+nkzMD0tlkKmZTtpv3llm+t3F0KXOqJ/mXtLC1Qm9rjFLuSQ0bDeybyF2S4ng6SHF+ig5nT7YHyosLwBtppSBHoj3/o0FQCGJuZ1p7Oij+5GLVncH2MRJsj0pbfhg4WmutEx+8D+0Wh3PmoQrG+OuoWkreeqbSOOZ0WJ9UMnAJhJpjXbFOvK1TfuA1AjMeOzfiNLHpNscpWZtAjPfzQ6KDbn2B4QFwXbtZw6rF6tmidRrPSOSyJPDI1aYIuM2lrnlYfyedmpwn3aph6iilZdtjTvA6Ri1yzHI4zKWAnMWpRdoGEMXPHrTKJTlfdMIjLvMgiF0/6iHo8GbwVNW9r2KZgjQ0HhoAOqryURnXldlvi6csZHnhUIQGZc27QJH/WKPtnzP8XcKCe6GwpmuOQhovhGShOBu/hLwVR+IFMhj10htJ4TRYwDJ1nwIjo+hxXqXLhcYVejPud7xxJgzi6DkAh17DnXRu4rwfrTtYHHZzm5XhQ1UKk5zDTgYYWXS+xLBoTybLRohQbmDY+vUriejlWieVzWHRjDiiGZALVwAobNTjxrzswisGCzMlghyE/U3Xk+FGDXAZ2OdYzNhJ/8ANiUcM/VLhNA3zTHuCb6ehhMJzfsthsNs89sTO22/WmFtf12B0Cfb41nuPTWQ74VuNBPp8DHzhFDi64Gk2SDAkDLHIN9qhI3cExn51FkxKp2KwCNtrIRoiPsAS/c1lRPAeXxO4DZlMq3MHf/zY8WplKWx/TVUPxeU5QlzRDDaNOZhc0bDhHt6N2o+sEZoTrX9UbcJoHRRolGfxcV+IdzdPcBcChAGKxRIN7AzaKDDUhFyMLGz/BuL1yLSG0Am9rY4qSGUY5wSJpSaNqhtcKjNv6zCjxoyl9vehqj9cvEWj43y0EvlJQ0WDbZNM/hsk/SxzxlP7JEqiFxB8idBgmyJlHi9sviBi5rP9UqfJMb5eya8WZAsZxMcYpLdY/RtjI8ONf8hz7OTkXv9MC4MlJicuetbGrwNlrrlFAAJtwRUdjLcYKMInlqffEtK330bV8vwLvmAC43HKPIaIhdxAMj2yiJSuXx9uT/XwJ/6H87em98pUSkH6XWPvOXNXQbjRQKqlZUSq5xPrI/3MfrNn8lYp+dNXhiM4JG5xh8AVWC6kLSaDnayEKPkKqOmMmCyz9dSqcVFyC01Tl+oIYDj1zbxTvM7DhC1RdMBoY6iEXH4V0WkhVWKirNzgYXAnZyG2ds5LBwUWan8N2sCCx+sEzRCH/13fZALun3KB1KCrsOxF1c9s5DdVG3912IErpz3KdhTDFsCHv9lNkhID0EXxMsSVFgidW/Ha6TdVMbSvK14c65MUHrWNsTTTlC+fZuiQ2i7AwLW0WcvErDjyGOgQsfzeh4E4e0pxK2JuU1BlJPeOmYys7aZBKf7MEshGvCM83Ei9L4bdQnrRdeyyVNpeq24DHC9YYTJtduMq189hGYK3P5IKFIxcPUeHgdT+CdMcLGXxDzsNnkbIpCI4KLbNclt3vnlVfhKA7td5WBivQvSzhl4fvbRFcVy7qHZ/FyFh60IZNhUd/rNkaK6ncct28FQ40JPPIvmSmFQ/reBWjzrpQIAffGdGbW/8fHvluO3l0F3U7eHi6uBLVLEsIPi9nXQWE2t36DdxvH6ejnloYULHdk51GwN7jZhe5PCrpImvUKDsBZu6IUKwjebCOpxObukjdl4SBb9zVisfBlsZgUDhDimK1H9aiSXOTj+HDsyNdtKnkJ+af+OCOz6dyfD6H48tzM748EuPzIRifj7Iwl68SVmMwQ1G7DUw8Lo5PjCgnpka5dbtM+hf0CtRBrTcSooalwp9grcr+t3IPxHkm02dSTDW3N+lIKlUE8CNMjJHRqJDUZdAv11ksSxBMPQ+OB14Al1NtCabGm0xRO7hGqEjWo3KAWz3nO++b4n5WwkLdb8Wz0xdcydrieZpHRlkXDOB2YHzvHHhg+2/aN790qA1bexFRPU4zM4JmzO9PDtWnYblAYMgeHlRdtdVGQd65ZWBptv9esSMMZD3vHC3uYvwqq9ZYtkmcsR3rLqrN2rh7cvjdLvn3TYSM+7Saqo2GQajQLpR3cKm7uLr1Gze7uwQDC7xMo0UjgUcgqqgso40G4vXErAGMOZFbff5CYN1DLXidyMlqDtTgFRzgRcT5aq8COQ1OiKpJZ5nRdBnT5X7u4DzaCLpzrxsaDWbVpaozTxZZRJe3jdXVgeAWRIdHTxp9g4fWkYLQy/XA2yhFUbpto9WwSBRJTNvOrFpDCOjDu+XA/feV9580BhZ2ySMaWOn1X9Iko8ouD5ALdv8VJRb8j6YXhvMkFWHo4c0PcYheqgtMlFMUYrCu53vfDoD6AT1T9UZesBTXcQJEV15QiaY8pu3VLm8IGVBpi3V2XlUvFzGPigRpOljbZPB+nTXEHSiZxRCkDKMfk1bLaSA+ufonbP80/fL2f9Ysv7ppYB/nTaXRogyc94IOLShOenb2i7ock2oKqWQAVN0V2vHmlpP0IUz1JgIJOMOAgNGOvcc5mFEgMBAaVV/fpCJzEYR3i2VAl0m+rhyMiKwKYMKBsRlxAMWd5nTnJZ+1sAJmBLEJUKyoDKFZNxw6JOFy7Q4CeG3gSfDs8Lo6UDzwFI9DF58vDWrHs/IKFp6O53KjvsWX10zRSBP+d3BWw2qiRY3644zrZAdN4onUHYfSkKv1RWK9sf9mY5PhpoIbvKe/XGHgfsEYMwdQBpk9RqtSCud7n6Cgoq6a4RYSb6fbS6pVgA0iATV4chVDqLwmxrMFn7c7WTggp1mUBsB6EbBcCM/dx4cgmDTDSsMES1XwUSPHWFDIo48E65K9nwl8FRvhxp6PT+dr2Lju0Mefw/34rpXZho8XxEmZzJYp2F/QBtzriqBTyPOxN1X3guSlYNuvj1vuXu1pswXJSqqEOzOu/MSP3K98bINO+aGCm5TWdQSGLQY8TKbNvtSL2IZroICuDc/Ly0G17FVogDMpgOJXbZnHOhilIeCQYgXqIsC6JPdq79Jz/ownYp40p0kapB/KkewM/RU2kqvYKfzDFZedkPGIID2FNowRN2yD2DKLZuKqBuCqIzgmpU0yBFduESFY59UZQNIoqeY4QWlpuuXkYupxdekFaSJiG29qKi2cOGLhgWV31Lodlc4CmZT5fhthrjDgiYmEHubtFsJYxNpOnO0U/6gOnxJbEl+arNdmz4a4dNGfocLyvLWhO5OgPR37kdqDLU9tf//RXTdy4MARXR6NsV081uU7jwPPQKEpWELTlI4zwozP1V1JJRbn8qURfHwMDH66grACC/wci2yB31kHdlVLs+b2isn8xUfe2guMTQMNOVLdppVdKtElGD7xW2sPc4U9+OhgFweSdMMLJMx5R7NP66RK5EmySY9i3Ib9xlr8LXgriWT1VGXQOGGAgA79BkW12BseeJ0x3M3eJliJKHM9b5+62joQTZBtV2G7fNn12L7cmu5I3eWLr13YVws0WhDw+CB4DBpkPT4Mju6hz5sPF/ujHPRlxSbQOtyokwusYsIK80hjUC6tCQTzpHY/+hv7+WoNS0a3ZhYBWHBk/slVps2IgZCeKuZkBdrkUkhHeLVG4mHSY+8gOBhaLT9jZV/Teh+WAlomKwyIxT6uRgtTkdg9dvGucvez97CKd1EDXxcujHK0+3n3s4fQqk9l7R7iqyKxIRlcpwSOMJrcU+LhAQkKKOHZa7Urqz9RjU++LvFGugwctr14XaQJBg2ouj1NRSo1rj5Vb83QQG7iaqvkoYGO92+H1m3Z6nOHDuzKasCy5i1kqlRc4b1Ln+OwXZEGCO5SH7D0aaVAkiGgfTwh2dd8gm+/J9bEFO8eaqdsbnNMfE01niRnpRyC1WixFZ/WN5UbAIeu0x49+z+mhvqPoNldkIKMWDuKILWX8jf10uIBLdsFNg4wc24Oj4ujb6XTolyFFpgOW0/uJ7x/2kdB0GkNM0jeymlYGQwrQWIUl8OM1CW0fLjPdOt99KdmxZq1p+uhX2ywG7mQPRfguwbUsfH9HrKyOYdRjjtpuHv2B+asxp2UnccF2YzhNtPpiwutu+hL8pVVjrQz59qGsdd4Lj1HFZOStjkpd2QltCeV2yuPHDUmOh4VIzdkfOA9HHaByf+HAh0btV6ixRVf+2RfC7rLGMsW5Hx8xmTLMUhOIKDs2xaZ67euGCE6k8rkx9gSsJfbzUn6dirwK5Sr43xlBvMuXKsJ/53ScRQgl3kHZJuccnoo0I3/3wdXQtpCywYohiLsQXs7wMrjijZsMUIQ7TDWSH+9k3a7uzfJSOaD+M7Eg6Bzfq7LPPjZet4bP+jb+ipJp+YVcNjI3W56w9wCLq91b9iInBxMR3QLlP459aV9ORlOR3KIiX4yvd0OXSqlttcFg6LH1XG4ts+OUO1G9vOMITbWxeBeU2WHcax9Rd8hiOPBlu5fdRZ8O1mBBlmU+Q0x5YN77TXl5Y5UsMcfENLAeGDkATbeloVAwaV0L3zvb2QHGXubqDimvmXwBrYB3SUIy/kNY/fN9NZxjdDm/k1LFt96A3Q51tXSMI0lm2CXdmZhZwtyvyMgoOPY2NccR13VM7mxsrawT8vuvZK3/asF2/pihHGENlPf3rY8rq92P1EzSST9C7FRV1yXo7JTgGguk1ED5zAP3tq1cLeDvo4w625R05Z4lsFwTaF6pz7TKB/V73Z2YB3CEGOdYUirFIaYEQ5DmTfkDPbOfwOlVuZQ')).decode()
        legacy_signature = hashlib.sha256((sam.to_csv() + repr((mode, epsilon_L, recurring, budget, recipients)) + legacy_source).encode()).hexdigest()
        if "Run signature" not in saved or not (saved["Run signature"].eq(signature).all() or saved["Run signature"].eq(legacy_signature).all()):
            raise ValueError("Checkpoint settings/code differ. Rename the old CSV before starting a new run.")
        saved["Run signature"] = signature
        rows = saved.to_dict("records")
        print(f"Resuming with {len(rows)} previous attempts.")

    def save_results():
        frame = pd.DataFrame(rows)
        tmp = checkpoint.with_suffix(".tmp")
        frame.sort_values("Scenario").to_csv(tmp, index=False)
        os.replace(tmp, checkpoint)
        good = frame[frame["Status"] == "Success"]
        for score, filename in [("Net jobs in 2031", "bayesian_jobs_rankings.csv"),
                                ("Capital income change in 2031", "bayesian_capital_rankings.csv")]:
            good.sort_values(score, ascending=False).to_csv(filename, index=False)
        frame[frame["Status"] == "Failed"].to_csv("bayesian_failed_scenarios.csv", index=False)

    rng = np.random.default_rng(42 + len(rows))
    d = len(recipients)
    initial = np.vstack([np.eye(d), np.full((1,d), 1/d),
                         np.random.default_rng(42).dirichlet(np.ones(d), size=4)])
    scores = ["Net jobs in 2031", "Capital income change in 2031"]

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
            gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0)*Matern(length_scale=0.4, nu=2.5),
                                          alpha=1e-6, normalize_y=False, random_state=42)
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
                        if not all(np.isfinite(row[k]) for k in scores):
                            raise ValueError("Nonfinite objective")
                        row.update(Status="Success", Error="")
                    except Exception as exc:
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
        plot_jobs(results)
        plot_money(results)


if __name__ == "__main__":
    main()
