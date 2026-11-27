"""Analyse the V6 exhaustive £1m investment grid without rerunning the CGE.

Usage:
    python cge_grid_sensitivity_analysis.py
    python cge_grid_sensitivity_analysis.py --csv path/to/results.csv --show

All investment amounts are in £ millions. Capital income stays in model units.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SECTORS = ["MATL", "ENG", "C27", "TRAN", "INFO", "FIN", "M72"]
JOBS = "Cumulative job-years 2027-2031"
CAPITAL = "Capital income change in 2031"
OBJECTIVES = [JOBS, CAPITAL]
SHORT = ["employment", "capital_income"]
Y_LABELS = ["Cumulative net job-years (2027–2031)",
            "Real capital income change in 2031 (model units)"]
BUDGET = 15
# Change or extend this list to investigate other two-sector combinations.
PAIRS = [("TRAN", "M72")]


def analyse(csv_file="grid_v6_1m_investment_results.csv",
            output_dir="grid_sensitivity_analysis", show=False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(csv_file)
    required = ["Scenario", "Status", *SECTORS, *OBJECTIVES]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = raw.loc[raw["Status"].eq("Success"), required].copy()
    if df.empty:
        raise ValueError("No successful scenarios")
    if df[SECTORS + OBJECTIVES].isna().any().any():
        raise ValueError("Successful scenarios contain missing allocations or objectives")
    X = df[SECTORS].to_numpy(dtype=float)
    Y = df[OBJECTIVES].to_numpy(dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Nonfinite allocations or objectives")
    if (X < 0).any() or not np.allclose(X.sum(axis=1), BUDGET):
        raise ValueError("Allocations must be nonnegative and sum to £15m")
    if not np.allclose(X, np.round(X)):
        raise ValueError("This script expects integer £1m grid allocations")
    if df.duplicated(SECTORS).any():
        raise ValueError("Duplicate allocations found")
    print(f"Using {len(df):,} successful allocations of {len(raw):,} rows.")
    if len(df) != 54264:
        print("Warning: grid is incomplete; conditional means may not represent the full grid.")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / f"{name}.svg", bbox_inches="tight")
        if not show:
            plt.close(fig)

    # 1. Sectoral sensitivity: average over remaining feasible allocations.
    # These are conditional means, not effects holding the other shares fixed.
    sensitivity = []
    for s in SECTORS:
        g = df.groupby(s, sort=True)[OBJECTIVES].agg(["mean", "std", "min", "max", "count"])
        for amount, row in g.iterrows():
            for objective in OBJECTIVES:
                sensitivity.append({"Sector": s, "Investment (£m)": amount,
                                    "Objective": objective, **row[objective].to_dict()})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(out / "sector_sensitivity.csv", index=False)
    for objective, name, ylabel in zip(OBJECTIVES, SHORT, Y_LABELS):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for s in SECTORS:
            g = sensitivity[(sensitivity.Sector == s) & (sensitivity.Objective == objective)]
            ax.plot(g["Investment (£m)"], g["mean"], marker="o", markersize=3, label=s)
        ax.axhline(0, linestyle="--", linewidth=0.8)
        ax.set(xlabel="Investment allocated to sector (£ million)",
               ylabel="Mean " + ylabel[0].lower() + ylabel[1:],
               title="Investment sensitivity across the £1m grid", xticks=range(BUDGET + 1))
        ax.legend(ncol=2)
        ax.grid(alpha=.2)
        save(fig, f"{name}_sensitivity")

    # 2. Linear mixture regression: f(I) = sum(beta_i * I_i).
    # Do not fit an intercept: sum(I_i)=15 makes it redundant.
    coefficients, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    if rank != len(SECTORS):
        raise ValueError("Investment matrix does not have full column rank")
    fitted = X @ coefficients
    coef = pd.DataFrame(coefficients, index=SECTORS, columns=OBJECTIVES)
    coef.index.name = "Sector"
    coef.to_csv(out / "linear_coefficients.csv")
    print("\nLinear mixture coefficients (per £m):\n", coef.round(6).to_string())

    # Pairwise transfer effects: move £1m FROM the row sector TO the column sector.
    for i, name in enumerate(SHORT):
        transfers = coefficients[None, :, i] - coefficients[:, None, i]
        pd.DataFrame(transfers, index=SECTORS, columns=SECTORS).rename_axis("From / To").to_csv(
            out / f"{name}_transfer_effects.csv")

    # 3. Independent approximation using only the seven single-sector evaluations.
    vertices = []
    vertex_rows = []
    for i, s in enumerate(SECTORS):
        mask = np.isclose(X[:, i], BUDGET) & np.isclose(X.sum(axis=1) - X[:, i], 0)
        if mask.sum() != 1:
            raise ValueError(f"Expected exactly one pure £15m allocation for {s}")
        vertices.append(Y[mask][0])
        vertex_rows.append({"Sector": s, "Scenario": df.loc[mask, "Scenario"].iloc[0],
                            **dict(zip(OBJECTIVES, Y[mask][0]))})
    vertices = np.array(vertices)
    pd.DataFrame(vertex_rows).to_csv(out / "single_sector_outcomes.csv", index=False)
    interpolated = (X / BUDGET) @ vertices
    mixed = X.max(axis=1) < BUDGET

    metrics = []
    predictions = df.copy()
    for method, prediction in [("Full-grid linear fit", fitted),
                               ("Seven single-sector simulations", interpolated)]:
        error = Y - prediction
        for i, objective in enumerate(OBJECTIVES):
            for population, mask in [("All", np.ones(len(df), dtype=bool)), ("Mixed only", mixed)]:
                actual = Y[mask, i]
                e = error[mask, i]
                if not len(e):
                    continue
                sst = np.sum((actual - actual.mean())**2)
                metrics.append({"Method": method, "Population": population,
                                "Objective": objective, "Count": len(e),
                                "R2": 1 - np.sum(e**2)/sst if sst else np.nan,
                                "RMSE": np.sqrt(np.mean(e**2)), "MAE": np.mean(np.abs(e)),
                                "Max absolute error": np.max(np.abs(e))})
            predictions[f"{method}: predicted {objective}"] = prediction[:, i]
            predictions[f"{method}: residual {objective}"] = error[:, i]
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "regression_metrics.csv", index=False)
    predictions.to_csv(out / "predictions_and_residuals.csv", index=False)
    print("\nAccuracy statistics:\n", metrics.to_string(index=False, float_format=lambda v: f"{v:.9g}"))

    # 4. Actual vs predicted outcomes, and residuals for the seven-run approximation.
    for i, (name, ylabel) in enumerate(zip(SHORT, Y_LABELS)):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(Y[:, i], interpolated[:, i], s=5, alpha=.15, rasterized=True)
        lo = min(Y[:, i].min(), interpolated[:, i].min())
        hi = max(Y[:, i].max(), interpolated[:, i].max())
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1, label="Perfect prediction")
        ax.set(xlabel=f"Actual: {ylabel}", ylabel=f"Predicted: {ylabel}",
               title="Prediction from seven single-sector simulations")
        ax.legend()
        ax.grid(alpha=.2)
        save(fig, f"{name}_actual_vs_predicted")

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(fitted[:, i], Y[:, i] - fitted[:, i], s=5, alpha=.15,
                   rasterized=True, label="Full-grid linear fit")
        ax.scatter(interpolated[:, i], Y[:, i] - interpolated[:, i], s=5, alpha=.15,
                   rasterized=True, label="Seven-simulation approximation")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set(xlabel=f"Predicted: {ylabel}", ylabel="Actual − predicted (original units)",
               title=f"Nonlinear residuals: {name.replace('_', ' ')}")
        ax.grid(alpha=.2)
        ax.legend()
        save(fig, f"{name}_residuals")

    # 5. Two-sector conditional heatmaps. Remaining budget is spread across
    # all feasible combinations of the other five sectors, then averaged.
    for a, b in PAIRS:
        if a == b or a not in SECTORS or b not in SECTORS:
            raise ValueError(f"Invalid sector pair: {a}, {b}")
        pair = df.groupby([a, b])[OBJECTIVES].agg(["mean", "count"])
        pair.to_csv(out / f"{a}_{b}_conditional_outcomes.csv")
        for objective, name, ylabel in zip(OBJECTIVES, SHORT, Y_LABELS):
            z = pair[(objective, "mean")].unstack(b).reindex(index=range(16), columns=range(16))
            fig, ax = plt.subplots(figsize=(8, 6.5))
            im = ax.imshow(z.to_numpy().T, origin="lower", extent=(-.5, 15.5, -.5, 15.5),
                           interpolation="nearest", aspect="equal")
            fig.colorbar(im, ax=ax, label="Mean " + ylabel[0].lower() + ylabel[1:])
            ax.set(xlabel=f"{a} investment (£m)", ylabel=f"{b} investment (£m)",
                   title=f"{a} and {b}: conditional {name.replace('_', ' ')}",
                   xticks=range(0, 16, 2), yticks=range(0, 16, 2))
            save(fig, f"{a}_{b}_{name}_heatmap")

    # 6. Pareto frontier of actual outcomes (maximising both objectives).
    ordered = df.sort_values([JOBS, CAPITAL], ascending=False)
    frontier = ordered.loc[ordered[CAPITAL] > ordered[CAPITAL].cummax().shift(fill_value=-np.inf)].sort_values(JOBS)
    frontier.to_csv(out / "pareto_allocations.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[JOBS], df[CAPITAL], s=5, alpha=.12, rasterized=True, label="Tested allocations")
    ax.plot(frontier[JOBS], frontier[CAPITAL], "o-", linewidth=1.2, markersize=3,
            label="Tested Pareto frontier")
    ax.axvline(0, linestyle=":", linewidth=1)
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.set(xlabel=Y_LABELS[0], ylabel=Y_LABELS[1], title="Employment–capital income trade-off")
    ax.legend()
    ax.grid(alpha=.15)
    save(fig, "pareto_frontier")

    print(f"\n{len(frontier)} distinct non-dominated objective pairs.")
    print(f"Results saved in: {out.resolve()}")
    if show:
        plt.show()
    return coef, metrics, predictions, frontier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="grid_v6_1m_investment_results.csv")
    parser.add_argument("--output", default="grid_sensitivity_analysis")
    parser.add_argument("--show", action="store_true", help="Display figures as well as saving them")
    args = parser.parse_args()
    analyse(args.csv, args.output, show=True)
