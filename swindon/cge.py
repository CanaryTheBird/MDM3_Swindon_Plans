if __name__ == "__main__":

    mode = "reallocate"
    epsilon_L = 0.6

    allocation = {
        "MATL": 0.0,
        "ENG": 0.0,
        "C27": 0.0,
        "TRAN": 0.0,
        "INFO": 0.0,
        "FIN": 15.0,
        "M72": 0.0
    }

    budget = sum(allocation.values())

    sam, _, _, _ = sam_to_std()

    # Create baseline model
    base, _, _, goods, netted, _ = base_model(
        mode, {}, epsilon_L
    )

    # Identify donor sectors
    donors = [i for i in goods if i not in allocation]

    # Withdraw investment proportionally
    donor_weights = sam.loc[donors, "INV"].clip(lower=0)
    donor_weights /= donor_weights.sum()

    amount = {
        i: -budget * donor_weights[i]
        for i in donors
    }

    amount.update(allocation)

    # Convert investment amounts into changes in investment shares
    scenario, _, _, _ = scenario_model(
        amount,
        mode,
        epsilon_L,
        cached=(base, sam, goods, netted)
    )

    # Baseline projection
    print("Running baseline...")

    base_annual, base_sectors = project_cge(
        base,
        goods,
        make_solver()
    )

    # One-off investment
    print("Running one-off investment...")

    one_annual, one_sectors = project_cge(
        base,
        goods,
        make_solver(),
        scenario=scenario,
        shock_year=2027,
        recurring=False
    )

    # Recurring investment
    print("Running recurring investment...")

    rec_annual, rec_sectors = project_cge(
        base,
        goods,
        make_solver(),
        scenario=scenario,
        shock_year=2027,
        recurring=True
    )

    # Calculate employment changes relative to baseline
    one_jobs = (
        one_sectors["Estimated jobs"]
        - base_sectors["Estimated jobs"]
    ).groupby(level="Year").sum()

    rec_jobs = (
        rec_sectors["Estimated jobs"]
        - base_sectors["Estimated jobs"]
    ).groupby(level="Year").sum()

    # Combine results
    results = pd.DataFrame({
        "Baseline jobs": base_sectors[
            "Estimated jobs"
        ].groupby(level="Year").sum(),

        "One-off job change": one_jobs,
        "Recurring job change": rec_jobs
    })

    print(results.round(3))

    results.to_csv("FIN_employment_projection.csv")

    # Plot employment changes
    plt.figure(figsize=(10, 6))

    plt.plot(
        one_jobs.index,
        one_jobs.values,
        marker="o",
        label="One-off investment"
    )

    plt.plot(
        rec_jobs.index,
        rec_jobs.values,
        marker="s",
        label="Recurring investment"
    )

    plt.axhline(0, color="grey", linestyle="--")

    plt.xlabel("Year")
    plt.ylabel("Net additional jobs relative to baseline")
    plt.title("£15m investment reallocation into Finance and Insurance")

    plt.xticks(range(2026, 2032))
    plt.ticklabel_format(axis="y", style="plain", useOffset=False)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()