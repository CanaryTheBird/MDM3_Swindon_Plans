from pathlib import Path
import pandas as pd

def sam_folder():
    # Change this path to wherever your SAM is saved.
    sam = pd.read_csv(r"swindon/swindon_sam.csv")

    factors = ["LAB", "CAP"]
    other_accounts = factors + ["HOH", "GOV", "INV", "OUT"]
    sectors = [account for account in sam.index if account not in other_accounts]
    accounts = sectors + other_accounts
    folder = Path(r"swindon/sam_input")
    folder.mkdir(parents=True, exist_ok=True)

    # Three lists identifying the account roles.
    pd.DataFrame({"i": sectors}).to_csv(folder / "set-i-.csv", index=False)
    pd.DataFrame({"h": factors}).to_csv(folder / "set-h-.csv", index=False)
    pd.DataFrame({"u": sam.index}).to_csv(folder / "set-u-.csv", index=False)

    # Complete SAM, with its values unchanged.
    sam.to_csv(folder / "param-sam-.csv", index_label="Account")




def sam_to_std():
    raw = pd.read_csv(r"swindon/swindon_sam.csv", index_col=0)

    factors = ["LAB", "CAP"]
    other_accounts = factors + ["HOH", "GOV", "INV", "OUT"]
    sectors = [account for account in raw.index if account not in other_accounts]
    accounts = sectors + factors + ["HOH", "GOV", "INV", "EXT", "IDT", "TRF"]
    sam_2 = pd.DataFrame(0.0, index=accounts, columns=accounts)

    core = sectors + factors + ["HOH", "GOV", "INV"]
    sam_2.loc[core, core] = raw.loc[core, core]
    # Production taxes can be negative (subsidies). No separate import
    # tariff is recorded in this SAM.
    sam_2.loc["IDT", sectors] = raw.loc["GOV", sectors]
    sam_2.loc["GOV", sectors] = 0
    sam_2.loc["GOV", "IDT"] = sam_2.loc["IDT", sectors].sum()
    sam_2.loc["EXT", sectors] = raw.loc["OUT", sectors]
    sam_2.loc[sectors, "EXT"] = raw.loc[sectors, "OUT"]

    # OUT payments by final users are unclassified by product. Allocate
    # them across products in proportion to each user's local purchases.
    for buyer in ["HOH", "GOV", "INV"]:
        local = raw.loc[sectors, buyer]
        sam_2.loc[sectors, buyer] += raw.at["OUT", buyer] * local / local.sum()
        sam_2.loc["EXT", sectors] += raw.at["OUT", buyer] * local / local.sum()

    # Some gross trade observations would imply negative domestic sales.
    # Net equal amounts of imports and exports in those sectors, leaving
    # a small positive domestic market for Armington/CET calibration.
    output_at_tax_inclusive_prices = sam_2.loc[sectors + factors + ["IDT"], sectors].sum()
    composite_use = sam_2.loc[sectors, sectors + ["HOH", "GOV", "INV"]].sum(axis=1)
    domestic_sales = output_at_tax_inclusive_prices - sam_2.loc[sectors, "EXT"]
    floor = 0.01 * output_at_tax_inclusive_prices.combine(composite_use, min)
    netted = (floor - domestic_sales).clip(lower=0)
    netted = netted.where(domestic_sales <= 0, 0)
    sam_2.loc[sectors, "EXT"] -= netted
    sam_2.loc["EXT", sectors] -= netted

    # Pool net cross-boundary government flows in the external balance.
    # Government saving is tax revenue less government consumption.
    sam_2.at["GOV", "INV"] = 0
    sam_2.at["INV", "GOV"] = sam_2.loc["GOV"].sum() - sam_2.loc[sectors, "GOV"].sum()
    sam_2.at["INV", "EXT"] = sam_2.loc["EXT", sectors].sum() - sam_2.loc[sectors, "EXT"].sum()

    data = {None: {
    "i": {None: sectors},
    "h": {None: factors},
    "u": {None: list(sam_2.index)},
    "sam": {(row, col): sam_2.at[row, col] for row in sam_2.index for col in sam_2.columns},
    }}

    return sam_2, sectors, netted, data


