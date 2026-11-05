import numpy as np
import pandas as pd
from helper import read_employment, location_quotients, flegg, load_iot, sic_divisions, load_imports

"""
    grouped into sectors but some SIC codes seperate due to the sector's heavy reliance on them 
    therefore if LQ > 1.5 and over 0.75% of swindon population employed then it's seperated
    
    Assumption: Each Swindon worker produces the same output, earns the same wage, and generates the same profit 
    as the average UK worker in the same industry (may not be accurate)
    
    Swindon output in industry i = UK output in i * (Swindon jobs in i ÷ UK jobs in i)

"""

# 1. Sector groupings 
# sector groupings are based on the 2-digit SIC divisions,
# but some sectors are split into separate groups because they rely heavily on a
# single division. For example, pharmaceuticals (C21) is separated from chemicals,
# plastics, minerals & metals (MATL) because it is a high-value industry that is important to Swindon.

# dictionary of 19 sectors 
GROUPS = {
    "PRIM": ("Agriculture & mining", [1, 2, 3, *range(5, 10)]),
    "LMAN": ("Food & light manufacturing", [*range(10, 19), 31, 32]),
    "MATL": ("Chemicals, plastics, minerals & metals", [19, 20, 22, 23, 24, 25]),
    "C21":  ("Pharmaceuticals", [21]),
    "ENG":  ("Machinery, electronics & vehicles", [26, 28, 29, 30, 33]),
    "C27":  ("Electrical equipment", [27]),
    "UTIL": ("Energy, water & waste", [*range(35, 40)]),
    "CONS": ("Construction", [41, 42, 43]),
    "TRAD": ("Wholesale & retail", [45, 46, 47]),
    "TRAN": ("Transport & logistics", [*range(49, 54)]),
    "HOSP": ("Hotels & restaurants", [55, 56]),
    "INFO": ("Information & communication", [*range(58, 64)]),
    "FIN":  ("Finance & insurance", [64, 65, 66]),
    "REAL": ("Real estate", [68]),
    "PROF": ("Other professional services", [69, 70, 71, 73, 74, 75]),
    "M72":  ("Scientific R&D", [72]),
    "ADMN": ("Administrative & support", [*range(77, 83)]),
    "PUBL": ("Public admin, education & health", [84, 85, 86, 87, 88]),
    "OTHR": ("Arts & other services", [*range(90, 98)]),
}

for sector, (name, divisions) in GROUPS.items():
    print(sector, name, divisions)

lookup = {}

# makes SIC codes 2 digits (e.g. 1 -> 01)
for group, (name, divisions) in GROUPS.items():
    for division in divisions:
        lookup[f"{division:02d}"] = group

# 2. Employment data

min_jobs = 5                # to prevent supply chain to be zeroed out by Nomis rounding data to 0
swindon = read_employment("swindonemployement.csv", "Swindon")
uk = read_employment("nationaldata.csv", "UK")
swindon[(swindon == 0) & (uk > 0)] = min_jobs    # only where the uk actually has jobs

# 3. Load the input-output table and location quotients
iot_uk, Z_uk, finaldemand_uk, iot_codes = load_iot("iot2023industry.xlsx")

iot_uk = iot_uk.astype(float)
fisim = Z_uk.loc["K64", "L68A"]
Z_uk.loc["K64", "L68A"] = 0
iot_uk.loc["L68A", "GOS & MI"] += fisim

x_uk = iot_uk["P1"].values
A_uk = Z_uk.values / np.where(x_uk > 0, x_uk, 1)
lq = location_quotients(iot_codes, swindon, uk)

# 4. Scale the UK input coefficients to Swindon 

swindon_share = lq * (swindon.sum() / uk.sum())
income_uk = iot_uk["D1"] + iot_uk["GOS & MI"]
jobs_based = []                      
for code in iot_codes:
    jobs_based.append(code != "L68A")     
income_share = (income_uk * swindon_share)[jobs_based].sum() / income_uk[jobs_based].sum()
swindon_share[iot_codes.index("L68A")] = income_share
swindon_iot = iot_uk.mul(swindon_share, axis=0)
x_swindon = swindon_iot["P1"].values
A_swindon, size = flegg(A_uk, lq, swindon.sum(), uk.sum())


# 5. Where Swindon industres buy from the rest of the UK
uk_inputs = A_uk * x_swindon            
local_inputs = A_swindon * x_swindon        
from_rest_of_uk = uk_inputs - local_inputs

# 6. Swindon final demand
# HOH - Households
# GOV - Government 
# INV - Investment

finaldemand_groups = {"HOH": ["P3 S14", "P3 S15"],
             "GOV": ["P3 S13"],
             "INV": ["P51G", "P52", "P53"]}

swindon_finaldemand = pd.DataFrame()                        
for account, columns in finaldemand_groups.items():
    scaled = finaldemand_uk[columns] * income_share                 
    swindon_finaldemand[account] = scaled.sum(axis=1)

local_share = np.minimum(lq, 1)
local_share[iot_codes.index("L68A")] = 1

finaldemand_local = swindon_finaldemand.loc[iot_codes].mul(local_share, axis=0)
finaldemand_rest_of_uk = swindon_finaldemand.loc[iot_codes] - finaldemand_local

# 7. Exports
local_sales = local_inputs.sum(axis=1) + finaldemand_local.sum(axis=1).values  

# If Swindon buys more from an industry than it makes,
# shrink those local sales so they equal output
for i, code in enumerate(iot_codes):
    if local_sales[i] > x_swindon[i]:
        cut = x_swindon[i] / local_sales[i] #cut is the fraction to shrink an industry’s local sales by, so that they come out equal to what Swindon actually makes
        print(f"{code}: Swindon buys £{local_sales[i]:.1f}m locally but only makes £{x_swindon[i]:.1f}m, scaling by {cut:.3f}")
        local_inputs[i, :] = local_inputs[i, :] * cut                    # row i: its sales to Swindon industries
        finaldemand_local.iloc[i, :] = finaldemand_local.iloc[i, :] * cut    # row i: its sales to households, government, investment

# Whatever was cut is now bought from outside Swindon instead
from_rest_of_uk = uk_inputs - local_inputs
finaldemand_rest_of_uk = swindon_finaldemand.loc[iot_codes] - finaldemand_local

# exports = output that isn't sold inside Swindon
local_sales = local_inputs.sum(axis=1) + finaldemand_local.sum(axis=1).values
exports = x_swindon - local_sales

# 8. Sector-level aggregation
sectors = list(GROUPS)
sector_of = []
for code in iot_codes:
    divisions = sic_divisions(code)
    first_division = divisions[0]
    sector = lookup[first_division]
    sector_of.append(sector)
S = pd.crosstab(pd.Series(iot_codes, name="industry"), pd.Series(sector_of, name="sector"))
S = S.reindex(index=iot_codes, columns=sectors, fill_value=0).values

Z_sectors = S.T @ local_inputs @ S                              # what each sector buys from each other sector
x_sectors = S.T @ x_swindon                              # total output of each sector
exports_sectors = S.T @ exports                          # sales to outside Swindon
finaldemand_sectors = S.T @ finaldemand_local.values     # local final demand (HOH, GOV, INV)


valueadded_sectors = S.T @ swindon_iot[["D1", "GOS & MI", "D29-D39"]].values    # wages, profits, production taxes columns for sectors
rest_uk_sectors = S.T @ from_rest_of_uk.sum(axis=0)                             # inputs bought from the rest of the UK
imports_sectors = S.T @ swindon_iot["imports"].values                           # inputs imported from abroad
ptax_sectors = S.T @ swindon_iot["product_tax"].values                          # product taxes paid on inputs


sales_sectors = Z_sectors.sum(axis=1) + finaldemand_sectors.sum(axis=1) + exports_sectors
costs_sectors = Z_sectors.sum(axis=0) + valueadded_sectors.sum(axis=1) + rest_uk_sectors + imports_sectors + ptax_sectors
assert np.allclose(sales_sectors, x_sectors) and np.allclose(costs_sectors, x_sectors), "sector totals don't match output"


finaldemand_codes = list(finaldemand_uk.columns)
uk_imports = load_imports("iot2023product.xlsx", iot_codes, finaldemand_codes)

imports_industry = uk_imports[iot_codes].values * swindon_share     
imports_by_account = {}
for accounts, columns in finaldemand_groups.items():
    imports_for_accounts = uk_imports[columns].sum(axis=1)
    scaled_imports = imports_for_accounts * income_share
    imports_by_account[accounts] = scaled_imports

imports_finaldemand = pd.DataFrame(imports_by_account)

imports_prod_sectors = S.T @ imports_industry @ S           
imports_prod_finaldemand = S.T @ imports_finaldemand.values          

assert np.allclose(imports_prod_sectors.sum(axis=0), imports_sectors), "imports by product don't add up to each sector's imports"
assert np.allclose(imports_prod_finaldemand.sum(axis=0), swindon_finaldemand.loc["imports"].values), "final demand imports don't add up"

resident_wage_share = 1.0
local_capital_share = 1.0

hoh_idx = 0
gov_idx = 1
inv_idx = 2

wages = valueadded_sectors[:, 0].sum()
profits = valueadded_sectors[:, 1].sum()
labour_to_hoh = wages * resident_wage_share
labour_to_out = wages - labour_to_hoh
capital_to_hoh = profits * local_capital_share
capital_to_out = profits - capital_to_hoh

finaldemand_out = finaldemand_rest_of_uk.sum().values + swindon_finaldemand.loc["imports"].values
finaldemand_ptax = swindon_finaldemand.loc["product_tax"].values #ptax - product tax
finaldemand_spend = finaldemand_sectors.sum(axis=0) + finaldemand_out + finaldemand_ptax

gov_income = valueadded_sectors[:, 2].sum() + ptax_sectors.sum() + finaldemand_ptax.sum()

hoh_saving = labour_to_hoh + capital_to_hoh - finaldemand_spend[hoh_idx]
gov_transfer = finaldemand_spend[gov_idx] - gov_income
capital_inflow = finaldemand_spend[inv_idx] - hoh_saving

finaldemand_accounts = ["HOH", "GOV", "INV"]
accounts = sectors + ["LAB", "CAP"] + finaldemand_accounts + ["OUT"]
sam = pd.DataFrame(0.0, index=accounts, columns=accounts)

sam.loc[sectors, sectors] = Z_sectors
sam.loc[sectors, finaldemand_accounts] = finaldemand_sectors
sam.loc[sectors, "OUT"] = exports_sectors

sam.loc["LAB", sectors] = valueadded_sectors[:, 0]
sam.loc["CAP", sectors] = valueadded_sectors[:, 1]
sam.loc["GOV", sectors] = valueadded_sectors[:, 2] + ptax_sectors
sam.loc["OUT", sectors] = rest_uk_sectors + imports_sectors

sam.loc["HOH", "LAB"] = labour_to_hoh
sam.loc["OUT", "LAB"] = labour_to_out
sam.loc["HOH", "CAP"] = capital_to_hoh
sam.loc["OUT", "CAP"] = capital_to_out

sam.loc["OUT", finaldemand_accounts] = finaldemand_out
sam.loc["GOV", finaldemand_accounts] = finaldemand_ptax

sam.loc["INV", "HOH"] = hoh_saving
sam.loc["GOV", "OUT"] = gov_transfer
if capital_inflow >= 0:
    sam.loc["INV", "OUT"] = capital_inflow
else:
    sam.loc["OUT", "INV"] += -capital_inflow

gap = sam.sum(axis=1) - sam.sum(axis=0)
assert np.allclose(gap, 0), f"SAM does not balance:\n{gap[abs(gap) > 1e-6]}"
print(f"SAM balances: {len(accounts)} accounts, total £{sam.values.sum():,.0f}m")

sam.to_csv("swindon_sam.csv")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 0)
print(sam.round(0).astype(int))