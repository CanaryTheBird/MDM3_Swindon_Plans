import numpy as np
import pandas as pd
from helper import read_employment, load_coefficients, location_quotients, flegg, load_iot

"""
    grouped into sectors but some SIC codes seperate due to the sector's heavy reliance on them 
    therefore if LQ > 1.5 and over 0.75% of swindon population employed then it's seperated
    
    Assumption: Each Swindon worker produces the same output, earns the same wage, and generates the same profit 
    as the average UK worker in the same industry (may not be accurate)
    
    Swindon output in industry i = UK output in i * (Swindon jobs in i ÷ UK jobs in i)
    
"""
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

min_jobs = 5                # to prevent supplie chain to be zeroed out by Nomis rounding data to 0
swindon = read_employment("swindonemployement.csv", "Swindon")
uk = read_employment("nationaldata.csv", "UK")
swindon[(swindon == 0) & (uk > 0)] = min_jobs    # only where the uk actually has jobs

A, codes, names = load_coefficients("coefficients.csv")
lq = location_quotients(codes, swindon, uk)

lookup = {}

for group, (name, divisions) in GROUPS.items():
    for d in divisions:
        lookup[f"{d:02d}"] = group

iot, iot_codes = load_iot("iot2023industry.xlsx")

assert iot_codes == codes, "IOT and coefficients list industries in a different order"

share = lq * (swindon.sum() / uk.sum())
swindon_iot = iot.mul(share, axis=0)
