import pandas as pd
import numpy as np
import re


delta = 0.3                 # flegg size parameter 
min_jobs = 5                # to prevent supplie chain to be zeroed out by Nomis rounding data to 0
demand = {"C28": 10, "C27": 5, "J62": 5, "C27": 20, "M72": 10}   # chosen sectors and amounts invested in £m
sort_by = "Output - Swindon (£m)"    # column to sort the results by


# 1. employment data
def read_employment(filepath, name):
    df = pd.read_csv(filepath, header=None, usecols=[0, 1], names=["Industry", name])
    df = df.dropna(subset=["Industry"])
    df[name] = df[name].astype(str).str.replace(",", "")
    df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0)
    df["sic"] = df["Industry"].str.extract(r"^(\d{2})")
    df = df.dropna(subset=["sic"])
    return df.groupby("sic")[name].sum()

swindon = read_employment("swindonemplyment.csv", "Swindon")
uk = read_employment("nationaldata.csv", "UK")
swindon[(swindon == 0) & (uk > 0)] = min_jobs    # only where the uk actually has jobs

total_swindon = swindon.sum()
total_uk = uk.sum()
print("Swindon jobs:", total_swindon, " UK jobs:", total_uk)


#national coefficient matrix
raw = pd.read_csv("coefficients.csv", header=None, names=range(120))
labels = raw[0].astype(str).str.strip().tolist()

start = labels.index("A01")
end = labels.index("_T")  # totals row sits just under the last industry
n = end - start

codes = labels[start:end]
names = raw.iloc[start:end, 1].astype(str).str.strip().tolist()
A = raw.iloc[start:end, 2:2 + n].apply(pd.to_numeric, errors="coerce").fillna(0).values

print(n, "industries, biggest column sum is", round(A.sum(axis=0).max(), 3))
assert A.sum(axis=0).max() < 1, "column sums should be below 1, check the slicing"


# 3. location quotient for each industry
def sic_divisions(code):
    #mpas the codes to the 2-digit divisions.
    out = []
    for part in re.split(r"[&,]", code):
        m = re.match(r"\s*[A-Z](\d{2})", part)
        if m:
            out.append(m.group(1))
    return out

lq = np.ones(n)
for i in range(n):
    divs = [d for d in sic_divisions(codes[i]) if d in uk.index]
    s = sum(swindon.get(d, 0) for d in divs)
    u = sum(uk[d] for d in divs)
    if u > 0:
        lq[i] = (s / total_swindon) / (u / total_uk)


#adjusts the nation matrix to reflect regionalisation using fleggs location quotient methodd
# row i = supplier, column j = buyer. a supplier swindon is short of gets scaled down
size = np.log2(1 + total_swindon / total_uk) ** delta
cilq = lq[:, None] / np.maximum(lq[None, :], 1e-9)
np.fill_diagonal(cilq, lq)
A_swindon = A * np.minimum(cilq * size, 1)


# 5. demand vector what industries are being invested in and how much
f = np.zeros(n)
for key, amount in demand.items():
    matches = [i for i in range(n) if key in codes[i]]
    if len(matches) != 1:
        raise ValueError(f"'{key}' matched {len(matches)} codes, be more specific")
    f[matches[0]] += amount


# 6. leontief: x = (I - A)^-1 f
I = np.eye(n)
x_national = np.linalg.solve(I - A, f)
x_swindon = np.linalg.solve(I - A_swindon, f)

results = pd.DataFrame({
    "Code": codes,
    "Industry": names,
    "Demand (£m)": f,
    "Output - national (£m)": x_national,
    "Output - Swindon (£m)": x_swindon,
})
results["Leakage (£m)"] = results["Output - national (£m)"] - results["Output - Swindon (£m)"]
results = results.sort_values(sort_by, ascending=False)

print("\nDemand: £%.1fm" % f.sum())
print("Total output, national: £%.1fm" % x_national.sum())
print("Total output, Swindon:  £%.1fm" % x_swindon.sum())
kept = (x_swindon.sum() - f.sum()) / (x_national.sum() - f.sum())
print("Share of the supply chain kept in Swindon: %.0f%%" % (kept * 100))

print(results.drop(columns="Industry").round(2).to_string(index=False))
results.to_excel("swindon_results.xlsx", index=False)
print("\nsaved swindon_results.xlsx")