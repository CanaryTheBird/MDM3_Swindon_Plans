import re
import numpy as np
import pandas as pd

# 1. employment data
def read_employment(filepath, name):
    df = pd.read_csv(filepath, header=None, usecols=[0, 1], names=["Industry", name])
    df = df.dropna(subset=["Industry"])
    df[name] = df[name].astype(str).str.replace(",", "")
    df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0)
    df["sic"] = df["Industry"].str.extract(r"^(\d{2})")
    df = df.dropna(subset=["sic"])
    return df.groupby("sic")[name].sum()


def load_coefficients(filepath):
    #national coefficient matrix
    raw = pd.read_csv(filepath, header=None, names=range(120))
    labels = raw[0].astype(str).str.strip().tolist()

    start = labels.index("A01")
    end = labels.index("_T")  # totals row sits just under the last industry
    n = end - start

    codes = labels[start:end]
    names = raw.iloc[start:end, 1].astype(str).str.strip().tolist()
    A = raw.iloc[start:end, 2:2 + n].apply(pd.to_numeric, errors="coerce").fillna(0).values

    assert A.sum(axis=0).max() < 1, "column sums should be below 1, check the slicing"
    return A, codes, names

# 3. location quotient for each industry
def sic_divisions(code):
    #maps the codes to the 2-digit divisions.
    out = []
    for part in re.split(r"[&,]", code):
        m = re.match(r"\s*[A-Z](\d{2})", part)
        if m:
            out.append(m.group(1))
    return out


def location_quotients(codes, swindon, uk):
    total_swindon = swindon.sum()
    total_uk = uk.sum()
    lq = np.ones(len(codes))
    for i, code in enumerate(codes):
        divs = [d for d in sic_divisions(code) if d in uk.index]
        s = sum(swindon.get(d, 0) for d in divs)
        u = sum(uk[d] for d in divs)
        if u > 0:
            lq[i] = (s / total_swindon) / (u / total_uk)
    return lq


def flegg(A, lq, regional_jobs, national_jobs, delta=0.3):
    size = np.log2(1 + regional_jobs / national_jobs) ** delta
    cilq = lq[:, None] / np.maximum(lq[None, :], 1e-9)
    np.fill_diagonal(cilq, lq)
    A_swindon = A * np.minimum(cilq * size, 1)
    return A_swindon, size

def load_iot(filepath):
    raw = pd.read_excel(filepath, sheet_name="IOT", header=None)
    labels = raw[0].astype(str).str.strip().tolist()

    start = labels.index("A01")
    end = labels.index("_T")  # totals row sits just under the last industry
    n = end - start

    codes = labels[start:end]
    rows = {}
    for label in ["D1", "GOS & MI", "D29-D39", "P1"]:
        current = labels.index(label)
        rows[label] = raw.iloc[current, 2:2 + n].astype(str).str.replace(",", "").apply(pd.to_numeric, errors="coerce").fillna(0).values

    iot = pd.DataFrame(rows, index=codes)
    return iot, codes