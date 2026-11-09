import re
import numpy as np
import pandas as pd

# 1. employment data
def read_employment(filepath, name):
    
    """
    Reads Nomis employment csv and adds the jobs up
    by 2-digit SIC division.

    Args:
        filepath (str): Path to the Nomis csv
        name (str): Name for the jobs column, e.g. "Swindon" or "UK"

    Returns:
        jobs (pandas Series): Jobs per division, indexed "01", "02", ...
    """
    
    df = pd.read_csv(filepath, header=None, usecols=[0, 1], names=["Industry", name])
    df = df.dropna(subset=["Industry"])
    df[name] = df[name].astype(str).str.replace(",", "")
    df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0)
    df["sic"] = df["Industry"].str.extract(r"^(\d{2})")
    df = df.dropna(subset=["sic"])
    return df.groupby("sic")[name].sum()


def load_coefficients(filepath):
    
    """
    This function reads the national input coefficient matrix.
    Not used by sam.py, which works out A from the 2023 table instead.

    Args:
        filepath (str): Path to coefficients.csv

    Returns:
        A (Numpy array): Input coefficients, row = supplier, column = buyer
        codes (list): Industry codes, in table order
        names (list): Industry names, in table order
    """
    
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
    
    """
    Maps the SIC codes to the 2-digit divisions.

    Args:
        code (str): SIC code

    Returns:
        list: List of 2-digit SIC divisions
    """
    
    out = []
    for part in re.split(r"[&,]", code):
        m = re.match(r"\s*[A-Z](\d{2})", part)
        if m:
            out.append(m.group(1))
    return out


def location_quotients(codes, swindon, uk):
    """
    Works out the location quotient for each industry:
    LQ = (Swindon share of jobs) / (National share of jobs)

    Args:
        codes (list): Industry codes from the input-output table
        swindon (pandas Series): Swindon jobs per 2-digit division
        uk (pandas Series): UK jobs per 2-digit division

    Returns:
        lq (Numpy array): Location quotient for each industry, in the order of codes
    """
    
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
    
    """
    Uses Flegg's location quotient to scale the national
    input coefficients down to the part bought inside the region.

    Args:
        A (Numpy array): National input coefficients, row = supplier, column = buyer
        lq (Numpy array): Location quotient for each industry
        regional_jobs (float): Total jobs in the region
        national_jobs (float): Total jobs in the UK
        delta (float): How strongly region size reduces local buying

    Returns:
        A_swindon (Numpy array): Regional input coefficients
        size (float): Size factor, smaller for smaller regions
    """
    
    size = np.log2(1 + regional_jobs / national_jobs) ** delta
    cilq = lq[:, None] / np.maximum(lq[None, :], 1e-9)
    np.fill_diagonal(cilq, lq)
    A_swindon = A * np.minimum(cilq * size, 1)
    return A_swindon, size

def load_iot(filepath):
    
    """
    Readsthe 2023 industry-by-industry input-output table
    (domestic use, basic prices, £m) from the "IOT" sheet.

    Args:
        filepath (str): Path to iot2023industry.xlsx

    Returns:
        iot (pandas DataFrame): One row per industry: wages, profits,
            production taxes, output, imports and product taxes
        Z (pandas DataFrame): Intermediate flows, row = supplier, column = buyer
        fd (pandas DataFrame): Final demand (households, government, investment)
            plus imports and product taxes as two extra rows
        codes (list): The 104 industry codes, in table order
    """
    
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

    text = raw[1].astype(str).str.strip().tolist()
    for key, label in [("imports", "Use of imported products, cif"),
                    ("product_tax", "Taxes less subsidies on products")]:
        current = text.index(label)
        rows[key] = raw.iloc[current, 2:2 + n].astype(str).str.replace(",", "").apply(pd.to_numeric, errors="coerce").fillna(0).values
    
    iot = pd.DataFrame(rows, index=codes)
    Z = raw.iloc[start:end, 2:2 + n].apply(pd.to_numeric, errors="coerce").fillna(0)
    Z.index, Z.columns = codes, codes
    
    header = raw.iloc[3].astype(str).str.strip().tolist()
    fd_codes = ["P3 S14", "P3 S15", "P3 S13", "P51G", "P52", "P53"]
    fd_cols = [header.index(c) for c in fd_codes]

    fd = raw.iloc[start:end, fd_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    fd.index, fd.columns = codes, fd_codes
    fd.loc["imports"] = pd.to_numeric(raw.iloc[text.index("Use of imported products, cif"), fd_cols]).values
    fd.loc["product_tax"] = pd.to_numeric(raw.iloc[text.index("Taxes less subsidies on products"), fd_cols]).values
    return iot, Z, fd, codes

def load_imports(filepath, codes, fd_codes):

    """
    Reads the imports use table (product by industry) from the
    "Imports use pxi" sheet, to split each buyer's imports by product.

    Args:
        filepath (str): Path to iot2023product.xlsx
        codes (list): Industry codes from load_iot, to check the order matches
        fd_codes (list): Final demand column codes to keep

    Returns:
        M (pandas DataFrame): Imports, row = product, column = buyer
            (the 104 industries, then the final demand columns)
    """

    raw = pd.read_excel(filepath, sheet_name="Imports use pxi", header=None)
    labels = raw[0].astype(str).str.strip().tolist()
    header = raw.iloc[3].astype(str).str.strip().tolist()

    start = labels.index("CPA_A01")
    n = len(codes)

    row_codes = [c.replace("CPA_", "") for c in labels[start:start + n]]
    assert [sic_divisions(c) for c in row_codes] == [sic_divisions(c) for c in codes], "imports table is in a different order"

    cols = list(range(2, 2 + n)) + [header.index(c) for c in fd_codes]
    M = raw.iloc[start:start + n, cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    M.index, M.columns = codes, codes + fd_codes
    return M