import pandas as pd
import numpy as np

# Altman Z‐score calculation
def altman_z(row: pd.Series) -> float:
    WC = row['item2201_current_assets'] - row['item3101_current_liabilities']
    TA = row['item2300_total_assets']
    RE = row['item3495_retained_earnings']
    EBIT = row['item18191_ebit']
    MVE = row['item7210_market_value_equity']
    TL = row['item3351_total_liabilities']
    Sales = row['item1001_sales']
    # Avoid division by zero
    if TA == 0 or TL == 0:
        return np.nan
    return (1.2 * (WC / TA) +
            1.4 * (RE / TA) +
            3.3 * (EBIT / TA) +
            0.6 * (MVE / TL) +
            1.0 * (Sales / TA))

def altman_z_private(row: pd.Series) -> float:
    WC = row['item2201_current_assets'] - row['item3101_current_liabilities']
    TA = row['item2300_total_assets']
    RE = row['item3495_retained_earnings']
    EBIT = row['item18191_ebit']
    MVE = row['item7210_market_value_equity']
    TL = row['item3351_total_liabilities']
    Sales = row['item1001_sales']
    # Avoid division by zero
    if TA == 0 or TL == 0:
        return np.nan
    return (0.717 * (WC / TA) +
            0.847 * (RE / TA) +
            3.107 * (EBIT / TA) +
            0.420 * (MVE / TL) +
            0.998 * (Sales / TA))


company_data = pd.read_parquet("data/company_data.parquet")
annual_data = pd.read_parquet("data/wrds_fundamentals_annual.parquet")

# Merge fundamentals data
data = pd.merge(company_data, annual_data, left_on="code_company_code", right_on="code_company_code", how="left")
data.dropna(subset=["item2300_total_assets"], inplace=True)

# Add SME indicator
data["sme_strict"] = ((data["item2300_total_assets"] <= 43e6) | (data["item1001_sales"] <= 50e6)) & (data["item7011_number_of_employees"] < 250)
data["sme_financial"] = ((data["item2300_total_assets"] <= 43e6) | (data["item1001_sales"] <= 50e6))

data["sme_strict"] = data["sme_strict"].astype(int)
data["sme_financial"] = data["sme_financial"].astype(int)

# Add credit indicator
data["altman_z"] = data.apply(altman_z, axis=1)
data["altman_z_private"] = data.apply(altman_z_private, axis=1)

data.to_parquet("data/altman_data_v2.parquet")
