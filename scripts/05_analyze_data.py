import pandas as pd
from statsmodels.formula.api import ols

# Load the company data
company_data = pd.read_parquet("data/altman_data_v2.parquet")

# Add EU dummy
eu_countries = ["GERMANY", "FRANCE", "ITALY", "SPAIN", "NETHERLANDS", "BELGIUM", "SWEDEN", "AUSTRIA", "IRELAND", "DENMARK", "FINLAND", "PORTUGAL", "GREECE", "CZECH REPUBLIC", "ROMANIA", "HUNGARY", "SLOVAKIA", "BULGARIA", "CROATIA", "SLOVENIA", "ESTONIA", "LATVIA", "LITHUANIA", "CYPRUS", "MALTA", "LUXEMBOURG", "POLAND"]
eg_countries = ["UNITED KINGDOM", "NORWAY", "SWITZERLAND", "TURKEY", "BOSNIA & HERZEGOVINA", "SERBIA", "MACEDONIA", "MONTENEGRO"]

company_data["is_eu"] = company_data["item6026_nation"].str.upper().isin(eu_countries)
company_data["is_eg"] = company_data["item6026_nation"].str.upper().isin(eu_countries  + eg_countries)

# Load the eligibility data
eligibility_data = pd.read_csv("data/eligibility_frame.csv", index_col=0)

# Deserialize the eligibility data
eligibility_data["eligibility"] = eligibility_data["eligibility"].apply(lambda x: eval(x) if not pd.isna(x) else None)
eligibility_data["activity"] = eligibility_data["activity"].apply(lambda x: eval(x) if not pd.isna(x) else None)

# Extract the eligibility score and eligible activity
eligibility_data["eligibility_score"] = eligibility_data["eligibility"].apply(lambda x: x.get("eligibility", 0) if x else 1)
eligibility_data["eligible_activity"] = eligibility_data["activity"].apply(lambda x: x.get("id", None) if not pd.isna(x) else None)

# Rename the company code col
eligibility_data["code_company_code"] = eligibility_data.pop("company_code")

# Only keep the relevant columns
eligibility_data = eligibility_data[["code_company_code", "eligibility_score", "eligible_activity"]]

data = pd.merge(eligibility_data, company_data, on="code_company_code", how="left")


# Run regression on Altman Z-score
data_x = data.copy()

# Add post 2020 dummy - this is when the policy break appears
data_x["post_2021"] = data_x["year__fiscal_year"] > 2021

# Filter out companies for which we could not determine eligibility status
data_x = data_x[data_x["eligibility_score"] != 0]

# Use a symmetric time window around the policy break, i.e. 2015 to 2025
data_x = data_x[data_x["year__fiscal_year"] >= 2015]
data_x["year__fiscal_year"] = data_x["year__fiscal_year"].astype(str)

# Convert bool to int for regression
data_x[data_x.select_dtypes(bool).columns] = data_x.select_dtypes(bool).astype(int)

# Clip the altman z-score to reasonable values
data_x = data_x[(data_x["altman_z"] > -100) & (data_x["altman_z"] < 100)]
data_x = data_x[(data_x["altman_z_private"] > -100) & (data_x["altman_z_private"] < 100)]

# Drop nations which are colinear with the industry
nation_industry_counts = data_x.groupby("item6026_nation")["item6011_industry_group"].nunique()
data_x = data_x[data_x["item6026_nation"].isin(nation_industry_counts[nation_industry_counts > 1].index)]

# Only keep companies which are SMEs according to the official EU definition
data_x = data_x[data_x["sme_strict"].astype(bool)]

# Specify the regression
causal_block = "altman_z_private ~ eligibility_score +  eligibility_score : is_eu + post_2021 : is_eu + eligibility_score : post_2021 + eligibility_score : post_2021 : is_eu "
mod = ols(causal_block + " + C(item6026_nation, Treatment)  + C(item6011_industry_group, Treatment) + C(year__fiscal_year, Treatment) : C(item6010_general_industry_classification, Treatment)", data=data_x)

# Estimate the regression
res = mod.fit()

# Only print the treatment related coefficients
summ = "\n".join(l for l in res.summary().as_text().splitlines() if "C(" not in l)
print(summ)
