import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the company data
company_data = pd.read_parquet("data/altman_data.parquet")

# Add EU dummy
eu_countries = ["GERMANY", "FRANCE", "ITALY", "SPAIN", "NETHERLANDS", "BELGIUM", "SWEDEN", "AUSTRIA", "IRELAND", "DENMARK", "FINLAND", "PORTUGAL", "GREECE", "CZECH REPUBLIC", "ROMANIA", "HUNGARY", "SLOVAKIA", "BULGARIA", "CROATIA", "SLOVENIA", "ESTONIA", "LATVIA", "LITHUANIA", "CYPRUS", "MALTA", "LUXEMBOURG"]
company_data["is_eu"] = company_data["item6026_nation"].str.upper().isin(eu_countries)

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


# Check distribution of eligibility score
data["eligibility_score"].plot.hist()

# Check distribution of altman z-score
data["altman_z"].plot.density()
plt.show()
data["altman_z"].quantile([0.01, 0.99])
data["altman_z"].quantile([0.02, 0.98])
data = data[(data["altman_z"] > data["altman_z"].quantile(0.01)) & (data["altman_z"] < data["altman_z"].quantile(0.99))]
data["altman_z"].plot.density()
plt.show()


data.groupby("is_eu")[["altman_z", "eligibility_score"]].mean()

c_data = data.groupby("item6026_nation")[["altman_z", "eligibility_score"]].mean()

c_data.corr()

# plot barchart
plt.figure(figsize=(10, 20))
sns.barplot(y=c_data.index, x=c_data["altman_z"], orient="h")
plt.xticks(rotation=90)
plt.show()

