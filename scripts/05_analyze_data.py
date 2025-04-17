import pandas as pd
from statsmodels.formula.api import ols

# Load the company data
company_data = pd.read_parquet("data/altman_data_v2.parquet")

# Add EU dummy
eu_countries = ["GERMANY", "FRANCE", "ITALY", "SPAIN", "NETHERLANDS", "BELGIUM", "SWEDEN", "AUSTRIA", "IRELAND", "DENMARK", "FINLAND", "PORTUGAL", "GREECE", "CZECH REPUBLIC", "ROMANIA", "HUNGARY", "SLOVAKIA", "BULGARIA", "CROATIA", "SLOVENIA", "ESTONIA", "LATVIA", "LITHUANIA", "CYPRUS", "MALTA", "LUXEMBOURG", "POLAND"]
eg_countries = ["UNITED KINGDOM", "NORWAY", "SWITZERLAND", "TURKEY", "BOSNIA & HERZEGOVINA", "SERBIA", "MACEDONIA", "MONTENEGRO"]

company_data["is_eu"] = company_data["item6026_nation"].str.upper().isin(eu_countries)
company_data["is_eg"] = company_data["item6026_nation"].str.upper().isin(eu_countries  + eg_countries)

# Map in the eligible trbc and nace codes
taxonomy_data = pd.concat(pd.read_excel("/Users/max/Downloads/sustainable-finance-taxonomy-nace-alternate-classification-mapping_en (1).xlsx",  sheet_name=None).values())
nace_mapping = taxonomy_data[["NACE Code", "TRBC  Name"]]

company_data["eligible_trbc"] = company_data["item7041_tr_business_classification"].isin(nace_mapping["TRBC  Name"].dropna().astype(str).unique().tolist())

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
data_x["eligible_binary"] = (data_x["eligibility_score"] > 5).astype(int)

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
data_x.columns
(data_x["year__fiscal_year"].astype(int) > 2021).mean()


# Specify the regression
causal_block = "altman_z_private ~ eligibility_score : is_eu + post_2021 : is_eu + eligibility_score * post_2021 + eligibility_score : post_2021 : is_eu "
mod = ols(causal_block + " + C(item6026_nation, Treatment)  + C(item6011_industry_group, Treatment) + C(year__fiscal_year, Treatment) : C(item6010_general_industry_classification, Treatment)", data=data_x)

# Estimate the regression
res = mod.fit()

# Only print the treatment related coefficients
summ = "\n".join(l for l in res.summary().as_text().splitlines() if "C(" not in l)
print(summ)

# eligibility_score:post_2021:is_eu    c0.3765      0.164      2.300      p0.021       0.056       0.697
# eligible_trbc:post_2021:is_eu         1.1151      1.035      1.077      p0.281      -0.914       3.144

from plotnine import ggplot, aes, geom_bar, labs, position_dodge, scale_x_continuous, theme_bw, theme

plot_data = data[["eligible_trbc", "eligibility_score"]].dropna()
plot_data["eligible_trbc"] = plot_data["eligible_trbc"].map(lambda x: "eligible" if x else "not eligible")
plot_data["eligibility_score"] = plot_data["eligibility_score"].map(lambda x: "eligible" if x > 5 else "not eligible")

pd.crosstab(plot_data["eligibility_score"], plot_data["eligible_trbc"], rownames=['TRBC'], colnames=['LLM']).apply(lambda x: x / sum(x), axis=1)


df = data.groupby(["eligible_trbc", "eligibility_score"]).size().reset_index(name="count")

# Create the ggplot
df['data_plot'] = df['eligibility_score'].astype(int)
df['eligible_trbc'] = df['eligible_trbc'].astype(str)

p = (ggplot(df, aes(x='eligibility_score', y='count', fill='eligible_trbc'))
     + geom_bar(stat='identity', position=position_dodge(width=0.8))
     + scale_x_continuous(breaks=range(df['eligibility_score'].min(), df['eligibility_score'].max() + 1))
     + labs(x='LLM Eligibility Score', y='Count', fill='Eligible TRBC Code')
     + theme_bw()+ theme(legend_position='bottom', aspect_ratio=0.5) )

# In PyCharm REPL, explicitly draw the figure:
fig = p.draw()
fig.show()

p.save("output/charts/llm_vs_trbc.png", bbox_inches="tight")
