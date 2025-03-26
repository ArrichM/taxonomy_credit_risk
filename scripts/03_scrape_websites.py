import pandas as pd
import trafilatura
from multiprocessing.dummy import Pool as ThreadPool
import tqdm
import pydantic
import requests

from taxonomy_credit_risk.classification.taxonomy import TaxonomyClient, Compliance, Activity
from taxonomy_credit_risk.common_crawl import ScrapingClient


class TaxonomyResult(pydantic.BaseModel):
    company_code: float
    url: str
    text: str = None
    activity: Activity = None
    compliance: Compliance = None

# Load the classification client
# taxonomy_client  = TaxonomyClient.from_path(taxonomy_path="data/taxonomy.xlsx")
taxonomy_client  = TaxonomyClient.from_chromadb_client_path(taxonomy_path="data/taxonomy.xlsx")

scraping_client = ScrapingClient()

# Load the company data
data = pd.read_parquet("data/altman_data.parquet")

# Only load sme data that is available for 2020 and 2021
data["has_2020_and_2021"] = data.groupby("code_company_code")["year__fiscal_year"].transform(lambda x: (x == 2020).any() and (x == 2021).any())
data = data[data["has_2020_and_2021"]]
data = data.loc[data["sme_financial"].astype(bool)]

# Add EU dummy
eu_countries = ["GERMANY", "FRANCE", "ITALY", "SPAIN", "NETHERLANDS", "BELGIUM", "SWEDEN", "AUSTRIA", "IRELAND", "DENMARK", "FINLAND", "PORTUGAL", "GREECE", "CZECH REPUBLIC", "ROMANIA", "HUNGARY", "SLOVAKIA", "BULGARIA", "CROATIA", "SLOVENIA", "ESTONIA", "LATVIA", "LITHUANIA", "CYPRUS", "MALTA", "LUXEMBOURG"]
data["is_eu"] = data["item6026_nation"].str.upper().isin(eu_countries)

urls_europe = data[data["is_eu"]].drop_duplicates("code_company_code")[["item6030_internet_address", "code_company_code"]].dropna()
urls_control = data[~data["is_eu"]].drop_duplicates("code_company_code")[["item6030_internet_address", "code_company_code"]].dropna()
urls = urls_europe

pages = {}
def load_page(url: str):


    for protocol in ["http://", "https://"]:
        try:
            address = url.split("://")[1].strip("/")

            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
            }

            content = requests.get(protocol + address, headers=headers, timeout=5, verify=False).text
            parsed = trafilatura.extract(content, with_metadata=True, favor_recall=True)
            pages[url] = parsed
            return parsed
        except:
            pass

    print(f"Failed to load {url}")
    return  None

with ThreadPool(8) as pool:
    urls["page_text"] = list(tqdm.tqdm(pool.imap(load_page, urls["item6030_internet_address"])))

# urls["page_text"] = urls["item6030_internet_address"].map(pages)

urls.dropna(subset=["page_text"], inplace=True)

results = [TaxonomyResult(company_code=code, url=url, text=page_text) for code, url, page_text in zip(urls["code_company_code"], urls["item6030_internet_address"], urls["page_text"])]

all_activities = []
for res in tqdm.tqdm(results):
    scored_pages, activities = taxonomy_client.get_activities_and_pages([res.text])
    all_activities.append(activities)


classified_activities = taxonomy_client.classify_activity(texts=[r.text for r in results], activities=all_activities)
for i, result in enumerate(results):
    result.activity = classified_activities[i]

results_to_classify = [result for result in results if result.activity is not None]

compliances = taxonomy_client.classify_compliance(texts=[r.text for r in results_to_classify], activities=[r.activity for r in results_to_classify])
for i, result in enumerate(results_to_classify):
    result.compliance = compliances[i]


compliance_frame = pd.DataFrame([r.model_dump() for r in results_to_classify])
compliance_frame = compliance_frame.dropna(subset=["compliance"])
# compliance_frame.to_csv("data/compliance_frame_controll.csv")
compliance_frame = pd.read_csv("data/compliance_frame_europe.csv", index_col=0)
compliance_frame["compliance"] = compliance_frame["compliance"].apply(eval)

n = 226
compliance_frame.iloc[n].compliance["dimensions"][0]["eligibility"]
compliance_frame.iloc[n].compliance["dimensions"][0]["reasoning_eligibility"]
compliance_frame.iloc[n].url


compliance_frame["eligibility"] = compliance_frame["compliance"].apply(lambda x: max([dim.get("eligibility", 0) for dim in x["dimensions"]]))
compliance_frame["alignment"] = compliance_frame["compliance"].apply(lambda x: max([dim.get("alignment", 0) for dim in x["dimensions"]]))

analysis_frame = compliance_frame[["company_code", "eligibility", "alignment"]]

analysis_frame["code_company_code"] = analysis_frame["company_code"]

data_analysis = pd.merge(analysis_frame, data, on="code_company_code", how="left")
data_analysis["post_2020"] = data_analysis["year__fiscal_year"] >= 2020

data_pre_2020 = data_analysis[data_analysis["year__fiscal_year"] < 2020]
data_post_2020 = data_analysis[data_analysis["year__fiscal_year"] >= 2020]

# Analyze Altman Z-score
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLS
from patsy.contrasts import Treatment
from statsmodels.formula.api import ols

# Run regression on Altman Z-score
features = ["eligibility", "item7011_number_of_employees", "item6011_industry_group", "is_eu", "post_2020"]
data_x = data_analysis.copy()[features + ["altman_z"]]
data_x = data_x[data_x["eligibility"] != 0]
data_x["eligibility"] = data_x["eligibility"] > 1
data_x = data_x.dropna().astype(float)

data_x.loc[data_x["is_eu"].astype(bool)]["eligibility"].mean()

mod = ols("altman_z ~ eligibility * post_2020 + item7011_number_of_employees + C(item6011_industry_group, Treatment)", data=data_x.loc[~data_x["is_eu"].astype(bool)])
res = mod.fit()
summ = "\n".join(l for l in res.summary().as_text().splitlines() if not "C(" in l)
print(summ)

# data_x = data_x[data_x["eligibility"] != 0]
model = OLS(data_x["altman_z"], pd.get_dummies(data_x[features], drop_first=True).astype(float))
results = model.fit()

print(results.summary())


data_x["eligibility"].value_counts()


# Plot correlation matrix
corr = data.select_dtypes(include=[np.number]).corr()

corr["altman_z"].abs().sort_values()

plt.figure(figsize=(12, 10))
# Plot heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=False)
# increase margins
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
plt.show()
