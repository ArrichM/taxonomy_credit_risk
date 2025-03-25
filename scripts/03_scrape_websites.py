import pandas as pd
import trafilatura
from multiprocessing.dummy import Pool as ThreadPool
import tqdm
import pydantic

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

# Select the target company
urls = data[data["item6026_nation"] == "GERMANY"].drop_duplicates("code_company_code")[["item6030_internet_address", "code_company_code"]]

def load_page(url: str):
    return trafilatura.extract(trafilatura.fetch_url(url), with_metadata=True, favor_recall=True)

with ThreadPool(8) as pool:
    urls["page_text"] = list(tqdm.tqdm(pool.imap(lambda x: trafilatura.extract(trafilatura.fetch_url(x), with_metadata=True, favor_recall=True), urls["item6030_internet_address"])))

urls.dropna(subset=["page_text"], inplace=True)

texts = urls["page_text"].to_list()

results = [TaxonomyResult(company_code=code, url=url, text=page_text) for code, url, page_text in zip(urls["code_company_code"], urls["item6030_internet_address"], urls["page_text"])]

all_activities = []
for text in tqdm.tqdm(texts):
    scored_pages, activities = taxonomy_client.get_activities_and_pages([text])
    all_activities.append(activities)


classified_activities = taxonomy_client.classify_activity(texts=texts, activities=all_activities)
for i, result in enumerate(results):
    result.activity = classified_activities[i]

results_to_classify = [result for result in results if result.activity is not None]

compliances = taxonomy_client.classify_compliance(texts=[r.text for r in results_to_classify], activities=[r.activity for r in results_to_classify])


for i, result in enumerate(results_to_classify):
    result.compliance = compliances[i]


compliance_frame = pd.DataFrame([r.model_dump() for r in results_to_classify])
compliance_frame = compliance_frame.dropna(subset=["compliance"])
compliance_frame["eligibility"] = compliance_frame["compliance"].apply(lambda x: max([dim.get("eligibility", 0) for dim in x["dimensions"]]))
compliance_frame["alignment"] = compliance_frame["compliance"].apply(lambda x: max([dim.get("alignment", 0) for dim in x["dimensions"]]))

analysis_frame = compliance_frame[["company_code", "eligibility", "alignment"]]

analysis_frame["code_company_code"] = analysis_frame["company_code"]

data = pd.merge(analysis_frame, data, on="code_company_code", how="left")




# Analyze Altman Z-score
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Plot correlation matrix
corr = data.select_dtypes(include=[np.number]).corr()

corr["altman_z"].abs().sort_values()

plt.figure(figsize=(12, 10))
# Plot heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=False)
# increase margins
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
plt.show()
