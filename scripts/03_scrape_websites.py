import pandas as pd
import trafilatura
from multiprocessing.dummy import Pool as ThreadPool
import tqdm
import pydantic
import requests

from taxonomy_credit_risk.classification.taxonomy import TaxonomyClient, Compliance, Activity, Eligibility
from taxonomy_credit_risk.common_crawl import ScrapingClient


class TaxonomyResult(pydantic.BaseModel):
    company_code: float
    url: str
    text: str = None
    activity: Activity = None
    compliance: Compliance = None
    eligibility: Eligibility = None

# Load the classification client
# taxonomy_client  = TaxonomyClient.from_path(taxonomy_path="data/taxonomy.xlsx")
taxonomy_client  = TaxonomyClient.from_chromadb_client_path(taxonomy_path="data/taxonomy.xlsx")

# Load the scraping client
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

urls = data.drop_duplicates("code_company_code")[["item6030_internet_address", "code_company_code"]].dropna()

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

# Scrape the homepages of the companies
with ThreadPool(20) as pool:
    urls["page_text"] = list(tqdm.tqdm(pool.imap(load_page, urls["item6030_internet_address"])))

# urls["page_text"] = urls["item6030_internet_address"].map(pages)

# Drop companies for which we could not load the page
urls.dropna(subset=["page_text"], inplace=True)

# Initialize the result containers
results = [TaxonomyResult(company_code=code, url=url, text=page_text) for code, url, page_text in zip(urls["code_company_code"], urls["item6030_internet_address"], urls["page_text"])]

# Retrieve the activities best matching the pages
all_activities = []
for res in tqdm.tqdm(results):
    scored_pages, activities = taxonomy_client.get_activities_and_pages([res.text])
    all_activities.append(activities)

# Classify the most relevant activity
classified_activities = taxonomy_client.classify_activity(texts=[r.text for r in results], activities=all_activities)
for i, result in enumerate(results):
    result.activity = classified_activities[i]

# Only evaluate eligibility for companies for which we could classify an activity
results_to_classify = [result for result in results if result.activity is not None]

# Classify the eligibility of the companies
eligibilities = taxonomy_client.classify_eligibility(texts=[r.text for r in results_to_classify], activities=[r.activity for r in results_to_classify])
for i, result in enumerate(results_to_classify):
    result.eligibility = eligibilities[i]

# Store the results to a CSV file
eligibility_frame = pd.DataFrame([r.model_dump() for r in results])
eligibility_frame.to_csv("data/eligibility_frame.csv")
