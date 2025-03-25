import pandas as pd
import requests
import trafilatura
import tldextract
from sentence_transformers import SentenceTransformer

from taxonomy_credit_risk.classification.taxonomy import TaxonomyClient
from taxonomy_credit_risk.common_crawl import ScrapingClient

embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")

# Load the classification client
# taxonomy_client  = TaxonomyClient.from_path(taxonomy_path="data/taxonomy.xlsx")
taxonomy_client  = TaxonomyClient.from_chromadb_client_path(taxonomy_path="data/taxonomy.xlsx")

scraping_client = ScrapingClient(embedding_model=embedding_model)

# Load the company data
data = pd.read_parquet("data/altman_data.parquet")

# Only load sme data that is available for 2020 and 2021
data["has_2020_and_2021"] = data.groupby("code_company_code")["year__fiscal_year"].transform(lambda x: (x == 2020).any() and (x == 2021).any())
data = data[data["has_2020_and_2021"]]
data = data.loc[data["sme_financial"].astype(bool)]


urls = data[data["item6026_nation"] == "GERMANY"].drop_duplicates("code_company_code")["item6030_internet_address"].unique()
target_domain= tldextract.extract(urls[11]).fqdn
print(target_domain)

text_frame = scraping_client.get_page_candidate_frame(target_url=target_domain, use_cc=False)

scored_pages, activities = taxonomy_client.get_activities_and_pages(text_frame["parsed"].to_list())


self = scraping_client

pages = text_frame["parsed"].to_list()
homepage_texts = text_frame.iloc[:10]["content"].apply(lambda x: trafilatura.extract(x, with_metadata=True)).to_list()
homepage_text = "\n--------------------------\n".join(homepage_texts)

homepage_text = trafilatura.extract(trafilatura.fetch_url(url), with_metadata=True)

print(homepage_text)
self = taxonomy_client
activities = taxonomy_client.classify_activity(texts=[homepage_text])

compliances = taxonomy_client.classify_compliance(texts=[homepage_text], activities=activities)

compliance = compliances[0]


for dimension in compliance.dimensions:
    print("-------------------------------------------------")
    print(dimension.dimension_name)
    print(dimension.eligibility)
    print(dimension.reasoning_eligibility)
    print(dimension.alignment)
    print(dimension.reasoning_alignment)


print(text_frame.iloc[0]["parsed"])
print(text_frame.iloc[1]["parsed"])
print(text_frame.iloc[2]["parsed"])
print(text_frame.iloc[8]["parsed"])

homepage = requests.get(urls[15])
texts = [trafilatura.extract(homepage.text)]
activities = client.classify_activity(texts)

compliances = client.classify_compliance(texts=texts, activities=activities)




