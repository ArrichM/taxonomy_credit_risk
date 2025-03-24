import pandas as pd
import requests
import trafilatura
from sentence_transformers import SentenceTransformer

from taxonomy_credit_risk.classification.taxonomy import TaxonomyClient
from taxonomy_credit_risk.common_crawl import ScrapingClient

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load the classification client
taxonomy_client  = TaxonomyClient.from_chromadb_client_path(taxonomy_path="data/taxonomy.xlsx")

scraping_client = ScrapingClient(embedding_model=embedding_model)

# Load the company data
data = pd.read_parquet("data/altman_data.parquet")

# Only load sme data that is available for 2020 and 2021
data["has_2020_and_2021"] = data.groupby("code_company_code")["year__fiscal_year"].transform(lambda x: (x == 2020).any() and (x == 2021).any())
data = data[data["has_2020_and_2021"]]
data = data.loc[data["sme_financial"].astype(bool)]

urls = data.drop_duplicates("code_company_code")["item6030_internet_address"].unique()
len(urls)

url = urls[15]

text_frame = scraping_client.get_page_candidate_frame(target_url=url)

homepage = requests.get(urls[15])
texts = [trafilatura.extract(homepage.text)]
activities = client.classify_activity(texts)

compliances = client.classify_compliance(texts=texts, activities=activities)

compliance = compliances[0]

compliance.dimensions[0].is_eligible
compliance.dimensions[0].alignment
compliance.dimensions[0].reasoning_alignment
compliance.dimensions[0].reasoning_eligibility
compliance.dimensions[0].score

