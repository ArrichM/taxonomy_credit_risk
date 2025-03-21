import os

import pandas as pd
import requests
import tqdm
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from .env
api_key = os.getenv("UK_API_KEY")

# Pull all companies
companies = []
pbar = tqdm.tqdm(total=10_000_000)
offset = 0
while True:
    response = requests.get(
        "https://api.company-information.service.gov.uk/advanced-search/companies",
        params={
            "size": 5000,
            "start_index": offset,
            "incorporated_to": "2015-01-01",
            "company_status": "Dissolved",
        },
        auth=(api_key, ""),
    )
    items = response.json()["items"]
    companies.extend(items)
    pbar.update(len(items))
    offset += len(items)

    if len(items) < 5000:
        break

companies_frame = pd.DataFrame(companies)

companies_frame.to_parquet("data/uk_companies/uk_companies.parquet")


data = pd.read_csv("/Users/max/Downloads/BasicCompanyDataAsOneFile.csv")

data["Accounts.AccountCategory"].value_counts()

# Only keep micro entities and small companies
data = data[
    data["Accounts.AccountCategory"].isin(
        ["MICRO ENTITY", "SMALL", "TOTAL EXEMPTION SMALL"]
    )
]
data["IncorporationDate"] = pd.to_datetime(data["IncorporationDate"])
data = data[data["IncorporationDate"] < "2013-01-01"]
data["CompanyStatus"].value_counts()


with open(os.path.expanduser("~/.keys/uk_ch.txt"), "r") as f:
    api_key = f.read()


#
# response = requests.get(
#     url,
#     params={"q": query, "items_per_page": 10},
#     auth=(api_key, ""),
# )
#


company_number = response.json()["items"][0]["company_number"]
filing_list_response = requests.get(
    f"https://api.company-information.service.gov.uk/company/{company_number}/filing-history",
    params={"items_per_page": 100},
    auth=(api_key, ""),
)

filing_frame = pd.DataFrame(filing_list_response.json()["items"])

metadata_url = filing_list_response.json()["items"][21]["links"]["document_metadata"]

metadata_response = requests.get(
    metadata_url,
    auth=(api_key, ""),
)

metadata_response.json()
