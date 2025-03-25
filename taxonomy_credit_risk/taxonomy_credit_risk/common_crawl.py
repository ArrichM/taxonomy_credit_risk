
import trafilatura
from more_itertools import chunked
from tqdm import tqdm

import ssl
import urllib.request
from bs4 import BeautifulSoup


from typing import Tuple

import pandas as pd
from warcio.archiveiterator import ArchiveIterator

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List, Dict
import json
from urllib.parse import quote_plus

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SERVER = 'http://index.commoncrawl.org/'
myagent = 'cc-get-started/1.0 (Example data retrieval script; yourname@example.com)'


# Main execution
INDICES = [
    "CC-MAIN-2025-08",
    "CC-MAIN-2025-05",
    "CC-MAIN-2024-51",
    "CC-MAIN-2024-46",
    "CC-MAIN-2024-42",
    "CC-MAIN-2024-38",
    "CC-MAIN-2024-33",
    # "CC-MAIN-2024-30",
    # "CC-MAIN-2024-26",
    # "CC-MAIN-2024-22",
    # "CC-MAIN-2024-18",
    # "CC-MAIN-2024-10",
]
def search_cc_index(url: str, indexes: List[str]) -> List[Dict]:
    encoded_url = quote_plus(url)
    all_records: List[Dict] = []
    session = create_session()
    for index in indexes:
        index_url = f'{SERVER}{index}-index?url={encoded_url}&output=json'
        try:
            response = session.get(index_url, timeout=30)
            if response.status_code == 200:
                records = response.text.strip().split('\n')
                all_records.extend([json.loads(record) for record in records if record])
            else:
                print(f"Index search failed for {index} with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed for index {index}: {e}")
    return all_records

# Function to fetch content from Common Crawl
def fetch_page_from_cc(record: Dict, session: requests.Session) -> Tuple[bytes, Dict[str, str]]:
    offset, length = int(record['offset']), int(record['length'])
    s3_url = f'https://data.commoncrawl.org/{record["filename"]}'

    # Define the byte range for the request
    byte_range = f'bytes={offset}-{offset + length - 1}'

    # Send the HTTP GET request to the S3 URL with the specified byte range
    response = session.get(
        s3_url,
        headers={'Range': byte_range},
        stream=True,
        timeout=30  # Optional: Set a timeout for the request
    )
    response.raise_for_status()

    stream = ArchiveIterator(response.raw)
    for warc_record in stream:
        if warc_record.rec_type == 'response':
            http_headers = warc_record.http_headers
            headers_dict = {k.lower(): v for k, v in http_headers.headers}
            body = warc_record.content_stream().read()
            return body, headers_dict
    else:
        raise ValueError(f'No response record found at {s3_url}')

# Create a session with retry strategy
def create_session() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({'User-Agent': myagent})
    return session

# Worker function to process a single record
def process_record(record: Dict, session: requests.Session) -> Dict[str, Dict]:
    try:
        if not record.get("source") == "sitemap":
            content, headers = fetch_page_from_cc(record, session)
        else:
            response = requests.get(record["url"], headers={'User-Agent': myagent})
            content, headers = response.content, response.headers

        return {"content": content, "headers": headers, "url": record["url"]}
    except Exception as e:
        # Optionally, log the error with more details
        print(f"Error fetching record {record.get('urlkey', 'N/A')}: {e}")
        return None

# Function to fetch webpages in parallel
def fetch_webpages_parallel(records: pd.DataFrame, max_workers: int = 20) -> list:
    webpages = []
    records_list = records.to_dict(orient='records')  # Convert DataFrame to list of dicts

    with create_session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_record = {executor.submit(process_record, record, session): record for record in records_list}

            # Use tqdm to display a progress bar
            for future in as_completed(future_to_record):
                result = future.result()
                if result:
                    webpages.append(result)
                # Optionally, handle None results or collect failed records

    return webpages


def split_text(text: str, chunk_size: int = 1_500, overlap: int = 400) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += chunk_size - overlap
    return chunks


def extract_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    return [a.get("href") for a in soup.find_all("a") if a.get("href")]


class ScrapingClient:

    def __init__(self):

        # Create an unverified SSL context and install an opener with it.
        ssl_context = ssl._create_unverified_context()
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)

    def get_record_frame_sitemap(self, target_url: str) -> pd.DataFrame:
        from usp.tree import sitemap_tree_for_homepage

        record_frame = pd.DataFrame()

        tree = sitemap_tree_for_homepage(f'https://{target_url}')
        sites = [{"url": page.url} for page in list(tree.all_pages())]
        site_frame = pd.DataFrame(sites)
        site_frame["length"] = 1_000
        site_frame["source"] = "sitemap"
        print(f"Found {len(site_frame)} sitemap pages")
        record_frame = pd.concat([record_frame, site_frame], axis=0)

        return record_frame

    def get_record_frame_cc(self, target_url: str) -> pd.DataFrame:

        records = search_cc_index(target_url + "/*", INDICES)

        # Create a DataFrame, sort by timestamp, and drop duplicates based on 'urlkey'
        try:
            record_frame = pd.DataFrame(records).sort_values(
                by=["timestamp"]).drop_duplicates(
                subset=["urlkey"], keep="first"
            ).reset_index(drop=True)
        except KeyError:
            record_frame = pd.DataFrame(records)

        if len(records) > 0:

            print(f"Found {len(record_frame)} total pages")

            # Drop duplicated pages
            record_frame.drop_duplicates(subset="digest", inplace=True)

            # Drop PDF
            record_frame = record_frame[~record_frame["mime-detected"].apply(lambda x: "pdf" in str(x))]

            # Drop 404
            record_frame = record_frame[record_frame["status"] != "404"]

            # Sort english pages to the front
            try:
                record_frame["is_en"] = record_frame["languages"].apply(lambda x: int("eng" in x) if not pd.isna(x) else 0)
                record_frame.sort_values(by="is_en", ascending=False, inplace=True)
            except KeyError:
                pass

            # Drop pages that redirect to pages we already have
            record_frame = record_frame[~record_frame["redirect"].isin(record_frame["urlkey"].str.split(")").str[1].to_list())]

            # Purge the language prefix and drop duplicates
            record_frame["url_clipped"] = record_frame["url"].apply(lambda x: "/".join(x.split("/")[4:]))
            record_frame.sort_values(by="url_clipped", ascending=False, inplace=True)
            record_frame.drop_duplicates(subset=["url_clipped"], inplace=True)

        return record_frame

    def get_page_candidate_frame(self, target_url: str, max_page_count: int = 10_000,  chunk_len: int = 1_500, use_cc: bool = True) -> pd.DataFrame:

        if use_cc:
            record_frame = self.get_record_frame_cc(target_url)

            if len(record_frame) < 50:
                record_frame_sitemap = self.get_record_frame_sitemap(target_url)
                record_frame = pd.concat([record_frame, record_frame_sitemap], axis=0).reset_index()

        else:
            record_frame = self.get_record_frame_sitemap(target_url)

        print(f"Found {len(record_frame)} unique pages")

        if len(record_frame) > max_page_count:
            # Then, drop very long URLS
            record_frame["url_len"] = record_frame["url"].str.len()
            record_frame.sort_values(by="url_len", ascending=True, inplace=True)
            record_frame = record_frame.iloc[:max_page_count]
            print(f"Reduced to {len(record_frame)} pages")

        record_frame = record_frame.reset_index(drop=True)

        max_size = max(record_frame["length"].astype(float).quantile(0.95), 30_000)
        record_frame = record_frame[record_frame["length"].astype(float) < max_size]
        record_frame = record_frame.iloc[:max_page_count]

        all_candidate_frames = []
        for batch_indexes in tqdm(chunked(record_frame.index, 100), total=(len(record_frame) // 100) + 1):

            try:
                batch_frame = record_frame.loc[batch_indexes]

                # Fetch webpages in parallel
                webpages = fetch_webpages_parallel(batch_frame, max_workers=20)
                pages_frame = pd.DataFrame(webpages)

                pages_frame["parsed"] = pages_frame["content"].apply(lambda x: trafilatura.extract(x, favor_recall=True, with_metadata=True))
                pages_frame = pages_frame.dropna(subset=["parsed"])

                candidate_frame = pages_frame.copy().dropna(subset=["parsed"]).drop_duplicates(subset=["parsed"])

                all_candidate_frames.append(candidate_frame)

            except Exception as e:
                print(e)

        final_candidate_frame = pd.concat(all_candidate_frames, axis=0)

        return final_candidate_frame
