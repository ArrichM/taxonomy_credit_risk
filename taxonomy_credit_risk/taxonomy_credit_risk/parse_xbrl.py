import os
import zipfile

import pandas as pd
import tqdm
from lxml import etree


def parse_and_dereference_xbrl(file_path):
    """
    Parses an XBRL file, extracting all tags and automatically dereferencing context and unit values.

    Parameters:
    file_path (str): The path to the XBRL file.

    Returns:
    pd.DataFrame: A DataFrame with all extracted and dereferenced data.
    """
    # Load and parse the XBRL file
    tree = etree.parse(file_path)
    root = tree.getroot()

    # Extracting namespace map and adjusting it for use with XPath
    namespaces = {k if k is not None else "default": v for k, v in root.nsmap.items()}

    # Extracting context and unit references
    context_refs = {}
    unit_refs = {}

    for context in root.findall(".//default:context", namespaces):
        context_id = context.attrib["id"]
        period_date = context.find(".//default:instant", namespaces)
        if period_date is not None:
            context_refs[context_id] = period_date.text

    for unit in root.findall(".//default:unit", namespaces):
        unit_id = unit.attrib["id"]
        measure = unit.find(".//default:measure", namespaces)
        if measure is not None:
            unit_refs[unit_id] = measure.text

    # Extracting all elements data
    data = []
    for elem in root.iter():
        context_ref = elem.attrib.get("contextRef")
        unit_ref = elem.attrib.get("unitRef")

        data.append(
            {
                "tag": elem.tag,
                "value": elem.text,
                "context": context_ref,
                "unit": unit_ref,
                "context_deref": context_refs.get(context_ref),
                "unit_deref": unit_refs.get(unit_ref),
            }
        )

    # Creating DataFrame
    all_data_df = pd.DataFrame(data)

    return all_data_df


zip_path = "/Users/max/Downloads/Accounts_Monthly_Data-April2010.zip"

# Unpack zipfile
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("/Users/max/Downloads/Accounts_Monthly_Data-April2010")

# List files
filings = os.listdir("/Users/max/Downloads/Accounts_Monthly_Data-April2010")

fact_frames = []
for filing in tqdm.tqdm(filings):
    try:
        frame = parse_and_dereference_xbrl(
            "/Users/max/Downloads/Accounts_Monthly_Data-April2010/" + filing
        )
        frame["document"] = filing
    except SyntaxError as e:
        print(e)

    fact_frames.append(frame)

fact_frame = pd.concat(fact_frames)

asssets = fact_frame[
    fact_frame["tag"]
    == "{http://www.xbrl.org/uk/fr/gaap/pt/2004-12-01}TotalAssetsLessCurrentLiabilities"
]
asssets["value"] = asssets["value"]
asssets["year"] = asssets["context_deref"].str[:4]
asssets.groupby("year").count()
asssets.groupby("year").mean()
asssets["context_deref"]

fact_counts = fact_frame["tag"].value_counts()
fact_counts.iloc[:20]

# Example usage
# file_path = 'path_to_xbrl_file.xml'
# df = parse_xbrl_to_dataframe(file_path)
# print(df.head())
