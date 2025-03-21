import pandas as pd
import wrds

conn = wrds.Connection()

conn.list_libraries().sort()

tables = conn.list_tables(library='tr_worldscope')

table_desc = conn.describe_table(library='tr_worldscope', table="wrds_ws_funda")
print(table_desc.to_csv())

for table in tables:
    table_info = conn.describe_table(library='tr_worldscope', table=table)
    if "item6030" in table_info["name"].values:
        print(table)



company = conn.get_table(library='tr_worldscope', table='wrds_ws_ids', obs=5)


query = """
SELECT COUNT(item6105) FROM trws.wrds_ws_company
LIMIT 10
"""
data_company = conn.raw_sql(query, date_cols=['date'])

query_revenue = """
SELECT COUNT(item6105) FROM trws.wrds_ws_funda
LIMIT 10
"""
data_revenue = conn.raw_sql(query_revenue, date_cols=['date'])

variables = {
    "total_debt": "item3255",
    "revenue": "item7240",
    "total_assets": "item7230"
}

query_annual = """
SELECT c.item6030, c.item6001, c.item6091, c.item6100, c.item6028, c.item6026, f.item3255, f.item7240, f.item7230, f.item5350
FROM trws.wrds_ws_company AS c
JOIN trws.wrds_ws_funda AS f
  ON c.item6105 = f.item6105
WHERE c.item6030 IS NOT NULL AND f.item7011 < 250 AND (f.item7240 < 50000000 OR f.item7230 < 43000000)
"""

data_join = conn.raw_sql(query_annual)

data_join.to_parquet("data/wrds_fundamentals_annual.parquet")

data_join = pd.read_parquet("data/wrds_fundamentals_annual.parquet")

data_join["item6026"].value_counts().iloc[:10]

import pyperclip
taxonomy_frame = pd.read_excel("data/taxonomy.xlsx", sheet_name=None)

csv_str = ""
for sheet_name, frame in taxonomy_frame.items():
    csv_str += f"\n\nSheet name: {sheet_name}\n" + frame.to_csv()

pyperclip.copy(csv_str)
