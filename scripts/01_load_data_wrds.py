import wrds

conn = wrds.Connection()

col_block = """
    code           AS code_company_code,
    year_          AS year__fiscal_year,
    item6105       AS item6105_perm_id,
    item5350       AS item5350_fiscal_period_end_date,
    item2201       AS item2201_current_assets,
    item3101       AS item3101_current_liabilities,
    item2300       AS item2300_total_assets,
    item3495       AS item3495_retained_earnings,
    item18191      AS item18191_ebit,
    item7210       AS item7210_market_value_equity,
    item3351       AS item3351_total_liabilities,
    item1001       AS item1001_sales,
    item7210       AS item7210_equity,
    item3051       AS item3051_short_term_debt,
    item3251       AS item3251_long_term_debt
""".strip()

query = f"""
SELECT
{col_block},
    item7011       AS item7011_number_of_employees,
    item7240       AS item7240_total_revenue,
FROM tr_worldscope.wrds_ws_funda
"""

df = conn.raw_sql(query)
df.to_parquet("data/wrds_fundamentals_annual.parquet")

query = f"""
SELECT
{col_block}
FROM tr_worldscope.wrds_ws_fundq
"""

df = conn.raw_sql(query)
df.to_parquet("data/wrds_fundamentals_quarterly.parquet")

query = """
SELECT 
    code     AS code_company_code,
    item6105 AS item6105_perm_id,
    item6030 AS item6030_internet_address,
    item6001 AS item6001_company_name, 
    item6091 AS item6091_business_description,
    item6100 AS item6100_entity_type,
    item6028 AS item6028_region, 
    item6026 AS item6026_nation,
    item6011 AS item6011_industry_group,
    item6010 AS item6010_general_industry_classification,
    item7041 AS item7041_tr_business_classification,
    item1000 AS item1000_company_status
FROM trws.wrds_ws_company
"""

df = conn.raw_sql(query)
df.to_parquet("data/company_data.parquet")
