import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import os

# Set plot style for academic publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Create output directories if they don't exist
os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

# Load data (same as original script)
print("Loading and preparing data...")
company_data = pd.read_parquet("data/altman_data_v2.parquet")

# Add EU dummy
eu_countries = ["GERMANY", "FRANCE", "ITALY", "SPAIN", "NETHERLANDS", "BELGIUM", "SWEDEN", "AUSTRIA", "IRELAND", "DENMARK", "FINLAND", "PORTUGAL", "GREECE", "CZECH REPUBLIC", "ROMANIA", "HUNGARY", "SLOVAKIA", "BULGARIA", "CROATIA", "SLOVENIA", "ESTONIA", "LATVIA", "LITHUANIA", "CYPRUS", "MALTA", "LUXEMBOURG", "POLAND"]
eg_countries = ["UNITED KINGDOM", "NORWAY", "SWITZERLAND", "TURKEY", "BOSNIA & HERZEGOVINA", "SERBIA", "MACEDONIA", "MONTENEGRO"]

company_data["is_eu"] = company_data["item6026_nation"].str.upper().isin(eu_countries)
company_data["is_eg"] = company_data["item6026_nation"].str.upper().isin(eu_countries + eg_countries)

# Load eligibility data
eligibility_data = pd.read_csv("data/eligibility_frame.csv", index_col=0)

# Deserialize eligibility data
eligibility_data["eligibility"] = eligibility_data["eligibility"].apply(lambda x: eval(x) if not pd.isna(x) else None)
eligibility_data["activity"] = eligibility_data["activity"].apply(lambda x: eval(x) if not pd.isna(x) else None)

# Extract eligibility score and eligible activity
eligibility_data["eligibility_score"] = eligibility_data["eligibility"].apply(lambda x: x.get("eligibility", 0) if x else 1)
eligibility_data["eligible_activity"] = eligibility_data["activity"].apply(lambda x: x.get("id", None) if not pd.isna(x) else None)

# Rename the company code col
eligibility_data["code_company_code"] = eligibility_data.pop("company_code")

# Only keep the relevant columns
eligibility_data = eligibility_data[["code_company_code", "eligibility_score", "eligible_activity"]]

# Merge data
data = pd.merge(eligibility_data, company_data, on="code_company_code", how="left")

# Process data for analysis (same as original)
data_x = data.copy()

# Add year-specific indicators for event study
data_x["year"] = data_x["year__fiscal_year"].astype(int)

# Create a categorical eligibility score variable
data_x["eligibility_category"] = pd.cut(
    data_x["eligibility_score"],
    bins=[0, 3, 7, 10],
    labels=["Low", "Medium", "High"]
)

# Add post 2021 dummy for main analysis
data_x["post_2021"] = data_x["year__fiscal_year"] > 2021

# Filter out companies with unknown eligibility
data_x = data_x[data_x["eligibility_score"] != 0]

# Use time window 2015 to 2025
data_x = data_x[data_x["year"] >= 2015]
data_x["year__fiscal_year"] = data_x["year__fiscal_year"].astype(str)

# Convert bool to int
data_x[data_x.select_dtypes(bool).columns] = data_x.select_dtypes(bool).astype(int)

# Clip z-scores to reasonable values
data_x = data_x[(data_x["altman_z"] > -100) & (data_x["altman_z"] < 100)]
data_x = data_x[(data_x["altman_z_private"] > -100) & (data_x["altman_z_private"] < 100)]

# Drop nations with industry collinearity
nation_industry_counts = data_x.groupby("item6026_nation")["item6011_industry_group"].nunique()
data_x = data_x[data_x["item6026_nation"].isin(nation_industry_counts[nation_industry_counts > 1].index)]

# Only keep SMEs
data_x = data_x[data_x["sme_strict"].astype(bool)]

print(f"Data prepared with {len(data_x)} observations")

# Create firm size categories for heterogeneity analysis
data_x["size_tercile"] = pd.qcut(data_x["item2300_total_assets"], 3, labels=["Small", "Medium", "Large"])

#---------------------------------------------------------------------------------
# 1. BASE REGRESSION (same as original)
#---------------------------------------------------------------------------------
print("Running base regression...")
print("=" * 80)
print("Base Regression")
print("=" * 80)

causal_block = "altman_z_private ~ eligibility_score + eligibility_score : is_eu + post_2021 : is_eu + eligibility_score : post_2021 + eligibility_score : post_2021 : is_eu"
controls = " + C(item6026_nation, Treatment) + C(item6011_industry_group, Treatment) + C(year__fiscal_year, Treatment) : C(item6010_general_industry_classification, Treatment)"
mod = ols(causal_block + controls, data=data_x)
res = mod.fit()

# Extract and format key coefficients
coef_table = res.summary2().tables[1]
main_coefs = coef_table.loc[[
    'eligibility_score',
    'eligibility_score:is_eu',
    'post_2021:is_eu',
    'eligibility_score:post_2021',
    'eligibility_score:post_2021:is_eu'
]]

formatted_table = pd.DataFrame({
    'Parameter': main_coefs.index,
    'Coef.': main_coefs['Coef.'].round(4),
    'Std.Err.': main_coefs['Std.Err.'].round(4),
    't-value': main_coefs['t'].round(4),
    'p-value': main_coefs['P>|t|'].round(4),
    '[0.025': main_coefs['[0.025'].round(4),
    '0.975]': main_coefs['0.975]'].round(4),
    '': np.where(main_coefs['P>|t|'] < 0.01, '***',
                 np.where(main_coefs['P>|t|'] < 0.05, '**',
                          np.where(main_coefs['P>|t|'] < 0.1, '*', '')))
})

print("Parameter                                          Coef.      Std.Err.   t-value    p-value    [0.025     0.975]    ")
print("------------------------------------------------------------------------------------------------------------------------")
for _, row in formatted_table.iterrows():
    print(f"{row['Parameter']:45} {row['Coef.']:10} {row['Std.Err.']:10} {row['t-value']:10} {row['p-value']:10} {row['[0.025']:10} {row['0.975]']:10} {row['']}")

print("------------------------------------------------------------------------------------------------------------------------")
print(f"R-squared: {res.rsquared:.4f}")
print(f"Adj. R-squared: {res.rsquared_adj:.4f}")
print(f"F-statistic: {res.fvalue:.4f}")
print(f"Number of observations: {res.nobs}")

# Save main results table
formatted_table.to_csv("results/tables/main_regression_results.csv", index=False)

#---------------------------------------------------------------------------------
# 2. PARALLEL TRENDS VISUALIZATION
#---------------------------------------------------------------------------------
# For parallel trends visualization, we'll examine the relationship between
# eligibility score and Z-score over time before treatment for EU vs non-EU firms

# Create a pre-treatment subset
pre_treatment = data_x[data_x["year"] <= 2021].copy()

# Group by year, EU status, and eligibility category
trends_data = pre_treatment.groupby(['year', 'is_eu', 'eligibility_category'])['altman_z_private'].mean().reset_index()

# Create parallel trends plot
plt.figure(figsize=(12, 8))
for eu_status in [0, 1]:
    for elig_cat in ['Low', 'Medium', 'High']:
        subset = trends_data[(trends_data['is_eu'] == eu_status) & (trends_data['eligibility_category'] == elig_cat)]
        if not subset.empty:
            plt.plot(subset['year'], subset['altman_z_private'],
                     marker='o', linestyle='-' if eu_status else '--',
                     label=f"{'EU' if eu_status else 'Non-EU'} - {elig_cat} Eligibility")

plt.axvline(x=2021, color='r', linestyle='--', alpha=0.7, label="Taxonomy Delegated Act")
plt.title("Pre-Treatment Trends in Altman Z-Score by EU Status and Eligibility Category")
plt.xlabel("Year")
plt.ylabel("Average Altman Z-Score (Private)")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/parallel_trends.png", dpi=300, bbox_inches='tight')

#---------------------------------------------------------------------------------
# 3. PLACEBO TESTS
#---------------------------------------------------------------------------------
print("Running placebo treatment year analysis...")

# Define placebo years to test
placebo_years = [2017, 2018, 2019]
placebo_results = []

for placebo_year in placebo_years:
    print("=" * 80)
    print(f"Placebo Treatment Year: {placebo_year}")
    print("=" * 80)

    # Create placebo treatment indicator
    data_x[f"post_{placebo_year}"] = data_x["year"] > placebo_year

    # Run the regression with placebo treatment
    placebo_formula = f"altman_z_private ~ eligibility_score + eligibility_score : is_eu + post_{placebo_year} : is_eu + eligibility_score : post_{placebo_year} + eligibility_score : post_{placebo_year} : is_eu + C(item6026_nation, Treatment) + C(item6011_industry_group, Treatment) + C(year__fiscal_year, Treatment) : C(item6010_general_industry_classification, Treatment)"

    # Filter to include only data from before the actual treatment
    placebo_data = data_x[data_x["year"] <= 2021].copy()
    placebo_mod = ols(placebo_formula, data=placebo_data)
    placebo_res = placebo_mod.fit()

    # Extract and print key coefficients
    placebo_coefs = placebo_res.summary2().tables[1]
    main_placebo_coefs = placebo_coefs.loc[[
        'eligibility_score',
        'eligibility_score:is_eu',
        f'post_{placebo_year}[True]:is_eu',
        f'eligibility_score:post_{placebo_year}[T.True]',
        f'eligibility_score:post_{placebo_year}[T.True]:is_eu'
    ]]

    # Store triple interaction coefficient for summary
    placebo_results.append({
        'Year': placebo_year,
        'Triple Interaction Coefficient': placebo_res.params[f'eligibility_score:post_{placebo_year}[T.True]:is_eu'],
        'Triple Interaction p-value': placebo_res.pvalues[f'eligibility_score:post_{placebo_year}[T.True]:is_eu'],
        'Double Interaction Coefficient': placebo_res.params[f'eligibility_score:post_{placebo_year}[T.True]'],
        'Double Interaction p-value': placebo_res.pvalues[f'eligibility_score:post_{placebo_year}[T.True]']
    })

    formatted_placebo_table = pd.DataFrame({
        'Parameter': main_placebo_coefs.index,
        'Coef.': main_placebo_coefs['Coef.'].round(4),
        'Std.Err.': main_placebo_coefs['Std.Err.'].round(4),
        't-value': main_placebo_coefs['t'].round(4),
        'p-value': main_placebo_coefs['P>|t|'].round(4),
        '[0.025': main_placebo_coefs['[0.025'].round(4),
        '0.975]': main_placebo_coefs['0.975]'].round(4),
        '': np.where(main_placebo_coefs['P>|t|'] < 0.01, '***',
                     np.where(main_placebo_coefs['P>|t|'] < 0.05, '**',
                              np.where(main_placebo_coefs['P>|t|'] < 0.1, '*', '')))
    })

    print("Parameter                                          Coef.      Std.Err.   t-value    p-value    [0.025     0.975]    ")
    print("------------------------------------------------------------------------------------------------------------------------")
    for _, row in formatted_placebo_table.iterrows():
        print(f"{row['Parameter']:45} {row['Coef.']:10} {row['Std.Err.']:10} {row['t-value']:10} {row['p-value']:10} {row['[0.025']:10} {row['0.975]']:10} {row['']}")

    print("------------------------------------------------------------------------------------------------------------------------")
    print(f"R-squared: {placebo_res.rsquared:.4f}")
    print(f"Adj. R-squared: {placebo_res.rsquared_adj:.4f}")
    print(f"F-statistic: {placebo_res.fvalue:.4f}")
    print(f"Number of observations: {placebo_res.nobs}")

# Run actual regression again to compare
print("Running base regression...")
print("=" * 80)
print("Base Regression")
print("=" * 80)
# This was already run above, so we just print the results here for reference

# Summary of placebo results
placebo_df = pd.DataFrame(placebo_results)
placebo_df['Year'] = placebo_df['Year'].astype(str)
# Add the actual treatment year results
placebo_df = pd.concat([placebo_df, pd.Series({
    'Year': '2021',
    'Triple Interaction Coefficient': res.params['eligibility_score:post_2021:is_eu'],
    'Triple Interaction p-value': res.pvalues['eligibility_score:post_2021:is_eu'],
    'Double Interaction Coefficient': res.params['eligibility_score:post_2021'],
    'Double Interaction p-value': res.pvalues['eligibility_score:post_2021']
}).to_frame().T], ignore_index=True, axis=0)

print("Placebo Test Results:")
print(placebo_df[['Year', 'Triple Interaction Coefficient', 'Triple Interaction p-value',
                  'Double Interaction Coefficient', 'Double Interaction p-value']].to_string(index=False))

# Save placebo results
placebo_df.to_csv("results/tables/placebo_test_results.csv", index=False)

# Plot placebo test results
plt.figure(figsize=(10, 6))
plt.bar(placebo_df['Year'], placebo_df['Triple Interaction Coefficient'], color=['blue', 'blue', 'blue', 'red'])
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.title("Triple Interaction Coefficients: Placebo vs. Actual Treatment Year")
plt.xlabel("Treatment Year")
plt.ylabel("Coefficient Value")
plt.grid(True, alpha=0.3)
plt.savefig("results/figures/placebo_test_coefficients.png", dpi=300, bbox_inches='tight')

#---------------------------------------------------------------------------------
# 4. SUB-PERIOD ANALYSIS
#---------------------------------------------------------------------------------
print("Running sub-period analysis...")

# Define sub-periods
sub_periods = [
    ("2015-2017", (2015, 2017)),
    ("2018-2019", (2018, 2019)),
    ("2021-2022", (2021, 2022)),
    ("2023-2025", (2023, 2025))
]

for period_name, (start_year, end_year) in sub_periods:
    print("=" * 80)
    print(f"Sub-period: {period_name}")
    print("=" * 80)

    # Filter data for this period
    period_data = data_x[(data_x["year"] >= start_year) & (data_x["year"] <= end_year)].copy()

    # For pre-treatment periods, we only want to see the basic relationship
    if end_year <= 2020:
        sub_formula = "altman_z_private ~ eligibility_score + eligibility_score : is_eu + C(item6026_nation, Treatment) + C(item6011_industry_group, Treatment)"
    else:
        sub_formula = "altman_z_private ~ eligibility_score + eligibility_score : is_eu + C(item6026_nation, Treatment) + C(item6011_industry_group, Treatment)"

    sub_mod = ols(sub_formula, data=period_data)
    sub_res = sub_mod.fit()

    # Extract relevant coefficients
    sub_coefs = sub_res.summary2().tables[1]

    if end_year <= 2020:  # Pre-treatment periods
        main_sub_coefs = sub_coefs.loc[[
            'eligibility_score',
            'eligibility_score:is_eu'
        ]]
    else:  # Post-treatment periods
        main_sub_coefs = sub_coefs.loc[[
            'eligibility_score',
            'eligibility_score:is_eu'
        ]]

    formatted_sub_table = pd.DataFrame({
        'Parameter': main_sub_coefs.index,
        'Coef.': main_sub_coefs['Coef.'].round(4),
        'Std.Err.': main_sub_coefs['Std.Err.'].round(4),
        't-value': main_sub_coefs['t'].round(4),
        'p-value': main_sub_coefs['P>|t|'].round(4),
        '[0.025': main_sub_coefs['[0.025'].round(4),
        '0.975]': main_sub_coefs['0.975]'].round(4),
        '': np.where(main_sub_coefs['P>|t|'] < 0.01, '***',
                     np.where(main_sub_coefs['P>|t|'] < 0.05, '**',
                              np.where(main_sub_coefs['P>|t|'] < 0.1, '*', '')))
    })

    print("Parameter                                          Coef.      Std.Err.   t-value    p-value    [0.025     0.975]    ")
    print("------------------------------------------------------------------------------------------------------------------------")
    for _, row in formatted_sub_table.iterrows():
        print(f"{row['Parameter']:45} {row['Coef.']:10} {row['Std.Err.']:10} {row['t-value']:10} {row['p-value']:10} {row['[0.025']:10} {row['0.975]']:10} {row['']}")

    print("------------------------------------------------------------------------------------------------------------------------")
    print(f"R-squared: {sub_res.rsquared:.4f}")
    print(f"Adj. R-squared: {sub_res.rsquared_adj:.4f}")
    print(f"F-statistic: {sub_res.fvalue:.4f}")
    print(f"Number of observations: {sub_res.nobs}")

#---------------------------------------------------------------------------------
# 5. FIRM SIZE HETEROGENEITY ANALYSIS
#---------------------------------------------------------------------------------
print("Running firm size heterogeneity analysis...")

size_results = []
for size in ["Small", "Medium", "Large"]:
    print("=" * 80)
    print(f"Firm Size: {size}")
    print("=" * 80)

    # Filter by size
    size_data = data_x[data_x["size_tercile"] == size].copy()

    # Run the main regression
    size_mod = ols(causal_block + controls, data=size_data)
    size_res = size_mod.fit()

    # Store key coefficients for comparison
    size_results.append({
        'Size Group': size,
        'eligibility_score': size_res.params['eligibility_score'],
        'eligibility_score:is_eu': size_res.params['eligibility_score:is_eu'],
        'post_2021:is_eu': size_res.params['post_2021:is_eu'],
        'eligibility_score:post_2021': size_res.params['eligibility_score:post_2021'],
        'eligibility_score:post_2021:is_eu': size_res.params['eligibility_score:post_2021:is_eu'],
        'N': size_res.nobs
    })

    # Create p-value annotations
    def add_stars(p_value):
        if p_value < 0.01:
            return "***"
        elif p_value < 0.05:
            return "**"
        elif p_value < 0.1:
            return "*"
        else:
            return ""

    # Extract and format key coefficients
    size_coefs = size_res.summary2().tables[1]
    main_size_coefs = size_coefs.loc[[
        'eligibility_score',
        'eligibility_score:is_eu',
        'post_2021:is_eu',
        'eligibility_score:post_2021',
        'eligibility_score:post_2021:is_eu'
    ]]

    formatted_size_table = pd.DataFrame({
        'Parameter': main_size_coefs.index,
        'Coef.': main_size_coefs['Coef.'].round(4),
        'Std.Err.': main_size_coefs['Std.Err.'].round(4),
        't-value': main_size_coefs['t'].round(4),
        'p-value': main_size_coefs['P>|t|'].round(4),
        '[0.025': main_size_coefs['[0.025'].round(4),
        '0.975]': main_size_coefs['0.975]'].round(4),
        '': [add_stars(p) for p in main_size_coefs['P>|t|']]
    })

    print("Parameter                                          Coef.      Std.Err.   t-value    p-value    [0.025     0.975]    ")
    print("------------------------------------------------------------------------------------------------------------------------")
    for _, row in formatted_size_table.iterrows():
        print(f"{row['Parameter']:45} {row['Coef.']:10} {row['Std.Err.']:10} {row['t-value']:10} {row['p-value']:10} {row['[0.025']:10} {row['0.975]']:10} {row['']}")

    print("------------------------------------------------------------------------------------------------------------------------")
    print(f"R-squared: {size_res.rsquared:.4f}")
    print(f"Adj. R-squared: {size_res.rsquared_adj:.4f}")
    print(f"F-statistic: {size_res.fvalue:.4f}")
    print(f"Number of observations: {size_res.nobs}")

# Create comparison table of coefficients across size groups
size_comparison_df = pd.DataFrame(size_results)

# Print the comparison table
print("Summary of Coefficients Across Size Groups")
print(size_comparison_df[['Size Group', 'eligibility_score', 'eligibility_score:is_eu',
                          'post_2021:is_eu', 'eligibility_score:post_2021',
                          'eligibility_score:post_2021:is_eu', 'N']].to_string(index=False))

# Save the comparison table
size_comparison_df.to_csv("results/tables/size_heterogeneity_comparison.csv", index=False)

# Plot the key coefficients across size groups
plt.figure(figsize=(12, 8))
barWidth = 0.2
r1 = np.arange(3)
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Extract coefficients
coeffs1 = size_comparison_df['eligibility_score']
coeffs2 = size_comparison_df['eligibility_score:post_2021']
coeffs3 = size_comparison_df['eligibility_score:post_2021:is_eu']

plt.bar(r1, coeffs1, width=barWidth, label='Eligibility Score (Base)', color='skyblue')
plt.bar(r2, coeffs2, width=barWidth, label='Eligibility Score × Post-2021', color='salmon')
plt.bar(r3, coeffs3, width=barWidth, label='Eligibility Score × Post-2021 × EU', color='lightgreen')

plt.xlabel('Firm Size', fontweight='bold')
plt.ylabel('Coefficient Value')
plt.xticks([r + barWidth for r in range(3)], ['Small', 'Medium', 'Large'])
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('Heterogeneity of Treatment Effects by Firm Size')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/size_heterogeneity_comparison.png", dpi=300, bbox_inches='tight')

#---------------------------------------------------------------------------------
# 6. ELIGIBILITY THRESHOLD ANALYSIS
#---------------------------------------------------------------------------------
print("Running eligibility threshold analysis...")

# Define different thresholds for high eligibility
thresholds = [5, 6, 7, 8]

threshold_results = []
for threshold in thresholds:
    print("=" * 80)
    print(f"Eligibility Threshold: {threshold}+")
    print("=" * 80)

    # Create binary high eligibility indicator
    data_x[f'high_elig_{threshold}'] = (data_x['eligibility_score'] >= threshold).astype(int)

    # Run regression with binary indicator
    threshold_formula = f"altman_z_private ~ high_elig_{threshold} + high_elig_{threshold} : is_eu + post_2021 : is_eu + high_elig_{threshold} : post_2021 + high_elig_{threshold} : post_2021 : is_eu + C(item6026_nation, Treatment) + C(item6011_industry_group, Treatment) + C(year__fiscal_year, Treatment) : C(item6010_general_industry_classification, Treatment)"

    threshold_mod = ols(threshold_formula, data=data_x)
    threshold_res = threshold_mod.fit()

    # Store key result
    threshold_results.append({
        'Threshold': threshold,
        'high_elig': threshold_res.params[f'high_elig_{threshold}'],
        'high_elig:is_eu': threshold_res.params[f'high_elig_{threshold}:is_eu'],
        'post_2021:is_eu': threshold_res.params['post_2021:is_eu'],
        'high_elig:post_2021': threshold_res.params[f'high_elig_{threshold}:post_2021'],
        'high_elig:post_2021:is_eu': threshold_res.params[f'high_elig_{threshold}:post_2021:is_eu'],
    })

    # Extract and print key coefficients
    threshold_coefs = threshold_res.summary2().tables[1]
    main_threshold_coefs = threshold_coefs.loc[[
        f'high_elig_{threshold}',
        f'high_elig_{threshold}:is_eu',
        'post_2021:is_eu',
        f'high_elig_{threshold}:post_2021',
        f'high_elig_{threshold}:post_2021:is_eu'
    ]]

    formatted_threshold_table = pd.DataFrame({
        'Parameter': main_threshold_coefs.index,
        'Coef.': main_threshold_coefs['Coef.'].round(4),
        'Std.Err.': main_threshold_coefs['Std.Err.'].round(4),
        't-value': main_threshold_coefs['t'].round(4),
        'p-value': main_threshold_coefs['P>|t|'].round(4),
        '[0.025': main_threshold_coefs['[0.025'].round(4),
        '0.975]': main_threshold_coefs['0.975]'].round(4),
        '': np.where(main_threshold_coefs['P>|t|'] < 0.01, '***',
                     np.where(main_threshold_coefs['P>|t|'] < 0.05, '**',
                              np.where(main_threshold_coefs['P>|t|'] < 0.1, '*', '')))
    })

    print("Parameter                                          Coef.      Std.Err.   t-value    p-value    [0.025     0.975]    ")
    print("------------------------------------------------------------------------------------------------------------------------")
    for _, row in formatted_threshold_table.iterrows():
        print(f"{row['Parameter']:45} {row['Coef.']:10} {row['Std.Err.']:10} {row['t-value']:10} {row['p-value']:10} {row['[0.025']:10} {row['0.975]']:10} {row['']}")

    print("------------------------------------------------------------------------------------------------------------------------")
    print(f"R-squared: {threshold_res.rsquared:.4f}")
    print(f"Adj. R-squared: {threshold_res.rsquared_adj:.4f}")
    print(f"F-statistic: {threshold_res.fvalue:.4f}")
    print(f"Number of observations: {threshold_res.nobs}")

# Also run a model with eligibility as categorical (Low, Medium, High)
print("=" * 80)
print("Eligibility Categories")
print("=" * 80)

# Create categorical variables for easier interpretation
data_x['elig_low'] = (data_x['eligibility_score'] < 4).astype(int)
data_x['elig_medium'] = ((data_x['eligibility_score'] >= 4) & (data_x['eligibility_score'] < 7)).astype(int)
data_x['elig_high'] = (data_x['eligibility_score'] >= 7).astype(int)

# Run categorical regression (omitting elig_low as the reference)
cat_formula = "altman_z_private ~ elig_medium + elig_high + elig_medium: is_eu + elig_high: is_eu + post_2021 : is_eu + elig_medium: post_2021 + elig_high: post_2021 + elig_medium: post_2021 : is_eu + elig_high: post_2021 : is_eu + C(item6026_nation, Treatment) + C(item6011_industry_group, Treatment) + C(year__fiscal_year, Treatment) : C(item6010_general_industry_classification, Treatment)"

cat_mod = ols(cat_formula, data=data_x)
cat_res = cat_mod.fit()

# Extract and print key coefficients
cat_coefs = cat_res.summary2().tables[1]
main_cat_coefs = cat_coefs.loc[[
    'elig_medium', 'elig_high',
    'elig_medium:is_eu', 'elig_high:is_eu',
    'post_2021:is_eu',
    'elig_medium:post_2021', 'elig_high:post_2021',
    'elig_medium:post_2021:is_eu', 'elig_high:post_2021:is_eu'
]]

formatted_cat_table = pd.DataFrame({
    'Parameter': main_cat_coefs.index,
    'Coef.': main_cat_coefs['Coef.'].round(4),
    'Std.Err.': main_cat_coefs['Std.Err.'].round(4),
    't-value': main_cat_coefs['t'].round(4),
    'p-value': main_cat_coefs['P>|t|'].round(4),
    '[0.025': main_cat_coefs['[0.025'].round(4),
    '0.975]': main_cat_coefs['0.975]'].round(4),
    '': np.where(main_cat_coefs['P>|t|'] < 0.01, '***',
                 np.where(main_cat_coefs['P>|t|'] < 0.05, '**',
                          np.where(main_cat_coefs['P>|t|'] < 0.1, '*', '')))
})

print("Parameter                                          Coef.      Std.Err.   t-value    p-value    [0.025     0.975]    ")
print("------------------------------------------------------------------------------------------------------------------------")
for _, row in formatted_cat_table.iterrows():
    print(f"{row['Parameter']:45} {row['Coef.']:10} {row['Std.Err.']:10} {row['t-value']:10} {row['p-value']:10} {row['[0.025']:10} {row['0.975]']:10} {row['']}")

print("------------------------------------------------------------------------------------------------------------------------")
print(f"R-squared: {cat_res.rsquared:.4f}")
print(f"Adj. R-squared: {cat_res.rsquared_adj:.4f}")
print(f"F-statistic: {cat_res.fvalue:.4f}")
print(f"Number of observations: {cat_res.nobs}")

#---------------------------------------------------------------------------------
# 7. SEPARATE EU AND NON-EU ANALYSIS
#---------------------------------------------------------------------------------
print("Running separate EU and non-EU analysis...")

# EU firms only
print("=" * 80)
print("EU Firms Only")
print("=" * 80)

eu_data = data_x[data_x['is_eu'] == 1].copy()
eu_formula = "altman_z_private ~ eligibility_score + post_2021 + eligibility_score : post_2021 + C(item6026_nation, Treatment) + C(item6011_industry_group, Treatment) + C(year__fiscal_year, Treatment) : C(item6010_general_industry_classification, Treatment)"

eu_mod = ols(eu_formula, data=eu_data)
eu_res = eu_mod.fit()

# Extract key coefficients
eu_coefs = eu_res.summary2().tables[1]
main_eu_coefs = eu_coefs.loc[[
    'eligibility_score',
    'post_2021',
    'eligibility_score:post_2021'
]]

formatted_eu_table = pd.DataFrame({
    'Parameter': main_eu_coefs.index,
    'Coef.': main_eu_coefs['Coef.'].round(4),
    'Std.Err.': main_eu_coefs['Std.Err.'].round(4),
    't-value': main_eu_coefs['t'].round(4),
    'p-value': main_eu_coefs['P>|t|'].round(4),
    '[0.025': main_eu_coefs['[0.025'].round(4),
    '0.975]': main_eu_coefs['0.975]'].round(4),
    '': np.where(main_eu_coefs['P>|t|'] < 0.01, '***',
                 np.where(main_eu_coefs['P>|t|'] < 0.05, '**',
                          np.where(main_eu_coefs['P>|t|'] < 0.1, '*', '')))
})

print("Parameter                                          Coef.      Std.Err.   t-value    p-value    [0.025     0.975]    ")
print("------------------------------------------------------------------------------------------------------------------------")
for _, row in formatted_eu_table.iterrows():
    print(f"{row['Parameter']:45} {row['Coef.']:10} {row['Std.Err.']:10} {row['t-value']:10} {row['p-value']:10} {row['[0.025']:10} {row['0.975]']:10} {row['']}")

print("------------------------------------------------------------------------------------------------------------------------")
print(f"R-squared: {eu_res.rsquared:.4f}")
print(f"Adj. R-squared: {eu_res.rsquared_adj:.4f}")
print(f"F-statistic: {eu_res.fvalue:.4f}")
print(f"Number of observations: {eu_res.nobs}")

# Non-EU firms only
print("=" * 80)
print("Non-EU Firms Only")
print("=" * 80)

noneu_data = data_x[data_x['is_eu'] == 0].copy()
noneu_formula = "altman_z_private ~ eligibility_score + post_2021 + eligibility_score : post_2021 + C(item6026_nation, Treatment) + C(item6011_industry_group, Treatment) + C(year__fiscal_year, Treatment) : C(item6010_general_industry_classification, Treatment)"

noneu_mod = ols(noneu_formula, data=noneu_data)
noneu_res = noneu_mod.fit()

# Extract key coefficients
noneu_coefs = noneu_res.summary2().tables[1]
main_noneu_coefs = noneu_coefs.loc[[
    'eligibility_score',
    'post_2021',
    'eligibility_score:post_2021'
]]

formatted_noneu_table = pd.DataFrame({
    'Parameter': main_noneu_coefs.index,
    'Coef.': main_noneu_coefs['Coef.'].round(4),
    'Std.Err.': main_noneu_coefs['Std.Err.'].round(4),
    't-value': main_noneu_coefs['t'].round(4),
    'p-value': main_noneu_coefs['P>|t|'].round(4),
    '[0.025': main_noneu_coefs['[0.025'].round(4),
    '0.975]': main_noneu_coefs['0.975]'].round(4),
    '': np.where(main_noneu_coefs['P>|t|'] < 0.01, '***',
                 np.where(main_noneu_coefs['P>|t|'] < 0.05, '**',
                          np.where(main_noneu_coefs['P>|t|'] < 0.1, '*', '')))
})

print("Parameter                                          Coef.      Std.Err.   t-value    p-value    [0.025     0.975]    ")
print("------------------------------------------------------------------------------------------------------------------------")
for _, row in formatted_noneu_table.iterrows():
    print(f"{row['Parameter']:45} {row['Coef.']:10} {row['Std.Err.']:10} {row['t-value']:10} {row['p-value']:10} {row['[0.025']:10} {row['0.975]']:10} {row['']}")

print("------------------------------------------------------------------------------------------------------------------------")
print(f"R-squared: {noneu_res.rsquared:.4f}")
print(f"Adj. R-squared: {noneu_res.rsquared_adj:.4f}")
print(f"F-statistic: {noneu_res.fvalue:.4f}")
print(f"Number of observations: {noneu_res.nobs}")

# Compare estimates between EU and non-EU samples
print("Comparison of EU and Non-EU Estimates:")
comparison_df = pd.DataFrame({
    'Coefficient': ['Intercept', 'eligibility_score', 'post_2021', 'eligibility_score:post_2021'],
    'EU_Estimate': [eu_res.params['Intercept'], eu_res.params['eligibility_score'],
                    eu_res.params['post_2021'], eu_res.params['eligibility_score:post_2021']],
    'EU_P-value': [eu_res.pvalues['Intercept'], eu_res.pvalues['eligibility_score'],
                   eu_res.pvalues['post_2021'], eu_res.pvalues['eligibility_score:post_2021']],
    'NonEU_Estimate': [noneu_res.params['Intercept'], noneu_res.params['eligibility_score'],
                       noneu_res.params['post_2021'], noneu_res.params['eligibility_score:post_2021']],
    'NonEU_P-value': [noneu_res.pvalues['Intercept'], noneu_res.pvalues['eligibility_score'],
                      noneu_res.pvalues['post_2021'], noneu_res.pvalues['eligibility_score:post_2021']],
    'Difference': [eu_res.params['Intercept'] - noneu_res.params['Intercept'],
                   eu_res.params['eligibility_score'] - noneu_res.params['eligibility_score'],
                   eu_res.params['post_2021'] - noneu_res.params['post_2021'],
                   eu_res.params['eligibility_score:post_2021'] - noneu_res.params['eligibility_score:post_2021']]
})
print(comparison_df)

# Save comparison results
comparison_df.to_csv("results/tables/eu_vs_noneu_comparison.csv", index=False)

#---------------------------------------------------------------------------------
# 8. EVENT STUDY ANALYSIS
#---------------------------------------------------------------------------------
print("Running event study analysis...")

# Create year dummies centered around 2020 (reference year)
years = sorted(data_x['year'].unique())
years.remove(2020)  # Reference year

# Create formula for event study regression
event_formula = "altman_z_private ~ eligibility_score + eligibility_score : is_eu"

for year in years:
    year_var = f"year_{year}"
    data_x[year_var] = (data_x['year'] == year).astype(int)
    event_formula += f" + {year_var} + {year_var}:eligibility_score + {year_var}: is_eu + {year_var}:eligibility_score : is_eu"

event_formula += " + C(item6026_nation, Treatment) + C(item6011_industry_group, Treatment)"

print(f"Event study formula: {event_formula}")

event_mod = ols(event_formula, data=data_x)
event_res = event_mod.fit()

# Extract triple interaction terms for each year
event_coefs = event_res.summary2().tables[1]
event_results = []

for year in years:
    year_var = f"year_{year}:eligibility_score:is_eu"
    if year_var in event_res.params:
        event_results.append({
            'Year': year,
            'Coefficient': event_res.params[year_var],
            'Std.Err': event_res.bse[year_var],
            'P-value': event_res.pvalues[year_var],
            'CI_Lower': event_res.conf_int().loc[year_var][0],
            'CI_Upper': event_res.conf_int().loc[year_var][1]
        })

event_df = pd.DataFrame(event_results)
print("Event Study Results:")
print(event_df.to_string(index=False))

# Test for parallel pre-trends: joint test of all pre-2020 coefficients
pre_treatment_vars = [f"year_{year}:eligibility_score:is_eu" for year in years if year < 2020]
if len(pre_treatment_vars) > 0:
    # Create restriction matrix for joint test
    R = np.zeros((len(pre_treatment_vars), len(event_res.params)))

    for i, var in enumerate(pre_treatment_vars):
        if var in event_res.params:
            R[i, list(event_res.params.index).index(var)] = 1

    # Null hypothesis: all pre-treatment coefficients = 0
    joint_test = event_res.f_test(R)

    print("Joint Test of Pre-trends (H0: All pre-treatment coefficients = 0)")
    print(f"F-statistic: {joint_test.statistic:.4f}")
    print(f"P-value: {joint_test.pvalue:.4f}")
    if joint_test.pvalue > 0.05:
        print("Fail to reject null hypothesis: No evidence of pre-trends")
    else:
        print("Reject null hypothesis: Evidence of pre-trends")

# Save event study results
event_df.to_csv("results/tables/event_study_results.csv", index=False)

# Create event study plot
plt.figure(figsize=(12, 8))
plt.plot(event_df['Year'], event_df['Coefficient'], 'o-', color='blue', linewidth=2)
plt.fill_between(event_df['Year'],
                 event_df['CI_Lower'],
                 event_df['CI_Upper'],
                 color='blue', alpha=0.2)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.axvline(x=2021, color='g', linestyle='--', label='EU Taxonomy Climate Delegated Act')
plt.xlabel('Year')
plt.ylabel('Differential Effect of Eligibility Score in EU vs. Non-EU Firms')
plt.title('Event Study: Differential Effect of Eligibility Score by Year')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/event_study.png", dpi=300, bbox_inches='tight')

print("All robustness checks completed.")
print("Results saved in the 'results' directory")