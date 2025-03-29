import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from pathlib import Path

# Create output directory for results
Path("results/robustness").mkdir(parents=True, exist_ok=True)

# Function to print abbreviated regression results
def print_abbreviated_results(results, title="Regression Results"):
    """Print key coefficients from regression results, excluding fixed effects"""
    result_str = f"\n{'='*80}\n{title}\n{'='*80}\n"

    # Extract coefficients of interest (excluding fixed effects)
    params = results.params
    pvalues = results.pvalues
    conf_int = results.conf_int()

    # Create a table of the main coefficients (excluding fixed effects)
    key_coefs = []
    for var in params.index:
        if "C(" not in var:  # Filter out fixed effects
            key_coefs.append((var, params[var], results.bse[var],
                              results.tvalues[var], pvalues[var],
                              conf_int.loc[var, 0], conf_int.loc[var, 1]))

    # Format the table
    result_str += f"{'Parameter':<50} {'Coef.':<10} {'Std.Err.':<10} {'t-value':<10} {'p-value':<10} {'[0.025':<10} {'0.975]':<10}\n"
    result_str += f"{'-'*120}\n"

    for var, coef, se, tval, pval, ci_l, ci_h in key_coefs:
        stars = ""
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"

        result_str += f"{var:<50} {coef:10.4f} {se:10.4f} {tval:10.4f} {pval:10.4f} {ci_l:10.4f} {ci_h:10.4f} {stars}\n"

    result_str += f"{'-'*120}\n"
    result_str += f"R-squared: {results.rsquared:.4f}\n"
    result_str += f"Adj. R-squared: {results.rsquared_adj:.4f}\n"
    result_str += f"F-statistic: {results.fvalue:.4f}\n"
    result_str += f"Number of observations: {results.nobs}\n"

    print(result_str)

    # Also save to file
    with open(f"results/robustness/{title.replace(' ', '_').replace(':', '_').lower()}.txt", 'w') as f:
        f.write(result_str)

    return result_str

def load_data():
    """Load and prepare data for analysis"""
    print("Loading and preparing data...")

    # Load the company data
    company_data = pd.read_parquet("data/altman_data_v2.parquet")

    # Add EU dummy
    eu_countries = ["GERMANY", "FRANCE", "ITALY", "SPAIN", "NETHERLANDS", "BELGIUM", "SWEDEN", "AUSTRIA", "IRELAND",
                    "DENMARK", "FINLAND", "PORTUGAL", "GREECE", "CZECH REPUBLIC", "ROMANIA", "HUNGARY", "SLOVAKIA",
                    "BULGARIA", "CROATIA", "SLOVENIA", "ESTONIA", "LATVIA", "LITHUANIA", "CYPRUS", "MALTA",
                    "LUXEMBOURG", "POLAND"]

    company_data["is_eu"] = company_data["item6026_nation"].str.upper().isin(eu_countries)

    # Load the eligibility data
    eligibility_data = pd.read_csv("data/eligibility_frame.csv", index_col=0)

    # Deserialize the eligibility data
    eligibility_data["eligibility"] = eligibility_data["eligibility"].apply(
        lambda x: eval(x) if not pd.isna(x) else None
    )
    eligibility_data["activity"] = eligibility_data["activity"].apply(
        lambda x: eval(x) if not pd.isna(x) else None
    )

    # Extract the eligibility score and eligible activity
    eligibility_data["eligibility_score"] = eligibility_data["eligibility"].apply(
        lambda x: x.get("eligibility", 0) if x else 1
    )
    eligibility_data["eligible_activity"] = eligibility_data["activity"].apply(
        lambda x: x.get("id", None) if not pd.isna(x) else None
    )

    # Rename the company code col
    eligibility_data["code_company_code"] = eligibility_data.pop("company_code")

    # Only keep the relevant columns
    eligibility_data = eligibility_data[["code_company_code", "eligibility_score", "eligible_activity"]]

    # Merge datasets
    data = pd.merge(eligibility_data, company_data, on="code_company_code", how="left")

    # Add post 2021 dummy - this is when the policy break appears
    data["post_2021"] = data["year__fiscal_year"] > 2021

    # Filter out companies for which we could not determine eligibility status
    data = data[data["eligibility_score"] != 0]

    # Use a symmetric time window around the policy break, i.e. 2015 to 2025
    data = data[data["year__fiscal_year"] >= 2015]
    data["year_numeric"] = data["year__fiscal_year"]  # Keep numeric version for analysis
    data["year__fiscal_year"] = data["year__fiscal_year"].astype(str)

    # Convert bool to int for regression
    data[data.select_dtypes(bool).columns] = data.select_dtypes(bool).astype(int)

    # Clip the altman z-score to reasonable values
    data = data[(data["altman_z"] > -100) & (data["altman_z"] < 100)]
    data = data[(data["altman_z_private"] > -100) & (data["altman_z_private"] < 100)]

    # Drop nations which are colinear with the industry
    nation_industry_counts = data.groupby("item6026_nation")["item6011_industry_group"].nunique()
    data = data[data["item6026_nation"].isin(nation_industry_counts[nation_industry_counts > 1].index)]

    # Only keep companies which are SMEs according to the official EU definition
    data = data[data["sme_strict"].astype(bool)]

    print(f"Data prepared with {len(data)} observations")
    return data

# Base regression
def run_base_regression(data):
    """Run the main regression specification"""
    print("\nRunning base regression...")

    formula = ("altman_z_private ~ eligibility_score + eligibility_score:is_eu + "
               "post_2021:is_eu + eligibility_score:post_2021 + "
               "eligibility_score:post_2021:is_eu + C(item6026_nation, Treatment) + "
               "C(item6011_industry_group, Treatment) + "
               "C(year__fiscal_year, Treatment):C(item6010_general_industry_classification, Treatment)")

    mod = ols(formula, data=data)
    results = mod.fit()

    print_abbreviated_results(results, "Base Regression")
    return results

# Robustness Check 1: Alternative Credit Risk Specifications
def robustness_alternative_credit_risk(data):
    """Test alternative measures of credit risk"""
    print("\nRunning alternative credit risk specifications...")

    # 1.1 Use the original Altman Z-score
    formula = ("altman_z ~ eligibility_score + eligibility_score:is_eu + "
               "post_2021:is_eu + eligibility_score:post_2021 + "
               "eligibility_score:post_2021:is_eu + C(item6026_nation, Treatment) + "
               "C(item6011_industry_group, Treatment) + "
               "C(year__fiscal_year, Treatment):C(item6010_general_industry_classification, Treatment)")

    mod = ols(formula, data=data)
    results = mod.fit()
    print_abbreviated_results(results, "Alternative Credit Risk: Original Altman Z")

    # 1.2 Use components of Z-score
    # Working capital / Total assets
    data["wc_ta"] = (data["item2201_current_assets"] - data["item3101_current_liabilities"]) / data["item2300_total_assets"]
    data["wc_ta"] = data["wc_ta"].clip(-10, 10)  # Clip outliers

    formula = ("wc_ta ~ eligibility_score + eligibility_score:is_eu + "
               "post_2021:is_eu + eligibility_score:post_2021 + "
               "eligibility_score:post_2021:is_eu + C(item6026_nation, Treatment) + "
               "C(item6011_industry_group, Treatment) + "
               "C(year__fiscal_year, Treatment):C(item6010_general_industry_classification, Treatment)")

    mod = ols(formula, data=data)
    results_wc_ta = mod.fit()
    print_abbreviated_results(results_wc_ta, "Alternative Credit Risk: Working Capital/Total Assets")

    return {"altman_z": results, "wc_ta": results_wc_ta}

# Robustness Check 2: Sample Sensitivity - Firm Size
def robustness_firm_size(data):
    """Test for heterogeneous effects across firm size"""
    print("\nRunning firm size heterogeneity analysis...")

    # Create size terciles based on total assets
    data["size_tercile"] = pd.qcut(data["item2300_total_assets"], 3, labels=["Small", "Medium", "Large"])

    results = {}
    for size in ["Small", "Medium", "Large"]:
        subset = data[data["size_tercile"] == size]

        formula = ("altman_z_private ~ eligibility_score + eligibility_score:is_eu + "
                   "post_2021:is_eu + eligibility_score:post_2021 + "
                   "eligibility_score:post_2021:is_eu + C(item6026_nation, Treatment) + "
                   "C(item6011_industry_group, Treatment) + "
                   "C(year__fiscal_year, Treatment):C(item6010_general_industry_classification, Treatment)")

        try:
            mod = ols(formula, data=subset)
            res = mod.fit()
            print_abbreviated_results(res, f"Firm Size: {size}")
            results[size] = res
        except:
            print(f"Error in estimation for {size} firms. Likely insufficient observations or multicollinearity.")
            results[size] = None

    # Create a summary table of coefficients across size groups
    coefficients = ["eligibility_score", "eligibility_score:is_eu",
                    "post_2021:is_eu", "eligibility_score:post_2021",
                    "eligibility_score:post_2021:is_eu"]

    summary_data = []
    for size, res in results.items():
        if res is not None:
            row = [size]
            for coef in coefficients:
                if coef in res.params:
                    row.append(f"{res.params[coef]:.4f}")
                    if res.pvalues[coef] < 0.01:
                        row[-1] += "***"
                    elif res.pvalues[coef] < 0.05:
                        row[-1] += "**"
                    elif res.pvalues[coef] < 0.1:
                        row[-1] += "*"
                else:
                    row.append("N/A")
            row.append(f"{res.nobs}")
            summary_data.append(row)

    columns = ["Size Group"] + coefficients + ["N"]
    summary_df = pd.DataFrame(summary_data, columns=columns)

    print("\nSummary of Coefficients Across Size Groups")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("results/robustness/size_heterogeneity_summary.csv", index=False)

    return results

# Robustness Check 3: Placebo Treatment Years
def robustness_placebo_years(data):
    """Test placebo treatment years to validate the identification assumption"""
    print("\nRunning placebo treatment year analysis...")

    placebo_years = [2017, 2018, 2019]  # Years before the actual treatment
    results = {}
    coefs = []

    # Run regressions with different placebo years
    for year in placebo_years:
        # Create placebo treatment variable
        data[f"post_{year}"] = (data["year_numeric"] >= year).astype(int)

        formula = (f"altman_z_private ~ eligibility_score + eligibility_score:is_eu + "
                   f"post_{year}:is_eu + eligibility_score:post_{year} + "
                   f"eligibility_score:post_{year}:is_eu + C(item6026_nation, Treatment) + "
                   f"C(item6011_industry_group, Treatment) + "
                   f"C(year__fiscal_year, Treatment):C(item6010_general_industry_classification, Treatment)")

        mod = ols(formula, data=data)
        res = mod.fit()

        print_abbreviated_results(res, f"Placebo Treatment Year: {year}")
        results[year] = res

        try:
            # Extract the triple interaction coefficient
            triple_coef = res.params[f"eligibility_score:post_{year}:is_eu"]
            triple_pval = res.pvalues[f"eligibility_score:post_{year}:is_eu"]
            double_coef = res.params[f"eligibility_score:post_{year}"]
            double_pval = res.pvalues[f"eligibility_score:post_{year}"]

            coefs.append({
                'Year': year,
                'Triple Interaction Coefficient': triple_coef,
                'Triple Interaction p-value': triple_pval,
                'Double Interaction Coefficient': double_coef,
                'Double Interaction p-value': double_pval
            })
        except:
            print(f"Could not extract coefficients for year {year}")

    # Add the actual treatment year (2021) for comparison
    try:
        triple_coef_actual = run_base_regression(data).params["eligibility_score:post_2021:is_eu"]
        triple_pval_actual = run_base_regression(data).pvalues["eligibility_score:post_2021:is_eu"]
        double_coef_actual = run_base_regression(data).params["eligibility_score:post_2021"]
        double_pval_actual = run_base_regression(data).pvalues["eligibility_score:post_2021"]

        coefs.append({
            'Year': 2021,
            'Triple Interaction Coefficient': triple_coef_actual,
            'Triple Interaction p-value': triple_pval_actual,
            'Double Interaction Coefficient': double_coef_actual,
            'Double Interaction p-value': double_pval_actual
        })
    except:
        print("Could not extract coefficients for actual treatment year 2021")

    # Create a summary dataframe and plot
    coef_df = pd.DataFrame(coefs)

    if not coef_df.empty and len(coef_df) > 1:
        plt.figure(figsize=(10, 6))

        # Plot triple interaction coefficients
        plt.subplot(2, 1, 1)
        plt.scatter(coef_df['Year'], coef_df['Triple Interaction Coefficient'], color='blue', s=50)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=2021, color='g', linestyle='--', label="EU Taxonomy Publication")
        plt.title('Placebo Test: Triple Interaction Coefficients by Treatment Year')
        plt.ylabel('Coefficient\n(eligibility × post × EU)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot double interaction coefficients
        plt.subplot(2, 1, 2)
        plt.scatter(coef_df['Year'], coef_df['Double Interaction Coefficient'], color='green', s=50)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=2021, color='g', linestyle='--', label="EU Taxonomy Publication")
        plt.title('Placebo Test: Double Interaction Coefficients by Treatment Year')
        plt.xlabel('Treatment Year')
        plt.ylabel('Coefficient\n(eligibility × post)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig("results/robustness/placebo_test_plot.png")
        plt.close()

        print("\nPlacebo Test Results:")
        print(coef_df.to_string(index=False))
        coef_df.to_csv("results/robustness/placebo_test_results.csv", index=False)

    return results

# Robustness Check 4: Sub-period Analysis
def robustness_sub_period(data):
    """Test for effect stability in different sub-periods"""
    print("\nRunning sub-period analysis...")

    # Define sub-periods
    data["period"] = pd.cut(
        data["year_numeric"],
        bins=[2014, 2017, 2019, 2022, 2026],
        labels=["2015-2017", "2018-2019", "2021-2022", "2023-2025"]
    )

    results = {}
    for period in data["period"].cat.categories:
        period_data = data[data["period"] == period]

        if len(period_data) < 1000:
            print(f"Insufficient data for period {period}, skipping...")
            continue

        # For pre-2021 periods, we can't estimate post_2021 effects
        if period in ["2015-2017", "2018-2019"]:
            formula = ("altman_z_private ~ eligibility_score + eligibility_score:is_eu + "
                       "C(item6026_nation, Treatment) + C(item6011_industry_group, Treatment)")
        else:
            formula = ("altman_z_private ~ eligibility_score + eligibility_score:is_eu + "
                       "C(item6026_nation, Treatment) + C(item6011_industry_group, Treatment)")

        mod = ols(formula, data=period_data)
        res = mod.fit()

        print_abbreviated_results(res, f"Sub-period: {period}")
        results[period] = res

    return results

# Robustness Check 5: Eligibility Score Thresholds
def robustness_eligibility_thresholds(data):
    """Test different thresholds of eligibility scores"""
    print("\nRunning eligibility threshold analysis...")

    # Create binary eligibility indicators at different thresholds
    thresholds = [5, 6, 7, 8]

    for threshold in thresholds:
        data[f"high_elig_{threshold}"] = (data["eligibility_score"] >= threshold).astype(int)

        formula = (f"altman_z_private ~ high_elig_{threshold} + high_elig_{threshold}:is_eu + "
                   f"post_2021:is_eu + high_elig_{threshold}:post_2021 + "
                   f"high_elig_{threshold}:post_2021:is_eu + C(item6026_nation, Treatment) + "
                   f"C(item6011_industry_group, Treatment) + "
                   f"C(year__fiscal_year, Treatment):C(item6010_general_industry_classification, Treatment)")

        mod = ols(formula, data=data)
        res = mod.fit()

        print_abbreviated_results(res, f"Eligibility Threshold: {threshold}+")

    # Create a non-linear specification with eligibility score categories
    data["elig_category"] = pd.cut(
        data["eligibility_score"],
        bins=[0, 3, 6, 10],
        labels=["Low (1-3)", "Medium (4-6)", "High (7-10)"]
    )

    # Create dummies for medium and high (low is reference)
    data["elig_medium"] = (data["elig_category"] == "Medium (4-6)").astype(int)
    data["elig_high"] = (data["elig_category"] == "High (7-10)").astype(int)

    formula = ("altman_z_private ~ elig_medium + elig_high + "
               "elig_medium:is_eu + elig_high:is_eu + "
               "post_2021:is_eu + elig_medium:post_2021 + elig_high:post_2021 + "
               "elig_medium:post_2021:is_eu + elig_high:post_2021:is_eu + "
               "C(item6026_nation, Treatment) + C(item6011_industry_group, Treatment) + "
               "C(year__fiscal_year, Treatment):C(item6010_general_industry_classification, Treatment)")

    mod = ols(formula, data=data)
    res = mod.fit()

    print_abbreviated_results(res, "Eligibility Categories")

    return res

# Robustness Check 6: Separate EU and Non-EU Analysis
def robustness_eu_noneu_separate(data):
    """Analyze EU and non-EU samples separately"""
    print("\nRunning separate EU and non-EU analysis...")

    # EU sample
    eu_data = data[data["is_eu"] == 1].copy()
    eu_formula = ("altman_z_private ~ eligibility_score + post_2021 + "
                  "eligibility_score:post_2021 + "
                  "C(item6026_nation, Treatment) + "
                  "C(item6011_industry_group, Treatment) + "
                  "C(year__fiscal_year, Treatment)")

    mod_eu = ols(eu_formula, data=eu_data)
    res_eu = mod_eu.fit()

    print_abbreviated_results(res_eu, "EU Firms Only")

    # Non-EU sample
    noneu_data = data[data["is_eu"] == 0].copy()
    noneu_formula = ("altman_z_private ~ eligibility_score + post_2021 + "
                     "eligibility_score:post_2021 + "
                     "C(item6026_nation, Treatment) + "
                     "C(item6011_industry_group, Treatment) + "
                     "C(year__fiscal_year, Treatment)")

    mod_noneu = ols(noneu_formula, data=noneu_data)
    res_noneu = mod_noneu.fit()

    print_abbreviated_results(res_noneu, "Non-EU Firms Only")

    # Compare the key coefficients between EU and non-EU
    comparison = pd.DataFrame({
        'Coefficient': ['Intercept', 'eligibility_score', 'post_2021', 'eligibility_score:post_2021'],
        'EU_Estimate': [res_eu.params['Intercept'], res_eu.params['eligibility_score'],
                        res_eu.params['post_2021'], res_eu.params['eligibility_score:post_2021']],
        'EU_P-value': [res_eu.pvalues['Intercept'], res_eu.pvalues['eligibility_score'],
                       res_eu.pvalues['post_2021'], res_eu.pvalues['eligibility_score:post_2021']],
        'NonEU_Estimate': [res_noneu.params['Intercept'], res_noneu.params['eligibility_score'],
                           res_noneu.params['post_2021'], res_noneu.params['eligibility_score:post_2021']],
        'NonEU_P-value': [res_noneu.pvalues['Intercept'], res_noneu.pvalues['eligibility_score'],
                          res_noneu.pvalues['post_2021'], res_noneu.pvalues['eligibility_score:post_2021']],
        'Difference': [res_eu.params['Intercept'] - res_noneu.params['Intercept'],
                       res_eu.params['eligibility_score'] - res_noneu.params['eligibility_score'],
                       res_eu.params['post_2021'] - res_noneu.params['post_2021'],
                       res_eu.params['eligibility_score:post_2021'] - res_noneu.params['eligibility_score:post_2021']]
    })

    print("\nComparison of EU and Non-EU Estimates:")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    comparison.to_csv("results/robustness/eu_noneu_comparison.csv", index=False)

    return {"eu": res_eu, "noneu": res_noneu}

# Robustness Check 7: Event Study
def robustness_event_study(data):
    """Run event study analysis to assess pre-trends and dynamic treatment effects"""
    print("\nRunning event study analysis...")

    # Create year dummies (using 2020 as reference year)
    ref_year = 2020
    data_es = data.copy()

    # Create interaction terms for each year
    year_dummies = []
    for year in sorted(data_es["year_numeric"].unique()):
        if year != ref_year:
            year_var = f"year_{year}"
            data_es[year_var] = (data_es["year_numeric"] == year).astype(int)
            year_dummies.append(year_var)

    # Create the formula with year interactions
    formula_terms = [
        "altman_z_private ~ eligibility_score",
        "eligibility_score:is_eu"
    ]

    # Add year dummies and interactions
    for year_var in year_dummies:
        formula_terms.extend([
            f"{year_var}",
            f"{year_var}:eligibility_score",
            f"{year_var}:is_eu",
            f"{year_var}:eligibility_score:is_eu"
        ])

    # Add fixed effects
    formula_terms.extend([
        "C(item6026_nation, Treatment)",
        "C(item6011_industry_group, Treatment)"
    ])

    formula = " + ".join(formula_terms)

    print("Event study formula:", formula)

    try:
        mod = ols(formula, data=data_es)
        res = mod.fit()

        # Extract coefficients for triple interaction by year
        coefs = []
        years = sorted(data_es["year_numeric"].unique())
        for year in years:
            if year != ref_year:
                year_var = f"year_{year}"
                triple_var = f"{year_var}:eligibility_score:is_eu"

                if triple_var in res.params:
                    coef = res.params[triple_var]
                    stderr = res.bse[triple_var]
                    pvalue = res.pvalues[triple_var]
                    ci_lower = res.conf_int().loc[triple_var, 0]
                    ci_upper = res.conf_int().loc[triple_var, 1]

                    coefs.append({
                        'Year': year,
                        'Coefficient': coef,
                        'Std.Err': stderr,
                        'P-value': pvalue,
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper
                    })

        # Plot the event study
        if coefs:
            coef_df = pd.DataFrame(coefs)
            coef_df.to_csv("results/robustness/event_study_coefficients.csv", index=False)

            plt.figure(figsize=(12, 6))
            plt.plot(coef_df['Year'], coef_df['Coefficient'], 'o-', color='blue', linewidth=2)
            plt.fill_between(coef_df['Year'], coef_df['CI_Lower'], coef_df['CI_Upper'],
                             color='blue', alpha=0.2)
            plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
            plt.axvline(x=2021, color='green', linestyle='--', linewidth=2,
                        label="EU Taxonomy Publication")
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Coefficient (eligibility × EU × year)', fontsize=12)
            plt.title('Event Study: Effect of Eligibility on Credit Risk for EU vs. Non-EU Firms by Year',
                      fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig("results/robustness/event_study_plot.png", dpi=300)
            plt.close()

            print("\nEvent Study Results:")
            print(coef_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

            # Formal test of pre-trends
            pre_trend_years = [y for y in years if y < ref_year]
            if pre_trend_years:
                pre_trend_vars = [f"year_{y}:eligibility_score:is_eu" for y in pre_trend_years]
                pre_trend_vars = [v for v in pre_trend_vars if v in res.params.index]

                if pre_trend_vars:
                    joint_test = res.f_test([f"{v} = 0" for v in pre_trend_vars])

                    print("\nJoint Test of Pre-trends (H0: All pre-treatment coefficients = 0)")
                    print(f"F-statistic: {joint_test.fvalue:.4f}")
                    print(f"P-value: {joint_test.pvalue:.4f}")

                    if joint_test.pvalue > 0.05:
                        print("Fail to reject null hypothesis: No evidence of pre-trends")
                    else:
                        print("Reject null hypothesis: Evidence of pre-trends exists")

        return res

    except Exception as e:
        print(f"Error in event study analysis: {e}")
        return None

# Main function to run all robustness checks
def run_all_robustness_checks():
    # Load data
    data = load_data()

    # Run base regression
    base_results = run_base_regression(data)

    # Run robustness checks
    alt_cr_results = robustness_alternative_credit_risk(data)
    size_results = robustness_firm_size(data)
    placebo_results = robustness_placebo_years(data)
    sub_period_results = robustness_sub_period(data)
    threshold_results = robustness_eligibility_thresholds(data)
    eu_noneu_results = robustness_eu_noneu_separate(data)
    event_study_results = robustness_event_study(data)

    print("\nAll robustness checks completed.")
    print("Results saved in the 'results/robustness' directory")

    return {
        "base": base_results,
        "alt_cr": alt_cr_results,
        "size": size_results,
        "placebo": placebo_results,
        "sub_period": sub_period_results,
        "threshold": threshold_results,
        "eu_noneu": eu_noneu_results,
        "event_study": event_study_results
    }


if __name__ == "__main__":
    results = run_all_robustness_checks()
