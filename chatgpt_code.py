import pandas as pd
import json
import duckdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# --- Main Analysis Function ---
def analyze_high_court_data():
    """
    Analyzes the Indian High Court Judgements dataset to answer three specific questions.
    """
    # Setup DuckDB connection with necessary extensions
    conn = duckdb.connect(database=':memory:', read_only=False)
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL parquet; LOAD parquet;")

    # Define the S3 path to the Parquet data
    s3_path = "'s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1'"

    # --- Question 1: Which high court disposed the most cases from 2019 - 2022? ---
    q1_query = f"""
        SELECT court
        FROM read_parquet({s3_path})
        WHERE year >= 2019 AND year <= 2022
        GROUP BY court
        ORDER BY COUNT(*) DESC
        LIMIT 1;
    """
    most_active_court = conn.execute(q1_query).fetchone()[0]

    # --- Question 2: What's the regression slope of the date_of_registration - decision_date by year in the court=33_10? ---
    # The query calculates delay_days and then finds the regression slope of delay_days vs. year.
    # It handles date conversion and filters out invalid or negative delays.
    q2_query = f"""
        SELECT
            REGR_SLOPE(
                (epoch(decision_date) - epoch(try_strptime(date_of_registration, '%d-%m-%Y'))) / 86400.0,
                CAST(year AS DOUBLE)
            ) AS slope
        FROM read_parquet({s3_path})
        WHERE court = '33_10'
          AND try_strptime(date_of_registration, '%d-%m-%Y') IS NOT NULL
          AND decision_date >= try_strptime(date_of_registration, '%d-%m-%Y');
    """
    regression_slope = conn.execute(q2_query).fetchone()[0]

    # --- Question 3: Plot the year and # of days of delay ---
    # Fetches the data needed for plotting. A LIMIT is used to keep the plot readable and the data size small.
    q3_query = f"""
        SELECT
            year,
            (epoch(decision_date) - epoch(try_strptime(date_of_registration, '%d-%m-%Y'))) / 86400.0 AS delay_days
        FROM read_parquet({s3_path})
        WHERE court = '33_10'
          AND try_strptime(date_of_registration, '%d-%m-%Y') IS NOT NULL
          AND decision_date >= try_strptime(date_of_registration, '%d-%m-%Y')
        LIMIT 5000;
    """
    plot_df = conn.execute(q3_query).fetchdf()

    # Generate the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    sns.regplot(
        x='year', 
        y='delay_days', 
        data=plot_df, 
        ax=ax,
        scatter_kws={'alpha': 0.2, 's': 15, 'edgecolor': 'none'},
        line_kws={'color': '#E53935', 'linewidth': 2}
    )
    
    ax.set_title('Case Delay vs. Decision Year for Court 33_10', fontsize=12)
    ax.set_xlabel('Decision Year', fontsize=10)
    ax.set_ylabel('Delay (Registration to Decision in Days)', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()

    # Encode plot to base64 data URI
    buf = io.BytesIO()
    plt.savefig(buf, format='webp', dpi=75) # Use webp for efficient compression
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plot_uri = f"data:image/webp;base64,{img_b64}"
    plt.close(fig)
    
    # Close the database connection
    conn.close()

    # --- Assemble the final JSON object ---
    final_result = {
        "Which high court disposed the most cases from 2019 - 2022?": most_active_court,
        "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": regression_slope,
        "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": plot_uri
    }

    return final_result

if __name__ == '__main__':
    # Execute the analysis and print the result as a JSON string
    result_json = analyze_high_court_data()
    print(json.dumps(result_json, indent=2))