import pandas as pd
import json
import duckdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

conn = duckdb.connect()
conn.execute("INSTALL httpfs; LOAD httpfs;")
conn.execute("INSTALL parquet; LOAD parquet;")

base_path = 's3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1'

# Query 1: Get the court with most cases
query1 = f"""
    SELECT court
    FROM '{base_path}'
    WHERE year BETWEEN 2019 AND 2022
    GROUP BY court
    ORDER BY COUNT(*) DESC
    LIMIT 1;
"""
most_cases_court = conn.execute(query1).fetchone()[0]

# Query 2: Delay calculation
query2 = f"""
    SELECT
        year,
        AVG(epoch(decision_date) - epoch(try_strptime(date_of_registration, '%d-%m-%Y'))) / 86400.0 AS avg_delay_days
    FROM '{base_path}'
    WHERE court = '33_10'
      AND try_strptime(date_of_registration, '%d-%m-%Y') IS NOT NULL
      AND decision_date IS NOT NULL
      AND epoch(decision_date) >= epoch(try_strptime(date_of_registration, '%d-%m-%Y'))
    GROUP BY year
    HAVING COUNT(*) > 1
    ORDER BY year;
"""
delay_df = conn.execute(query2).fetchdf()

conn.close()

# Data cleaning
delay_df.dropna(subset=['year', 'avg_delay_days'], inplace=True)
regression_slope = None
plot_uri = None

# Regression + plotting
if len(delay_df) >= 2:
    slope, _ = np.polyfit(delay_df['year'], delay_df['avg_delay_days'], 1)
    regression_slope = float(slope)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=90)
    sns.regplot(
        x='year', 
        y='avg_delay_days', 
        data=delay_df, 
        ax=ax, 
        scatter_kws={'alpha': 0.6}, 
        line_kws={'color': 'red', 'linewidth': 2}
    )
    ax.set_title('Average Case Delay by Year for Court 33_10', fontsize=10)
    ax.set_xlabel('Year', fontsize=8)
    ax.set_ylabel('Average Delay (Days)', fontsize=8)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='webp')
    plt.close(fig)
    
    buf.seek(0)
    img_bytes = buf.read()
    b64_string = base64.b64encode(img_bytes).decode('utf-8')
    plot_uri = f"data:image/webp;base64,{b64_string}"

# Output results
results = {
    "Which high court disposed the most cases from 2019 - 2022?": most_cases_court,
    "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": regression_slope,
    "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": plot_uri
}

print(json.dumps(results))
