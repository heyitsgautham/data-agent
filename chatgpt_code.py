
import pandas as pd
import json
import duckdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

conn = duckdb.connect(database=':memory:', read_only=False)
conn.execute("INSTALL httpfs; LOAD httpfs;")
conn.execute("INSTALL parquet; LOAD parquet;")

S3_PATH = "'s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1'"

query1 = f"""
SELECT
    court
FROM read_parquet({S3_PATH})
WHERE
    year >= 2019 AND year <= 2022 AND disposal_nature IS NOT NULL
GROUP BY
    court
ORDER BY
    COUNT(*) DESC
LIMIT 1;
"""
most_cases_court_df = conn.execute(query1).fetchdf()
most_cases_court = most_cases_court_df['court'].iloc[0] if not most_cases_court_df.empty else None

query2 = f"""
SELECT
    year,
    AVG(DATE_DIFF('day', STRPTIME(date_of_registration, '%d-%m-%Y'), CAST(decision_date AS DATE))) AS avg_delay_days
FROM read_parquet({S3_PATH})
WHERE
    court = '33_10'
    AND date_of_registration IS NOT NULL
    AND decision_date IS NOT NULL
    AND TRY_STRPTIME(date_of_registration, '%d-%m-%Y') IS NOT NULL
GROUP BY
    year
HAVING
    COUNT(*) > 1 
ORDER BY
    year;
"""

query3 = f"""
SELECT * 
FROM read_parquet({S3_PATH})
"""
delay_data_df = conn.execute(query2).fetchdf().dropna()

regression_slope = 0.0
plot_base64 = ""

if not delay_data_df.empty and len(delay_data_df) > 1:
    x = delay_data_df['year']
    y = delay_data_df['avg_delay_days']
    slope, _ = np.polyfit(x, y, 1)
    regression_slope = slope

    plt.figure(figsize=(6, 4))
    sns.set_style("whitegrid")
    ax = sns.regplot(data=delay_data_df, x='year', y='avg_delay_days',
                     scatter_kws={'s': 20, 'alpha': 0.7},
                     line_kws={'color': 'red', 'linewidth': 2})
    ax.set_title('Avg. Case Delay by Year (Court 33_10)', fontsize=10)
    ax.set_xlabel('Year', fontsize=8)
    ax.set_ylabel('Average Delay (Days)', fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='webp', dpi=75)
    plt.close()
    buf.seek(0)
    encoded_string = base64.b64encode(buf.read()).decode('utf-8')
    plot_base64 = f"data:image/webp;base64,{encoded_string}"

conn.close()

result = {
  "Which high court disposed the most cases from 2019 - 2022?": most_cases_court,
  "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": regression_slope,
  "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": plot_base64
}

print(json.dumps(result))
print(query3)