
import pandas as pd
import json
import duckdb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

conn = duckdb.connect()
conn.execute("INSTALL httpfs; LOAD httpfs;")
conn.execute("INSTALL parquet; LOAD parquet;")

q1_query = """
SELECT court
FROM read_parquet('')
WHERE year BETWEEN 2019 AND 2022
GROUP BY court
ORDER BY COUNT(*) DESC
LIMIT 1;
"""
q1_result = conn.execute(q1_query).fetchone()
most_active_court = q1_result[0] if q1_result else None

q2_q3_query = """
SELECT
    year,
    AVG(epoch(CAST(decision_date AS DATE)) - epoch(strptime(date_of_registration, '%d-%m-%Y'))) / 86400.0 AS avg_delay_days
FROM read_parquet('')
WHERE
    court = '33_10'
    AND TRY_STRPTIME(date_of_registration, '%d-%m-%Y') IS NOT NULL
GROUP BY year
HAVING avg_delay_days IS NOT NULL AND avg_delay_days > 0
ORDER BY year;
"""
delay_df = conn.execute(q2_q3_query).fetchdf()

slope = None
if not delay_df.empty and len(delay_df) >= 2:
    delay_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    delay_df.dropna(inplace=True)
    if len(delay_df) >= 2:
        delay_df['year'] = pd.to_numeric(delay_df['year'])
        delay_df['avg_delay_days'] = pd.to_numeric(delay_df['avg_delay_days'])
        slope, _ = np.polyfit(delay_df['year'], delay_df['avg_delay_days'], 1)

plot_b64 = ""
if not delay_df.empty:
    plt.figure(figsize=(6, 4))
    sns.set_theme(style="whitegrid")
    reg_plot = sns.regplot(
        x='year',
        y='avg_delay_days',
        data=delay_df,
        scatter_kws={'alpha': 0.7},
        line_kws={'color': 'red'}
    )
    reg_plot.set_title('Avg. Case Delay by Year (Court 33_10)', fontsize=12)
    reg_plot.set_xlabel('Year', fontsize=10)
    reg_plot.set_ylabel('Average Delay (Days)', fontsize=10)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='webp', optimize=True)
    plt.close()
    buf.seek(0)
    
    b64_string = base64.b64encode(buf.read()).decode('utf-8')
    plot_b64 = f"data:image/webp;base64,{b64_string}"

conn.close()

final_result = {
  "Which high court disposed the most cases from 2019 - 2022?": most_active_court,
  "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": slope,
  "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": plot_b64
}

print(json.dumps(final_result))
