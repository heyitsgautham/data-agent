
import pandas as pd
import json
import duckdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def run_analysis():
    conn = duckdb.connect(database=':memory:', read_only=False)
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL parquet; LOAD parquet;")

    S3_PATH = "s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1"

    q1_query = f"""
        SELECT
            court
        FROM read_parquet('{S3_PATH}')
        WHERE year BETWEEN 2019 AND 2022
        GROUP BY court
        ORDER BY COUNT(*) DESC
        LIMIT 1;
    """
    most_cases_court_df = conn.execute(q1_query).fetchdf()
    most_cases_court = most_cases_court_df['court'].iloc[0] if not most_cases_court_df.empty else None

    q2_3_query = f"""
        SELECT
            year,
            AVG(decision_date - strptime(date_of_registration, '%d-%m-%Y')) as avg_delay_interval
        FROM read_parquet('{S3_PATH}')
        WHERE
            court = '33_10'
            AND try_strptime(date_of_registration, '%d-%m-%Y') IS NOT NULL
            AND decision_date IS NOT NULL
        GROUP BY
            year
        HAVING
            AVG(decision_date - strptime(date_of_registration, '%d-%m-%Y')) IS NOT NULL
        ORDER BY
            year;
    """
    delay_df = conn.execute(q2_3_query).fetchdf()
    conn.close()
    
    slope = 0.0
    plot_data_uri = ""

    if not delay_df.empty:
        delay_df['avg_delay'] = delay_df['avg_delay_interval'].dt.total_seconds() / (24 * 3600)
        delay_df.dropna(subset=['avg_delay'], inplace=True)
        
        if len(delay_df) > 1:
            model = np.polyfit(delay_df['year'], delay_df['avg_delay'], 1)
            slope = model[0]

            plt.figure(figsize=(8, 6))
            sns.regplot(x='year', y='avg_delay', data=delay_df)
            plt.title('Average Case Delay by Year for Court 33_10')
            plt.xlabel('Year')
            plt.ylabel('Average Delay (Days)')
            plt.grid(True)
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='webp', dpi=75)
            img_buffer.seek(0)
            
            base64_encoded_img = base64.b64encode(img_buffer.read()).decode('utf-8')
            plot_data_uri = f"data:image/webp;base64,{base64_encoded_img}"
            plt.close()

    result = {
      "Which high court disposed the most cases from 2019 - 2022?": most_cases_court,
      "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": slope,
      "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": plot_data_uri
    }
    
    print(json.dumps(result))

if __name__ == '__main__':
    run_analysis()
