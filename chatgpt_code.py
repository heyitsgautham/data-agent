import pandas as pd
import json
import matplotlib.pyplot as plt
import io
import base64

# --- Main Analysis Function ---
def analyze_sales_data(file_path):
    """
    Analyzes sales data from a CSV file to calculate key metrics and generate charts.

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        dict: A dictionary containing the analysis results in JSON format.
    """
    # Load the dataset from the provided CSV file
    df = pd.read_csv(file_path)

    # Ensure the 'date' column is in datetime format for time-series analysis
    df['date'] = pd.to_datetime(df['date'])

    # --- Question 1: What is the total sales across all regions? ---
    total_sales = df['sales'].sum()

    # --- Question 2: Which region has the highest total sales? ---
    sales_by_region = df.groupby('region')['sales'].sum()
    top_region = sales_by_region.idxmax()

    # --- Question 3: What is the correlation between day of month and sales? ---
    df['day_of_month'] = df['date'].dt.day
    day_sales_correlation = df['day_of_month'].corr(df['sales'])

    # --- Question 4: Plot total sales by region as a bar chart. ---
    plt.figure(figsize=(8, 5))
    sales_by_region.sort_values(ascending=False).plot(kind='bar', color='blue')
    plt.title('Total Sales by Region')
    plt.xlabel('Region')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    bar_chart_buf = io.BytesIO()
    plt.savefig(bar_chart_buf, format='png')
    plt.close()
    bar_chart_buf.seek(0)
    bar_chart_base64 = base64.b64encode(bar_chart_buf.getvalue()).decode('utf-8')

    # --- Question 5: What is the median sales amount across all orders? ---
    median_sales = df['sales'].median()

    # --- Question 6: What is the total sales tax if the tax rate is 10%? ---
    tax_rate = 0.10
    total_sales_tax = total_sales * tax_rate

    # --- Question 7: Plot cumulative sales over time as a line chart. ---
    df_sorted = df.sort_values('date')
    df_sorted['cumulative_sales'] = df_sorted['sales'].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(df_sorted['date'], df_sorted['cumulative_sales'], color='red', marker='o', linestyle='-')
    plt.title('Cumulative Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Sales')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    cumulative_chart_buf = io.BytesIO()
    plt.savefig(cumulative_chart_buf, format='png')
    plt.close()
    cumulative_chart_buf.seek(0)
    cumulative_sales_chart_base64 = base64.b64encode(cumulative_chart_buf.getvalue()).decode('utf-8')

    # --- Assemble the final JSON object ---
    result = {
        "total_sales": float(total_sales),
        "top_region": top_region,
        "day_sales_correlation": float(day_sales_correlation) if pd.notna(day_sales_correlation) else None,
        "bar_chart": bar_chart_base64,
        "median_sales": float(median_sales),
        "total_sales_tax": float(total_sales_tax),
        "cumulative_sales_chart": cumulative_sales_chart_base64
    }
    
    return result

# --- Execution ---
if __name__ == '__main__':
    # The filename is taken from the ALLOWED_DATA_SOURCES in the problem description
    csv_file_path = 'ProvidedCSV.csv'
    
    # Perform the analysis
    analysis_result = analyze_sales_data(csv_file_path)
    
    # Print the final result as a JSON string
    print(json.dumps(analysis_result, indent=4))