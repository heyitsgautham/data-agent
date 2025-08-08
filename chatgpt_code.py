import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# This script performs an analysis on the list of highest-grossing films.
# It reads data from a local CSV file, processes it to answer several questions,
# and generates a plot. The final output is a JSON array of strings.

# --- Data Loading and Preparation ---
# Load the dataset from the provided CSV file into a pandas DataFrame.
# The data summary indicates the columns are already in a clean, numeric format where appropriate.
# We will ensure the data types are correct for robust calculations.
try:
    df = pd.read_csv('data.csv')
    # Explicitly convert columns to their expected numeric types to prevent errors.
    df['Worldwide gross'] = pd.to_numeric(df['Worldwide gross'])
    df['Year'] = pd.to_numeric(df['Year'])
    df['Rank'] = pd.to_numeric(df['Rank'])
    df['Peak'] = pd.to_numeric(df['Peak'])
except (FileNotFoundError, KeyError) as e:
    # Handle potential errors during file loading or column access.
    print(json.dumps({"error": f"Failed to load or process data.csv: {e}"}))
    exit()

# --- Workflow: Answering the Questions ---

# **Question 1: How many $2 bn movies were released before 2000?**
# 1. Filter rows where 'Worldwide gross' is >= 2,000,000,000.
# 2. From the filtered data, select rows where 'Year' is < 2000.
# 3. Count the number of resulting rows.
movies_over_2bn_before_2000 = df[(df['Worldwide gross'] >= 2_000_000_000) & (df['Year'] < 2000)]
answer1 = str(len(movies_over_2bn_before_2000))

# **Question 2: Which is the earliest film that grossed over $1.5 bn?**
# 1. Filter rows where 'Worldwide gross' is >= 1,500,000,000.
# 2. Sort the filtered data by 'Year' in ascending order.
# 3. Select the first row and retrieve its 'Title'.
movies_over_1_5bn = df[df['Worldwide gross'] >= 1_500_000_000]
# To handle cases where the dataframe might be empty, we use a try-except block.
if not movies_over_1_5bn.empty:
    earliest_movie = movies_over_1_5bn.sort_values(by='Year', ascending=True).iloc[0]
    answer2 = str(earliest_movie['Title'])
else:
    answer2 = "No film found"

# **Question 3: What's the correlation between the Rank and Peak?**
# 1. Select the 'Rank' and 'Peak' columns.
# 2. Calculate the Pearson correlation coefficient between them.
correlation = df['Rank'].corr(df['Peak'])
answer3 = str(correlation)

# **Question 4: Draw a scatterplot of Rank and Peak with a regression line.**
# 1. Create a scatterplot using seaborn's regplot for 'Rank' vs 'Peak'.
# 2. Style the regression line to be dotted and red.
# 3. Render the plot into an in-memory PNG image.
# 4. Encode the image data into a base64 string and format it as a data URI.
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(
    x='Rank',
    y='Peak',
    data=df,
    ax=ax,
    line_kws={'color': 'red', 'linestyle': '--'},
    scatter_kws={'alpha': 0.7}
)
ax.set_title('Rank vs. Peak of Highest-Grossing Films')
ax.set_xlabel('Overall Rank')
ax.set_ylabel('Peak Rank Achieved')

# Save the plot to a bytes buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
plt.close(fig) # Close the figure to free up memory

# Encode the image to base64
image_base64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
buf.close()

# Format as a data URI
answer4 = f"data:image/png;base64,{image_base64_string}"

# --- Final Output ---
# Consolidate all answers into a list of strings.
final_answers = [
    answer1,
    answer2,
    answer3,
    answer4
]

# Print the final result as a JSON array, as required.
print(json.dumps(final_answers, indent=4))