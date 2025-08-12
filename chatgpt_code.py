
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

df = pd.read_csv('data.csv')

df['Worldwide gross'] = pd.to_numeric(df['Worldwide gross'])
df['Year'] = pd.to_numeric(df['Year'])
df['Rank'] = pd.to_numeric(df['Rank'])
df['Peak'] = pd.to_numeric(df['Peak'])

q1_df = df[(df['Worldwide gross'] >= 2000000000) & (df['Year'] < 2000)]
answer1 = str(len(q1_df))

q2_df = df[df['Worldwide gross'] > 1500000000]
q2_df_sorted = q2_df.sort_values(by='Year', ascending=True)
answer2 = q2_df_sorted.iloc[0]['Title']

correlation = df['Rank'].corr(df['Peak'])
answer3 = str(correlation)

fig, ax = plt.subplots(figsize=(5, 3.5), dpi=80)
x = df['Rank']
y = df['Peak']
ax.scatter(x, y, s=15)
m, b = np.polyfit(x, y, 1)
ax.plot(x, m * x + b, 'r--')
ax.set_xlabel('Rank')
ax.set_ylabel('Peak')
ax.set_title('Rank vs. Peak Scatter Plot')
plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
plt.close(fig)
answer4 = f"data:image/png;base64,{image_base64}"

results = [answer1, answer2, answer3, answer4]
print(json.dumps(results))
