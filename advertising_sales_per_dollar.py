import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("advertising.csv")

# Sales per dollar spent (more meaningful than ROI here)
df['TV_sales_per_dollar'] = df['sales'] / df['TV']
df['Radio_sales_per_dollar'] = df['sales'] / df['radio']
df['Newspaper_sales_per_dollar'] = df['sales'] / df['newspaper']

# Handle any division by zero (if any spend is 0)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['TV_sales_per_dollar', 'Radio_sales_per_dollar', 'Newspaper_sales_per_dollar'])

# Average sales generated per $1 spent
sales_per_dollar = pd.DataFrame({
    'Channel': ['TV', 'Radio', 'Newspaper'],
    'Avg_Sales_per_Dollar': [
        df['TV_sales_per_dollar'].mean(),
        df['Radio_sales_per_dollar'].mean(),
        df['Newspaper_sales_per_dollar'].mean()
    ]
}).round(3)

print("Average Sales Units per $1 Spent:")
print(sales_per_dollar)

# Plot
sales_per_dollar.plot(
    x='Channel', 
    y='Avg_Sales_per_Dollar', 
    kind='bar', 
    figsize=(10, 6), 
    color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
    legend=False
)

plt.title('Sales Units Generated per $1 of Ad Spend', fontsize=18, pad=20)
plt.ylabel('Sales Units per $1 Spent', fontsize=14)
plt.xlabel('Advertising Channel', fontsize=14)
plt.xticks(rotation=0)

# Add value labels on bars
for i, v in enumerate(sales_per_dollar['Avg_Sales_per_Dollar']):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("sales_per_dollar.png", dpi=300, bbox_inches='tight')
plt.show()