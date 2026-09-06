"""
Generates a realistic synthetic sales dataset with trend + weekly & yearly
seasonality + promotions + noise, in the same spirit as the Kaggle
'Store Sales - Time Series Forecasting' dataset.

Run once: python generate_sample_data.py
Output:   data/sales_data.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)

start_date = "2021-01-01"
end_date = "2024-12-31"
dates = pd.date_range(start_date, end_date, freq="D")

stores = ["Store_A", "Store_B", "Store_C"]
categories = ["Grocery", "Electronics", "Clothing"]

rows = []
for store in stores:
    store_base = np.random.uniform(180, 320)
    for category in categories:
        cat_mult = {"Grocery": 1.3, "Electronics": 0.8, "Clothing": 1.0}[category]
        for i, d in enumerate(dates):
            trend = i * 0.03  # slow upward trend over the years
            weekly = 40 * np.sin(2 * np.pi * d.dayofweek / 7) + (60 if d.dayofweek in (4, 5) else 0)
            yearly = 70 * np.sin(2 * np.pi * d.dayofyear / 365.25 + 1.2)
            holiday_boost = 150 if d.month == 12 and d.day >= 15 else 0
            promo = 1 if np.random.rand() < 0.08 else 0
            promo_boost = 80 * promo
            noise = np.random.normal(0, 25)
            sales = max(0, (store_base + trend + weekly + yearly + holiday_boost + promo_boost + noise) * cat_mult)
            rows.append([d, store, category, promo, round(sales, 2)])

df = pd.DataFrame(rows, columns=["date", "store", "category", "promotion", "sales"])
df.to_csv("data/sales_data.csv", index=False)
print(f"Saved {len(df):,} rows to data/sales_data.csv")
print(df.head())
