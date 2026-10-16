import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 1. Generate Synthetic E-Commerce Dataset
np.random.seed(42)
n_rows = 1000

customer_ids = [f"CUST_{np.random.randint(100, 150)}" for _ in range(n_rows)]
categories = ["Electronics", "Apparel", "Home & Kitchen", "Books", "Beauty"]
product_categories = np.random.choice(categories, n_rows, p=[0.3, 0.3, 0.2, 0.1, 0.1])
quantities = np.random.randint(1, 5, n_rows)

# Assign realistic unit prices per category
price_map = {
    "Electronics": 250,
    "Apparel": 45,
    "Home & Kitchen": 80,
    "Books": 20,
    "Beauty": 35,
}
unit_prices = [
    price_map[cat] * np.random.uniform(0.8, 1.2) for cat in product_categories
]

start_date = pd.to_datetime("2025-01-01")
end_date = pd.to_datetime("2025-12-31")
order_dates = [
    start_date + pd.Timedelta(days=int(np.random.randint(0, 365)))
    for _ in range(n_rows)
]

df = pd.DataFrame(
    {
        "CustomerID": customer_ids,
        "OrderDate": order_dates,
        "Category": product_categories,
        "Quantity": quantities,
        "UnitPrice": unit_prices,
    }
)
df.to_csv('SampleData.csv')
# 2. Data Cleaning & Feature Engineering
df["TotalRevenue"] = df["Quantity"] * df["UnitPrice"]
df["Month"] = df["OrderDate"].dt.to_period("M").astype(str)

print("--- Data Sample ---")
print(df.head(), "\n")

# 3. Exploratory Data Analysis & Aggregations
print("--- Total Revenue by Category ---")
category_revenue = (
    df.groupby("Category")["TotalRevenue"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
print(category_revenue, "\n")

print("--- Top 5 Customers by Lifetime Spend ---")
top_customers = (
    df.groupby("CustomerID")["TotalRevenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
)
print(top_customers, "\n")

# 4. Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Revenue by Category
sns.barplot(
    data=category_revenue,
    x="Category",
    y="TotalRevenue",
    hue="Category",
    legend=False,
    palette="Blues_r",
    ax=axes[0],
)
axes[0].set_title("Total Revenue by Product Category", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Revenue ($)")
axes[0].tick_params(axis="x", rotation=30)

# Chart 2: Monthly Revenue Trend
monthly_trend = df.groupby("Month")["TotalRevenue"].sum().sort_index().reset_index()
sns.lineplot(
    data=monthly_trend,
    x="Month",
    y="TotalRevenue",
    marker="o",
    color="teal",
    linewidth=2.5,
    ax=axes[1],
)
axes[1].set_title("Monthly Revenue Trend", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Revenue ($)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()