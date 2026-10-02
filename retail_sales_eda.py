# # Retail Sales — Exploratory Data Analysis
#
# **Objective:** Uncover sales patterns, customer behaviour trends, and actionable
# business insights from a retail sales dataset.
#
# **Stack:** Python · pandas · matplotlib · seaborn
#
# ---
#
# ### Dataset
#
# This notebook is written to work with any retail transaction dataset that has
# (roughly) these fields: an order date, a customer id, customer age, customer
# gender, a product category/name, quantity, unit price, a discount, a computed
# sales/revenue amount, and a region.
#
# Good sources on Kaggle (search these on kaggle.com and download the CSV into
# the same folder as this notebook):
# - **"Superstore Sales Dataset"**
# - **"Retail Sales Dataset"**
# - **"E-Commerce Sales Dataset"**
#
# Drop the CSV in this folder, set `DATA_PATH` in the cell below to its filename,
# and re-run. If no file is found at `DATA_PATH`, the notebook automatically
# **generates a realistic synthetic retail dataset** instead, so every cell below
# still runs end-to-end and you can see exactly what the analysis produces
# before you plug in real data.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---- config ----
DATA_PATH = "retail_sales_dataset.csv"   # <- change this to your Kaggle CSV filename
RANDOM_SEED = 42

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titleweight"] = "bold"
pd.set_option("display.max_columns", 50)


def generate_synthetic_retail_data(n_rows=6000, seed=RANDOM_SEED):
    """Creates a realistic synthetic retail transactions dataset so this
    notebook is fully runnable even without a Kaggle download."""
    rng = np.random.default_rng(seed)

    categories = {
        "Electronics": ["Wireless Earbuds", "Smartphone Case", "Bluetooth Speaker",
                         "Laptop Stand", "USB-C Cable", "Smartwatch"],
        "Apparel": ["Cotton T-Shirt", "Denim Jeans", "Running Shoes",
                    "Hooded Jacket", "Wool Sweater"],
        "Home & Kitchen": ["Non-Stick Pan", "Coffee Maker", "LED Desk Lamp",
                            "Storage Bins", "Blender"],
        "Beauty": ["Face Moisturizer", "Shampoo", "Lipstick Set", "Sunscreen SPF50"],
        "Sports": ["Yoga Mat", "Dumbbell Set", "Cycling Helmet", "Water Bottle"],
        "Books": ["Fiction Novel", "Cookbook", "Self-Help Guide"],
    }
    category_price_range = {
        "Electronics": (15, 250), "Apparel": (8, 90), "Home & Kitchen": (10, 150),
        "Beauty": (5, 60), "Sports": (7, 120), "Books": (6, 35),
    }
    regions = ["North", "South", "East", "West", "Central"]
    genders = ["Male", "Female", "Other"]
    ship_modes = ["Standard", "Express", "Same-Day"]

    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    # give weekends & Nov/Dec a sales boost (seasonality + weekly pattern)
    day_weights = np.array([
        1.4 if d.month in (11, 12) else (1.15 if d.weekday() >= 5 else 1.0)
        for d in dates
    ])
    order_dates = rng.choice(dates, size=n_rows, p=day_weights / day_weights.sum())

    rows = []
    n_customers = 900
    customer_ages = rng.integers(18, 70, size=n_customers)
    customer_genders = rng.choice(genders, size=n_customers, p=[0.47, 0.48, 0.05])

    for i in range(n_rows):
        cust_id = rng.integers(0, n_customers)
        category = rng.choice(list(categories.keys()), p=[0.28, 0.22, 0.18, 0.14, 0.12, 0.06])
        product = rng.choice(categories[category])
        lo, hi = category_price_range[category]
        unit_price = round(rng.uniform(lo, hi), 2)
        quantity = int(rng.choice([1, 1, 1, 2, 2, 3, 4], p=[0.32, 0.18, 0.15, 0.15, 0.1, 0.06, 0.04]))
        discount = round(rng.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2],
                                     p=[0.45, 0.1, 0.1, 0.15, 0.1, 0.06, 0.04]), 2)
        sales = round(unit_price * quantity * (1 - discount), 2)

        rows.append({
            "OrderID": 100000 + i,
            "OrderDate": pd.Timestamp(order_dates[i]),
            "CustomerID": f"C{cust_id:04d}",
            "CustomerAge": int(customer_ages[cust_id]),
            "CustomerGender": customer_genders[cust_id],
            "ProductCategory": category,
            "ProductName": product,
            "Quantity": quantity,
            "UnitPrice": unit_price,
            "Discount": discount,
            "Sales": sales,
            "Region": rng.choice(regions),
            "ShipMode": rng.choice(ship_modes, p=[0.6, 0.3, 0.1]),
        })

    df_ = pd.DataFrame(rows).sort_values("OrderDate").reset_index(drop=True)

    # sprinkle a few realistic nulls so the missing-value check has something to find
    for col in ["CustomerAge", "Discount", "ShipMode"]:
        null_idx = rng.choice(df_.index, size=int(0.01 * len(df_)), replace=False)
        df_.loc[null_idx, col] = np.nan

    return df_


data_path = Path(DATA_PATH)
if data_path.exists():
    df = pd.read_csv(data_path)
    print(f"Loaded real dataset from '{DATA_PATH}'.")
else:
    df = generate_synthetic_retail_data()
    print(f"'{DATA_PATH}' not found — using a generated synthetic dataset "
          f"({len(df):,} rows) so the notebook runs end-to-end.\n"
          f"Download a real dataset from Kaggle and point DATA_PATH at it to replace this.")

df.head()


# ## 1. Initial Inspection

print("Shape (rows, columns):", df.shape)
print("\nColumn dtypes:")
print(df.dtypes)


null_summary = pd.DataFrame({
    "null_count": df.isnull().sum(),
    "null_pct": (df.isnull().mean() * 100).round(2),
}).sort_values("null_count", ascending=False)
null_summary


# Make sure OrderDate is a real datetime, and build helper time columns used later
if "OrderDate" in df.columns:
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    df["YearMonth"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()
    df["Quarter"] = df["OrderDate"].dt.to_period("Q").astype(str)
    df["DayOfWeek"] = df["OrderDate"].dt.day_name()

df.info()


# **Observations**
# - The dataset has the shape printed above, with a mix of numeric (`Quantity`,
#   `UnitPrice`, `Discount`, `Sales`, `CustomerAge`) and categorical
#   (`ProductCategory`, `Region`, `CustomerGender`, `ShipMode`) columns.
# - A small number of nulls appear in `CustomerAge`, `Discount`, and `ShipMode`
#   — low enough (~1%) to drop or impute without materially affecting results;
#   we'll leave them as-is and let pandas/seaborn skip them where relevant.

# ## 2. Descriptive Statistics (Numerical Columns)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

desc = df[numeric_cols].agg(["mean", "median", "std", "min", "max"]).T
desc["mode"] = df[numeric_cols].mode().iloc[0]
desc = desc[["mean", "median", "mode", "std", "min", "max"]].round(2)
desc


# **Observations**
# - Compare `mean` vs `median` for `Sales`: if the mean sits noticeably above the
#   median, revenue per order is right-skewed — a handful of large orders pull
#   the average up, which is typical in retail.
# - A high `std` relative to the mean for `Sales` or `Quantity` signals wide
#   variance in basket size — worth segmenting customers by order value later.

# ## 3. Time Series Analysis — Monthly & Quarterly Sales Trends

monthly_sales = df.groupby("YearMonth")["Sales"].sum().reset_index()

plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly_sales, x="YearMonth", y="Sales", marker="o", linewidth=2)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


quarterly_sales = df.groupby("Quarter")["Sales"].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=quarterly_sales, x="Quarter", y="Sales", marker="o",
             linewidth=2, color="darkorange")
plt.title("Quarterly Sales Trend")
plt.xlabel("Quarter")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# **Observations**
# - Look for a consistent seasonal peak (e.g. Nov–Dec, holiday shopping) and any
#   quarter that consistently underperforms — that's a candidate for a targeted
#   promotion.
# - A generally upward slope across months indicates business growth; a flat or
#   declining slope alongside rising customer counts would suggest *shrinking
#   order value* rather than shrinking demand — check average order value
#   (`Sales` / order count) per month to confirm which one it is.

# ## 4. Customer Demographics

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["CustomerAge"].dropna(), bins=15, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Customer Age Distribution")
axes[0].set_xlabel("Age")

gender_counts = df["CustomerGender"].value_counts()
axes[1].pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%",
            startangle=90, colors=sns.color_palette("pastel"))
axes[1].set_title("Gender Breakdown")

plt.tight_layout()
plt.show()


age_bins = [17, 25, 35, 45, 55, 100]
age_labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
df["AgeGroup"] = pd.cut(df["CustomerAge"], bins=age_bins, labels=age_labels)

age_group_sales = df.groupby("AgeGroup", observed=True)["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(x=age_group_sales.index, y=age_group_sales.values, hue=age_group_sales.index,
            palette="viridis", legend=False)
plt.title("Total Sales by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()


# **Observations**
# - Note which age bracket contributes the most revenue — that's your core
#   demographic to design campaigns around.
# - If one gender or age group is under-represented in the *customer base* but
#   spends more *per order* (check with a groupby on `Sales` mean), that's a
#   high-value niche worth targeted marketing rather than broad campaigns.

# ## 5. Product Analysis

top_products = (
    df.groupby("ProductName")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.values, y=top_products.index, hue=top_products.index,
            palette="crest", legend=False)
plt.title("Top 10 Best-Selling Products by Revenue")
plt.xlabel("Total Sales")
plt.ylabel("Product")
plt.tight_layout()
plt.show()


category_revenue = df.groupby("ProductCategory")["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(x=category_revenue.index, y=category_revenue.values, hue=category_revenue.index,
            palette="flare", legend=False)
plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# **Observations**
# - The top 10 list shows where revenue concentration lives — if a small number
#   of SKUs dominate, that's inventory risk (stock-outs hurt disproportionately)
#   as well as a marketing opportunity (bundle top sellers with slower movers).
# - Compare category revenue against category *order count*: a category with
#   high revenue but few orders is being carried by high price points, not
#   volume — different pricing/promo strategy than a high-volume, low-margin
#   category.

# ## 6. Correlation Matrix — Numerical Variables

corr = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
            linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# **Observations**
# - `Quantity` and `UnitPrice` typically correlate weakly or negatively with
#   each other — customers buy more units of cheaper items.
# - `Discount` correlating negatively with `Sales` per unit (but not necessarily
#   with *total* revenue, since discounts can drive higher quantities) tells you
#   whether your promotions are actually growing revenue or just eroding margin
#   — worth a dedicated follow-up analysis if the correlation is strong.

# ## 7. Additional Insight — Day-of-Week × Discount Behaviour
#
# Beyond the required charts, this heatmap checks whether **discount depth
# varies by day of the week** and whether that lines up with when people
# actually buy — a pattern that's easy to miss in a top-line trend chart but
# directly actionable for promotion scheduling.

dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

pivot = df.pivot_table(index="DayOfWeek", columns="ProductCategory",
                        values="Discount", aggfunc="mean", observed=True)
pivot = pivot.reindex(dow_order)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5)
plt.title("Average Discount Rate by Day of Week and Category")
plt.xlabel("Product Category")
plt.ylabel("Day of Week")
plt.tight_layout()
plt.show()


# **Observations**
# - If deep discounts cluster on days that already have naturally high traffic
#   (e.g. weekends), you may be giving away margin on sales you'd have made
#   anyway — discounting under-performing weekdays instead could lift overall
#   traffic more efficiently than discounting already-strong days.

# ## Conclusion & Business Recommendations
#
# **Summary of key findings**
# 1. Sales show a clear seasonal/monthly pattern with identifiable peak and
#    trough periods (see Section 3).
# 2. Revenue is concentrated in a handful of top products and 1–2 dominant
#    product categories (Section 5).
# 3. Specific age groups and discount patterns are disproportionately driving
#    — or dragging on — revenue (Sections 4, 6, 7).
#
# **Actionable recommendations**
# 1. **Shift inventory and marketing spend toward peak months identified in the
#    time-series analysis** (e.g. pre-holiday stock-up and a targeted email/ads
#    push 4–6 weeks ahead of the identified peak), and run a small promotion in
#    the weakest month to smooth the trough rather than let it go unaddressed.
# 2. **Double down on the top 10 products and leading category** with bundling
#    ("frequently bought together" with top sellers) and safety-stock buffers,
#    while testing a clearance or bundle push on the lowest-revenue category to
#    decide whether to keep, reposition, or discontinue it.
# 3. **Re-target discounting using the day-of-week × category pattern**: pull
#    discount depth back on days/categories that already sell well without
#    incentives, and reallocate that margin to underperforming days/categories
#    where a discount is more likely to create *incremental* revenue rather than
#    subsidize sales that would have happened anyway.
# 4. *(Optional 4th)* **Build a lightweight customer segmentation** (age group ×
#    average order value) to prioritize retention campaigns on the segment with
#    the highest revenue-per-customer, since acquiring a similar new customer is
#    typically more expensive than retaining an existing high-value one.
