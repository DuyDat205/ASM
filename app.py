import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="ABC Sales Dashboard", layout="wide")

st.title("📊 ABC Manufacturing: Sales Analysis and Forecasting")

df = pd.read_csv("extended_sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Week_Index'] = np.arange(1, len(df) + 1)

filtered = df  # no filters for simplicity

st.subheader("1. Weekly Sales Trend")
fig1, ax1 = plt.subplots()
sns.lineplot(data=filtered, x='Date', y='Sales', ax=ax1, color='blue')
ax1.set_title("Weekly Sales Over Time")
st.pyplot(fig1)

st.subheader("2. Sales Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered['Sales'], bins=20, kde=True, ax=ax2, color='orange')
ax2.set_title("Sales Distribution")
st.pyplot(fig2)

st.subheader("3. 4‑Week Moving Average")
filtered['Sales_MA4'] = filtered['Sales'].rolling(4).mean()
fig3, ax3 = plt.subplots()
sns.lineplot(data=filtered, x='Date', y='Sales', label='Actual', ax=ax3)
sns.lineplot(data=filtered, x='Date', y='Sales_MA4', label='4‑Week MA', color='red', ax=ax3)
ax3.set_title("Sales vs 4‑Week Moving Average")
st.pyplot(fig3)

st.subheader("4. Sales by Month (Boxplot)")
filtered['Month'] = filtered['Date'].dt.month_name()
fig4, ax4 = plt.subplots(figsize=(12,6))
sns.boxplot(data=filtered, x='Month', y='Sales', order=sorted(filtered['Month'].unique(), key=lambda m: pd.to_datetime(m, format='%B')), ax=ax4)
ax4.set_title("Monthly Sales Distribution")
st.pyplot(fig4)

st.subheader("5. Linear Regression Forecast")
X = filtered[['Week_Index']]
y = filtered['Sales']
model = LinearRegression()
model.fit(X, y)
filtered['Predicted_Sales'] = model.predict(X)
rmse = mean_squared_error(y, filtered['Predicted_Sales'], squared=False)

fig5, ax5 = plt.subplots()
sns.lineplot(data=filtered, x='Date', y='Sales', label='Actual', ax=ax5)
sns.lineplot(data=filtered, x='Date', y='Predicted_Sales', label='Predicted', linestyle='--', ax=ax5)
ax5.set_title(f"Actual vs Predicted Sales (RMSE: {rmse:.2f})")
st.pyplot(fig5)

st.caption("Powered by ABC Manufacturing Data Team")
