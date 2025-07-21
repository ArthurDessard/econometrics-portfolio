import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Get the data for X
tickers = ['^GSPC', 'MSFT']
data = yf.download(tickers, start='2020-01-01', end='2025-01-01')
returns = data['Close'].pct_change().dropna()
X = returns[['^GSPC', 'MSFT']]

# 2. Calculate the Correlation Matrix
print("--- Correlation Matrix ---")
print(X.corr())
print("\n" + "-" * 30 + "\n")

# 3. Calculate the Variance Inflation Factor (VIF)
print("--- Variance Inflation Factor (VIF) ---")
X_vif = sm.add_constant(X)
vif_df = pd.DataFrame()
vif_df["feature"] = X_vif.columns
vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_df)