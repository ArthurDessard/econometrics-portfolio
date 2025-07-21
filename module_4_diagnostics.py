import yfinance as yf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# 1. Recreate the MLR model
tickers = ['AAPL', '^GSPC', 'MSFT']
data = yf.download(tickers, start='2020-01-01', end='2025-01-01')
returns = data['Close'].pct_change().dropna()
Y = returns['AAPL']
X = returns[['^GSPC', 'MSFT']]
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()

# 2. Run the Breusch-Pagan Test
bp_test_results = het_breuschpagan(model.resid, model.model.exog)
labels = ['Lagrange Multiplier Statistic', 'p-value', 'F-value', 'F p-value']
print("--- Breusch-Pagan Test Results ---")
print(dict(zip(labels, bp_test_results)))
print("\n" + "-" * 30 + "\n")

# 3. FIX: Rerun the regression with Robust Standrd Errors
robust_model = sm.OLS(Y, X).fit(cov_type='HC1')

# 4. Compare the results
print("--- Original Model Summary ---")
print(model.summary())
print("\n" + "=" * 80 + "\n")
print("--- Model with Robust Standard Errors ---")
print(robust_model.summary())

