# How does the daily movement of the overall stock market affect the daily movement of a specific company's stock?
import yfinance as yf
import statsmodels.api as sm

# 1. Download historical stock data
data = yf.download(['^GSPC', 'AAPL'], start='2020-01-01', end='2025-01-01')

# 2. Calculate the Daily Return
returns = data['Close'].pct_change().dropna()

# 3. Defini X and Y Variables
Y = returns['AAPL']
X = returns['^GSPC']

# 4. Intercept
X = sm.add_constant(X)

# 5. Fit the OLS Model
model = sm.OLS(Y, X).fit()

# 6. Print the Summary Table
print(model.summary())
