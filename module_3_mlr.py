# Controlling for the overall market's movement, does the performance of a major peer like Microsoft have an additional, separate effect on Apple's returns?
import yfinance as yf
import statsmodels.api as sm

# 1. Download data for multiple tickers
tickers = ['AAPL', '^GSPC', 'MSFT']
data = yf.download(tickers, start='2020-01-01', end='2025-01-01')

# 2. Calculate daily returns
returns = data['Close'].pct_change().dropna()

# 3. Define X and Y variables
Y = returns['AAPL']
X = returns[['^GSPC', 'MSFT']]

# 4. Add the constant
X = sm.add_constant(X)

# 5. Fit the MLR model
model = sm.OLS(Y, X).fit()

# 6. Print the result
print(model.summary())
