import yfinance as yf

company = yf.Ticker("TSLA") # Company name tag

print("\nHistorical Market Data\n")

# get historical market data
print(company.history(period="max"))

print("\nStock Info\n")

# get stock info
print(company.info)

print("\nActions\n")

# show actions (dividends, splits)
print(company.actions)

print("\nDividends\n")

# show dividends
print(company.dividends)

print("\nSplits\n")

# show splits
print(company.splits)