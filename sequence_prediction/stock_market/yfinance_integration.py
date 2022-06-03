from datetime import datetime

import pandas as pd
# from pandas_datareader.data import DataReader

"""
Trzeba zaimportować tą bibliotekę przez: 
pip install yfinance
"""

import yfinance as yf

end = datetime.now()
start = datetime(end.year-20, end.month-2, end.day)

g = yf.download('EURPLN=X', start, end)  # S&P500: ^GSPC, DAX: ^GDAXI, EURUSD=X, EURPLN=X
print(type(g))
print(g)
print(g['Close'])
print(g['Close'].tolist())  # lista cen zamknięcia
print(g.axes[0].tolist())   # lista dat

# g.to_csv('tesla.csv')


"""
raw_data = yfinance.download (tickers = "^GSPC", start = "1994-01-07", 
                              end = "2019-09-01", interval = "1d")
"""
