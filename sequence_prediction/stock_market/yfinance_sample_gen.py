from datetime import datetime, timedelta
from random import randint

import pandas as pd

# from pandas_datareader.data import DataReader

"""
Trzeba zaimportować tą bibliotekę przez: 
pip install yfinance

przykłady tickerów: 
S&P500: ^GSPC 
DAX: ^GDAXI
EURUSD=X
EURPLN=X

Tesla: TSLA


"""

import yfinance as yf


def normalize(array, value, scale):
    s_array = [x / scale for x in array]
    s_value = value / scale
    return s_array, s_value


def get_samples(date_from: datetime, date_to: datetime, history_len, n_samples, ticker):
    g = yf.download(ticker, date_from, date_to)
    data = g['Close'].tolist()
    n = len(data)
    print(f'pulled {n} data points')
    samples, outputs = [], []
    for _ in range(n_samples):
        st = randint(0, n - 1 - history_len)
        history = data[st:st + history_len].copy()
        predict = data[st + history_len]
        history, predict = normalize(history, predict, history[0])
        samples.append(history)
        outputs.append(predict)
    return samples, outputs


if __name__ == '__main__':
    end = datetime.now()
    # start = datetime(end.year, end.month, end.day)
    start = end - timedelta(days=150)

    s, o = get_samples(start, end, history_len=3, n_samples=5, ticker='AAPL')
    for his, nxt in zip(s, o):
        print(his, nxt)
