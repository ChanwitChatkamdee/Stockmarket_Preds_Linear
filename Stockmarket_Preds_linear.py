import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')[0]['Symbol']

for ticker in tickers[0:1] :
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365)

    history = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    history = history.loc[:,['Open','Close','Volume']]

    history['Prev_Close'] = history.loc[:,'Close'].shift(1)
    history['Prev_Volume'] = history.loc[:,'Volume'].shift(1)


datetimes = history.index.values
weekdays = []

for dt in datetimes:
    dt = datetime.strptime(str(dt), '%Y-%m-%dT%H:%M:%S.000000000')
    weekdays.append(dt.weekday())


history['weekday'] = weekdays
history['50SMA'] = history['Prev_Close'].rolling(50).mean()
history['100SMA'] = history['Prev_Close'].rolling(100).mean()
history['200SMA'] = history['Prev_Close'].rolling(200).mean()

x = history.index.values

plt.figure(figsize=(15,5))
plt.plot(x,history['Prev_Close'], color='blue')
plt.plot(x,history['50SMA'], color='red')
plt.plot(x,history['100SMA'], color='yellow')
plt.plot(x,history['200SMA'], color='green')
plt.show()


def calc_macd(data, len1,len2,len3):
    shortEMA = data.ewm(span=len1, adjust=False).mean()
    longEMA = data.ewm(span=len2, adjust=False).mean()
    MACD = shortEMA - longEMA
    signal = MACD.ewm(span=len3, adjust=False).mean()
    return MACD, signal

MACD, signal = calc_macd(history['Prev_Close'],12,26,9)
history['MACD'] = MACD
history['MACD_signal'] = signal

plt.figure(figsize=(15,3))
colors = np.array(['green'] * len(history['MACD']))
colors[history['MACD']<0] = 'red'
plt.bar(x, history['MACD'], color=colors)
plt.plot(x, history['MACD_signal'],color='blue')
plt.xlim(x[1000],x[1500])
plt.show()


def calc_rsi(data, period):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=period, adjust=False).mean()
    ema_down = down.ewm(com=period, adjust=False).mean()
    rs = ema_up/ema_down
    rsi = 100 - (100/(1+rs))
    return rsi

history['RSI'] = calc_rsi(history['Prev_Close'], 13)
history['RSI_Volume'] = calc_rsi(history['Prev_Volume'], 13)

plt.figure(figsize=(15,3))
plt.plot(x, history['RSI'], color = 'purple')
plt.plot([x[0], x[-1]], [80,80], color = 'green')
plt.plot([x[0], x[-1]], [20,20], color = 'red')
plt.xlim(x[1000],x[1500])
plt.ylim(0,100)
plt.show()

def calc_bollinger(data, period):
    mean = data.rolling(period).mean()
    std = data.rolling(period).std()
    upper_band = np.array(mean) + 2*np.array(std)
    lower_band = np.array(mean) - 2*np.array(std)
    return upper_band, lower_band

upper, lower = calc_bollinger(history['Prev_Close'],20)
history['Upper_Band'] = upper
history['lower_Band'] = lower

plt.figure(figsize=(15,5))
plt.plot(x,history['Prev_Close'], color='blue')
plt.plot(x,history['50SMA'], color='red')
plt.plot(x,history['100SMA'], color='yellow')
plt.plot(x,history['200SMA'], color='green')
plt.plot(x,history['Upper_Band'], color='orange')
plt.plot(x,history['lower_Band'], color='orange')
plt.show()


labels = ['Prev_Close','Prev_Volume','50SMA','100SMA','200SMA','MACD','MACD_signal','RSI','RSI_Volume','Upper_Band','lower_Band']
period = 1
new_labels = [str(period)+'d_'+label for label in labels]
history[new_labels] = history[labels].pct_change(period, fill_method='ffill')

period = 2
new_labels = [str(period)+'d_'+label for label in labels]
history[new_labels] = history[labels].pct_change(period, fill_method='ffill')

period = 5
new_labels = [str(period)+'d_'+label for label in labels]
history[new_labels] = history[labels].pct_change(period, fill_method='ffill')

period = 10
new_labels = [str(period)+'d_'+label for label in labels]
history[new_labels] = history[labels].pct_change(period, fill_method='ffill')


history = history.replace(np.inf,np.nan).dropna()


from sklearn.linear_model import LinearRegression

y = history['Close']
x = history.drop(['Close','Volume'], axis=1).values


num_test = 365
x_train = x[:-1*num_test] #0-3199
y_train = y[:-1*num_test]
x_test = x[-1*num_test:] #3199-3564
y_test = y[-1*num_test:]


model = LinearRegression()
model = model.fit(x_train,y_train)
preds = model.predict(x_test)


plt.figure(figsize=(15,5))
plt.plot(range(len(y_test)),y_test, 'blue')
plt.plot(range(len(preds)),preds, 'red')
plt.show()


def test_it(opens, closes,preds, start_account = 1000, thresh=0):
    account = start_account
    changes = []

    for i in range(len(preds)):
        if (preds[i]-opens[i])/opens[i] >= thresh :
            account += account*(closes[i] - opens[i])/opens[i]
        changes.append(account)
    changes = np.array(changes)
    print(changes)

    plt.plot(range(len(changes)),changes)
    plt.show()

    invest_total = start_account + start_account*(closes[-1]-opens[0])/opens[0]
    print('Invest_Total:',invest_total,str(round((invest_total-start_account)/start_account*100,1 ))+'%')
    print('Algo-Trading Total:',account,str(round((account-start_account)/start_account*100,1 ))+'%')


test_it(x_test.T[0], y_test,preds,1000,0)


