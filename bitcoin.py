import pandas as pd
btc_df = pd.read_csv('data/bitcoin_price.csv')
print(btc_df.head())
print(btc_df['symbol'].unique())
btc_df['time'] = pd.to_datetime(btc_df['time'], unit='ms')
print(btc_df.head())
btc_df.info()
btc_df.set_index('time', inplace=True)
btc_df['close'].plot(logy=True)