import matplotlib.pyplot as mp
import matplotlib.dates as mdates
from Bitcoin_data import df
from matplotlib.ticker import MaxNLocator
mp.rcParams['font.family'] = 'SimHei'
mp.rcParams['axes.unicode_minus'] = False

root_save_path = '../resource/saved_plot/'
# 可视化
# 按照月份重采样, 开盘价，收盘价，最高，最低价
btc_data_monthly = df.resample('1M', on='Date').agg(
    {
        'Open':'first',
        'High': 'max',
        'Low':'min',
        'Close': 'last'
    }
).dropna()

# 月度K线图
fig, ax = mp.subplots(figsize=(20, 7))

for i, row in btc_data_monthly.iterrows():
    color = 'red' if row['Close'] >= row['Open'] else "green"
    ax.plot([row.name, row.name], [row['Low'], row['High']], color=color)
    ax.plot([row.name, row.name], [row['Open'], row['Close']], linewidth=6,
            color=color)
ax.xaxis_date()
ax.xaxis.set_major_locator(MaxNLocator(10))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
mp.xticks(rotation=0)
mp.xlabel('日期')
mp.ylabel('价格 (USD)')
mp.title('比特币月K线图 (2014-2024)')
mp.grid(True)
mp.savefig(root_save_path+ 'K.png', format='png', dpi=200)

# 收盘价趋势分析
mp.figure(figsize=(20, 6))
mp.plot(df['Date'], df['Close'], label='收盘价', color='blue')
mp.xlabel('日期')
mp.ylabel('收盘价')
mp.title("比特币收盘价趋势(2014-2024)")
mp.xticks(rotation=0)
mp.legend()
mp.grid(True)
mp.savefig(root_save_path+ '收盘价趋势.png', format='png', dpi=200)

# 考虑数据量，绘制30天和100天移动均线
df_30day = df['Close'].rolling(window=30).mean()
df_100day = df['Close'].rolling(window=100).mean()

mp.figure(figsize=(20,7))
mp.plot(df['Date'],df['Close'],label='收盘价',color='blue')
mp.plot(df['Date'], df_30day, label='30日收盘均线', color='orange')
mp.plot(df['Date'], df_100day, label='100日收盘均线', color='green')
mp.xlabel('日期')
mp.ylabel('价格')
mp.title("比特币均线分析(2014-2024)")
mp.xticks(rotation=0)
mp.legend()
mp.grid(True)
mp.savefig(root_save_path+ '移动均线分析.png', format='png', dpi=200)

# rsi指数，基于收盘价判断价格变化的快慢
# 默认周期为14
def calculate_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 -(100 / (1 + rs))

df['RSI'] = calculate_rsi(df['Close'])

mp.figure(figsize=(20, 6))
mp.plot(df['Date'], df['RSI'], label='RSI')
mp.xlabel('日期')
mp.ylabel('RSI')
mp.title('比特币相对强弱指数(RSI)')
mp.axhline(y=70, color='r', linestyle='--', label='超买(70)')
mp.axhline(y=30, color='r', linestyle='--', label='超卖(30)')
mp.legend()
mp.grid(True)
mp.savefig(root_save_path+ 'RSI指数分析.png', format='png', dpi=200)

# 计算6个月周期布林带
btc_data_monthly['std'] = btc_data_monthly['Close'].rolling(window=6).std()
btc_data_monthly['upper'] = btc_data_monthly['Close'].rolling(window=6).mean() + 2 * btc_data_monthly['std']
btc_data_monthly['lower'] = btc_data_monthly['Close'].rolling(window=6).mean() - 2 * btc_data_monthly['std']
mp.figure(figsize=(20, 6))
mp.plot(btc_data_monthly.index, btc_data_monthly['upper'], label='Upper', linestyle='--', color='orange')
mp.plot(btc_data_monthly.index, btc_data_monthly['lower'], label='Lower', linestyle='--', color='green')
mp.plot(btc_data_monthly.index, btc_data_monthly['Close'], label='Close', color='blue')
mp.xlabel('日期')
mp.ylabel('价格(USD)')
mp.title('比特币6月周期布林带')
mp.legend()
mp.grid(True)
mp.savefig(root_save_path+ '6月布林带分析.png', format='png', dpi=200)

# 成交量趋势图
mp.figure(figsize=(14,8))
mp.plot(df['Date'], df['Volume'], label='成交量', color='green')
mp.xlabel('日期')
mp.ylabel('成交量')
mp.title('比特币成交量走势')
mp.legend()
mp.grid(True)
mp.savefig(root_save_path+ '成交量走势.png', format='png', dpi=200)