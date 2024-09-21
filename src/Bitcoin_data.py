import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
df= pd.read_csv('../resource/BTC-USD (2014-2024).csv')

# print(df.info())
# print(df.head())

# 查看数据后发现第3413行为null,利用excel填充十日均值

# 孤立森林异常值检测
from sklearn.ensemble import IsolationForest

columns_to_detect = df.columns[1:]

iso_forest = IsolationForest(n_estimators=100, contamination=0.01)
iso_forest.fit(df[columns_to_detect])
outliers = iso_forest.predict(df[columns_to_detect])
df['outlier'] = outliers

# 35个异常值，选择不填充，替换异常值为NaN
df.loc[df['outlier'] == -1, 'Open'] = np.nan
df = df.dropna()
df = df.reindex()

# 调整日期格式
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

start_date = df['Date'].min()
end_date = df['Date'].max()

# 检查缺失日期
full_date_range = pd.date_range(start=start_date, end=end_date)
missing_dates = full_date_range.difference(df['Date'])

# print(missing_dates)