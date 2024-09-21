import matplotlib.pyplot as mp
from Bitcoin_data import df
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
mp.rcParams['font.family'] = 'SimHei'
mp.rcParams['axes.unicode_minus'] = False

root_save_path = '../resource/kmeans_plot/'
# 聚类分析
# 价格波动数据
df['Price_volatility'] = df['High'] - df['Low']
# 聚类特征
features = df[['Price_volatility', 'Volume']]
# 标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 聚类参数选择
import scikitplot as skplot
silhouette_scores = []
k_range = range(2, 11)
model = KMeans()
mp.figure(figsize=(12,8))
mp.grid(True)
skplot.cluster.plot_elbow_curve(model, features, cluster_ranges=k_range)
mp.savefig(root_save_path+ '手肘图.png', format='png', dpi=200)

for i in k_range:
    model = KMeans(n_clusters=i, random_state=15)
    labels = model.fit_predict(features)
    silhouette_scores.append(silhouette_score(features, labels))
mp.figure(figsize=(12,8))
mp.plot(k_range, silhouette_scores, marker='o')
mp.grid(True)
mp.title("聚类轮廓系数图")
mp.xlabel('聚类中心数目')
mp.ylabel("轮廓系数")
mp.savefig(root_save_path+ '轮廓系数.png', format='png', dpi=200)

# 肘部法则，选择参数为5
kmeans = KMeans(n_clusters=5, random_state=15)
df['Cluster'] = kmeans.fit_predict(scaled_features)


# 聚类结果可视化

mp.figure(figsize=(12,8))
sns.boxplot(x='Cluster', y='Price_volatility', hue='Cluster', data=df)
mp.xlabel('聚类')
mp.ylabel('价格波动')
mp.title('不同聚类价格波动分布')
mp.savefig(root_save_path+ '聚类价格波动-箱线图.png', format='png', dpi=200)


mp.figure(figsize=(12,8))
sns.boxplot(x='Cluster', y='Volume', hue='Cluster', data=df)
mp.xlabel('聚类')
mp.ylabel('成交量')
mp.title('不同聚类价格成交量分布')
mp.savefig(root_save_path+ '聚类成交量波动-箱线图.png', format='png', dpi=200)


mp.figure(figsize=(20, 7))
mp.scatter(df['Date'], df['Price_volatility'], c=df['Cluster'], cmap='viridis', label='聚类结果')
mp.title('价格波动聚类分析')
mp.xlabel('日期')
mp.ylabel('价格波动(USD)')
mp.grid(True)
mp.legend()
mp.savefig(root_save_path+ '聚类价格波动.png', format='png', dpi=200)


mp.figure(figsize=(20, 7))
mp.scatter(df['Date'], df['Volume'], c=df['Cluster'], cmap='viridis', label='聚类结果')
mp.title('成交量聚类分析')
mp.xlabel('日期')
mp.ylabel('成交量')
mp.grid(True)
mp.legend()
mp.savefig(root_save_path+ '聚类成交量波动.png', format='png', dpi=200)