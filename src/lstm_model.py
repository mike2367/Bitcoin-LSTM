from sklearn.preprocessing import MinMaxScaler
from Bitcoin_data import df
import torch
from torch import nn
import numpy as np
closing_data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close_data = scaler.fit_transform(closing_data)

STEP_LENGTH = 5

# 设定训练与窗口长度
def create_dataset(data, time_step=STEP_LENGTH):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)


X, Y = create_dataset(scaled_close_data)

# 70%作为训练集
train_size = int(len(X) * 0.7)
# 调整至RNN输入格式
train_x = X[:train_size].reshape(-1, 1, STEP_LENGTH)
train_y = Y[:train_size].reshape(-1, 1, 1)
# float将float64转换为float32,避免RTE
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()


# 2层隐藏层LSTM
class LSTM_regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # 全连接层输出
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)
        return x