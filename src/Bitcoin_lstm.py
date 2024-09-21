from Bitcoin_data import df
import matplotlib.pyplot as mp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import r2_score
mp.rcParams['font.family'] = 'SimHei'
mp.rcParams['axes.unicode_minus'] = False

# LSTM 预测模型
import torch
from torch import nn
# 安排GPU
device = torch.cuda.device('cuda')

closing_data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_close_data = scaler.fit_transform(closing_data)

# 设定训练与窗口长度
def create_dataset(data, time_step=5):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i+time_step)])
        Y.append(data[i+time_step])
    return np.array(X), np.array(Y)


X, Y = create_dataset(scaled_close_data)

# 70%作为训练集
train_size = int(len(X) * 0.7)
# 调整至RNN输入格式
train_x = X[:train_size].reshape(-1, 1,5)
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

if __name__ == '__main__':
    t0 = time.time()
    model = LSTM_regression(5, 8, output_size=1, num_layers=2)
    model=model.cuda()
    # 计算总参数量
    model_total = sum([param.nelement() for param in model.parameters()])

    print("number of total parameters: %.8fM" % (model_total / 1e6))

    train_loss = []
    # mean square error
    loss_function = nn.MSELoss()
    loss_function = loss_function.cuda()
    # betas 为Adam算法参数，weight_decay防止过拟合，一般取1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=0)

    for i in range(200):
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        out = model(train_x)
        loss = loss_function(out, train_y)
        loss.backward()
        # optimizer更新参数
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())

        # 记录训练过程
        with open('../resource/train_result/log.txt', 'a+') as f:
            f.write('{}-{}\n'.format(i+1, loss.item()))
        print('Epoch: {}, Loss:{:5f}'.format(i+1, loss.item()))

    #训练曲线
    mp.figure()
    mp.plot(train_loss, 'b', label='loss')
    mp.title('Train loss curve')
    mp.xlabel('epoch_num')
    mp.ylabel('train_loss')
    mp.savefig('../resource/train_result/loss.png', format='png', dpi=200)

    #保存模型参数
    torch.save(model.state_dict(), '../resource/train_result/LSTM_param.pkl')
    t1 = time.time()
    T = t1 - t0
    print("total time %.2f" % (T / 60) + 'min')

    # 模型评估
    model = model.eval()
    # 加载参数
    model.load_state_dict(torch.load('../resource/train_result/LSTM_param.pkl'))
    # 全量测试
    test_X = X.reshape(-1, 1, 5)
    test_X = torch.from_numpy(test_X).float().cuda()
    # cuda张量需要先转为cpu pytorch张量再变为numpy
    pred = model(test_X).view(-1).data.cpu().numpy()
    # 补0使长度相同
    pred = np.concatenate((np.zeros(5), pred))

    mp.figure()
    mp.plot(pred, 'orange', label='prediction')
    mp.plot(scaled_close_data, 'aqua', label='real')
    mp.plot((train_size, train_size), (0, 1), 'g--') # 绘制训练集分割线
    mp.legend(loc='best')
    mp.savefig('../resource/train_result/test result.png', format='png', dpi=200)
    mp.close()

    # r2 得分
    test_true = scaled_close_data[int(len(scaled_close_data) * 0.7):-1]
    test_pred = pred[int(len(scaled_close_data) * 0.7):]
    print('r2 score: %.2f' % r2_score(test_true, test_pred))