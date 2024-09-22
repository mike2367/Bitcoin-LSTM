import matplotlib.pyplot as mp
import numpy as np
from lstm_model import train_x, train_y, train_size
from lstm_model import LSTM_regression
from lstm_model import X,scaled_close_data, STEP_LENGTH
import time
from sklearn.metrics import r2_score
mp.rcParams['font.family'] = 'SimHei'
mp.rcParams['axes.unicode_minus'] = False

# LSTM 预测模型
import torch
from torch import nn
# 安排GPU
device = torch.cuda.device('cuda')



if __name__ == '__main__':
    t0 = time.time()
    model = LSTM_regression(STEP_LENGTH, 8, output_size=1, num_layers=2)
    model=model.cuda()
    # 计算总参数量
    model_total = sum([param.nelement() for param in model.parameters()])

    print("number of total parameters: %.8fM" % (model_total / 1e6))

    train_loss = []
    # mean square error
    loss_function = nn.MSELoss()
    loss_function = loss_function.cuda()
    # betas 为Adam算法参数，weight_decay防止过拟合，一般取1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, betas=(0.9, 0.99),
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
    test_X = X.reshape(-1, 1, STEP_LENGTH)
    test_X = torch.from_numpy(test_X).float().cuda()
    # cuda张量需要先转为cpu pytorch张量再变为numpy
    pred = model(test_X).view(-1).data.cpu().numpy()
    # 补0使长度相同
    pred = np.concatenate((np.zeros(STEP_LENGTH), pred))

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