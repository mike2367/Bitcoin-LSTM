import torch
from torch import nn
import torch.optim as optim
from ray import tune
from lstm_model import train_x, train_y
from lstm_model import LSTM_regression
from sklearn.metrics import r2_score
from lstm_model import X, scaled_close_data, STEP_LENGTH
import time
import numpy as np


# 1次epoch训练
def train(optimizer, model, data, loss_function):
    for i in range(200):
        train_x_, train_y_ = data
        out = model(train_x_)
        loss = loss_function(out, train_y_)
        loss.backward()
        # optimizer更新参数
        optimizer.step()
        optimizer.zero_grad()

def test(model, X):
    model = model.eval()
    # 全量测试
    test_X = X.reshape(-1, 1, STEP_LENGTH)
    test_X = torch.from_numpy(test_X).float().cuda()
    # cuda张量需要先转为cpu pytorch张量再变为numpy
    pred = model(test_X).view(-1).data.cpu().numpy()
    # 补0使长度相同
    pred = np.concatenate((np.zeros(STEP_LENGTH), pred))
    test_true = scaled_close_data[int(len(scaled_close_data) * 0.7):-1]
    test_pred = pred[int(len(scaled_close_data) * 0.7):]

    return r2_score(test_true, test_pred)

def train_process(config):

    model = LSTM_regression(STEP_LENGTH, 8, output_size=1, num_layers=2)
    model = model.cuda()

    data = (train_x.cuda(), train_y.cuda())
    loss_function = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=config['lr'], betas=config['betas'],
                           eps=1e-08, weight_decay=config['weight_decay'])


    train(optimizer, model, data, loss_function)
    acc = test(model, X)
    return {"score": float(acc)}


config = {
    'lr':tune.grid_search([0.05, 0.01, 0.001]),
    'betas':tune.grid_search([(0.9, 0.999), (0.9, 0.99)]),
    'weight_decay': tune.grid_search([1e-4, 1e-5, 0])
}

t0 = time.time()
tuner = tune.Tuner(train_process, param_space=config)
res = tuner.fit()
t1 = time.time()
T = t1 - t0
print("total time %.2f" % (T / 60) + 'min')
print(res.get_best_result(metric="score", mode="max").config)
