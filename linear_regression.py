import numpy as np
from matplotlib import pyplot as plt
import random
import copy

# 随机创建一些初始x,y值,n:参数个数(特征数-1)
def loadData(n):

    # 创建随机数据
    theta = [0.0] * n
    x = []
    for i in range(n):
        x.append([1])

    for idx in range(n):
        x[idx] = np.array([random.random()*10-5 for _ in range(100)])
        theta[idx] = random.random()*10-5
    # x0 初始为1
    x[0] = np.array([1.0] * 100)

    # 标准值加一些噪声得到数据y
    std_y = 0
    for i in range(n):
        std_y += theta[i]*x[i]
    noise_y = copy.deepcopy(std_y)
    for y in range(len(std_y)):
        noise_y[y] += random.random()*2-1

    return x,noise_y

def gradient_decent(x,noise_y,plot=False):
    theta = np.random.uniform(size=[len(x)])
    lr = 0.001
    loss_set = []
    epoch_set = []
    for epoch in range(3000):
        pred_y = 0
        for i in range(len(x)):
            pred_y += theta[i] * x[i]
        # 先算出所有变量梯度
        grad = []
        for i in range(len(theta)):
            grad.append(sum((pred_y - noise_y)*x[i]) / 100)
        # 开始梯度下降
        for i in range(len(theta)):
            theta[i] -= grad[i]*lr
        temp_loss = sum((pred_y - noise_y)) / 100
        loss = sum((pred_y-noise_y)**2)/200
        loss_set.append(loss)
        epoch_set.append(epoch)
    if(plot):
        plt.plot(epoch_set, loss_set)
        plt.xlabel('Iteration times')
        plt.ylabel('Loss')
        plt.show()
    print("Gradient descent loss: %.4f" % loss_set[-1])

def normal_equation(x,noise_y):
    # theta = (X.T*X).I*X.T*Y
    # 把样本数据拼接成一个矩阵
    X = np.mat([[x[j][i] for j in range(len(x))] for i in range(100)])
    # 根据正规方程求得参数
    theta = np.matmul(np.matmul(np.matmul(X.transpose(),X).I,X.T),noise_y)
    theta = np.array(theta)[0]
    # 查看loss
    pred_y = 0
    for i in range(len(x)):
        pred_y += theta[i] * x[i]
    loss = sum((pred_y-noise_y)**2)/200
    print("Normal equation loss: %.4f" % loss)

if __name__ == '__main__':
    x,noise_y = loadData(8)
    gradient_decent(x,noise_y,True)
    normal_equation(x,noise_y)