from matplotlib import pyplot as plt
import numpy as np
import random
import math

def createData(m,n):
    x = np.random.random((m,n))
    x = x*20-10
    x[0] = np.array([1.]*n)
    while True:
        theta = np.random.random(n)
        theta = theta*10-5
        y = np.matmul(x,theta)
        class_y = np.array([1 if x+random.random()*10-5>0 else 0 for x in y])
        true_num = sum([x>0 for x in y])
        print(true_num,len(y))
        if 0.45<true_num/len(y)<0.55:
            break
    return x,class_y

def gradient_decent(x,y):
    theta = np.random.random(len(x[0]))
    theta = theta * 10 - 5
    lr = 0.0005
    # 计算梯度
    epoch_set = []
    loss_set = []
    for epoch in range(100):
        pred_y = np.matmul(x, theta)
        pred_y = np.array([1 / (1 + math.exp(-i)) for i in pred_y])
        grad = [0.]*len(theta)
        for j in range(len(theta)):
            for i in range(len(y)):
                grad[j] += (pred_y[i]-y[i])*x[i][j]
        for i in range(len(theta)):
            theta[i] -= lr*grad[i]
        #print(pred_y)
        # pred_y 中的值为概率
        class_y = np.array([1 if i>0.5 else 0 for i in pred_y])
        loss = sum([class_y[i]==y[i] for i in range(len(y))])
        #print(loss)
        loss_set.append(1-loss/len(x))
        epoch_set.append(epoch)
    plt.plot(epoch_set,loss_set)
    plt.show()


if __name__=='__main__':
    x,y = createData(1000,10)
    gradient_decent(x,y)
