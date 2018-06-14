import numpy as np
from matplotlib import pyplot as plt

def load_data(size, dim):
    # generate x
    x = np.random.randn(size, dim+1)*10
    for i in range(size):
        x[i,0] = 1.
    x = np.mat(x)

    # generate line
    true_num = 0
    while(not 0.4<true_num/size<0.6):
        theta = np.random.randn(dim+1)*5
        y = np.matmul(x, theta)
        y = np.array(y)[0]
        y = [1 if yi>0 else 0 for yi in y]
        true_num = sum(y)
        print(true_num)

    x = np.array(x)
    if(dim==2):
        true_sample = x[[True if yi==1 else False for yi in y]]
        plt.scatter(true_sample[:,1].reshape(-1),true_sample[:,2].reshape(-1),c='r',marker='x')
        false_sample = x[[True if yi==0 else False for yi in y]]
        print(len(false_sample))
        plt.scatter(false_sample[:,1].reshape(-1), false_sample[:,2].reshape(-1), c='b', marker='x')

        x0 = np.array([1.]*size)
        x1 = np.random.randn(size)*100
        x2 = -(x0*theta[0]+x1*theta[1])/theta[2]
        plt.ylim([-30,30])
        plt.xlim([-30, 30])
        plt.plot(x1,x2,c='g')
        plt.show()

    return x,y

#def svm(x, y):


if __name__ == "__main__":
    load_data(100,2)
