import numpy as np
from matplotlib import pyplot as plt

def generate_data():
    data_set = []
    for epoch in range(10):
        data = np.random.randn(50,2)*10-np.array([np.random.randint(-100,100),np.random.randint(-100,100)])
        plt.scatter(data[:,0].reshape(-1), data[:,1].reshape(-1))
        data_set.append(data)
    return np.array(data_set).reshape(-1,2)

def EM():
    # init
    data = generate_data()
    category = 6
    miu = np.array([[np.random.randint(-100,100),np.random.randint(-100,100)] for _ in range(category)])
    eps = [np.matrix([[np.random.randint(1,5),0],[0,np.random.randint(1,5)]]) for _ in range(category)]
    fi = np.array([1/category]*category)
    omega = [[j for j in range(len(data))]for _ in range(category) ]

    for epoch in range(10):
        # E-step
        for j in range(len(data)):
            p = []
            for i in range(category):
                #print(np.matrix(data[j]),np.matrix(miu[i]))
                p.append([1/((2*np.pi)**(category/2)*np.linalg.det(eps[i]))*(np.e**-
                np.matmul(np.matmul((np.matrix(data[j])-np.matrix(miu[i])),eps[i].I),
                          (np.matrix(data[j])-np.matrix(miu[i])).T).item(0)),fi[i]])
            temp = sum(map(lambda x:x[0]*x[1],p))
            if temp==0.:
                for i in range(category):
                    omega[i][j] = 1/category
            else:
                for i in range(category):
                    omega[i][j] = p[i][0]*p[i][1]/temp
        # M-step
        omega = np.array(omega)
        fi = [sum(omega[i])/len(data) for i in range(category)]
        miu = [sum(omega[j][i]*data[i] for i in range(len(data)))/fi[j]/len(data) for j in range(category)]
        eps = [sum(np.matmul(np.matrix(data[i]-miu[j]).T,np.matrix(data[i]-miu[j]))*omega[j][i] for i in
            range(len(data)))/sum(omega[j]) for j in range(category)]

    # plot
    for i in range(category):
        data = np.random.multivariate_normal(miu[i],eps[i],[150])
        plt.plot(data[:,0].reshape(-1),data[:,1].reshape(-1))
    plt.show()

if __name__ == "__main__":
    EM()








