import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


def DomainPlot(X, y ,sample_domain):
    src_idx = np.where(sample_domain == 1)
    tar_idx = np.where(sample_domain == -1)
    Xs = X[src_idx]
    Xt = X[tar_idx]
    ys = y[src_idx]
    yt = y[tar_idx]

    plt.subplots(1,2,figsize=(10,6))
    plt.subplot(121)
    plt.scatter(Xs[:, 0], Xs[:, 1], c=ys, marker="+", label="Source samples", alpha=0.3, s=10)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc=0)
    plt.title("Source  samples")
    
    plt.subplot(122)
    plt.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker="o", label="Target samples", alpha=0.3, s=10)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc=0)
    plt.title("Target samples")
    plt.tight_layout()

def PCAPlot(X, y ,sample_domain):

    src_idx = np.where(sample_domain == 1)
    tar_idx = np.where(sample_domain == -1)
    Xs = X[src_idx]
    Xt = X[tar_idx]
    ys = y[src_idx]
    yt = y[tar_idx]


    pca = PCA(svd_solver='full',n_components=2)
    pca.fit(X)
    Us = pca.transform(Xs)
    Ut = pca.transform(Xt)

    plt.subplots(1,2,figsize=(10,6))
    plt.subplot(121)
    plt.scatter(Us[:,0], Us[:,1], c=ys, cmap='tab10', marker='+', label='Source', alpha=0.3, s=10)
    plt.legend()
    plt.xlabel('PC1',fontsize=20)
    plt.ylabel('PC2',fontsize=20)
    plt.axis('equal')
    plt.title("Source  samples")
    
    plt.subplot(122)
    plt.scatter(Ut[:,0], Ut[:,1], c=yt, cmap='tab10', marker='o', label='Target', alpha=0.3, s=10)
    plt.legend()
    plt.xlabel('PC1',fontsize=20)
    plt.ylabel('PC2',fontsize=20)
    plt.axis('equal')
    plt.title("Target  samples")
    plt.tight_layout()

def CurvePlot(x, y1, y2, label_list):
    plt.figure(figsize=(8,5))
    plt.plot(x, y1, label=label_list[0])
    if y2 != None:
        plt.plot(x, y2 , 'm--', label=label_list[1])
    plt.grid()
    plt.legend()
    plt.show()

def GradientDescentPlot(x, x_, f):
    plt.figure(figsize=(8,5))
    plt.plot(x, f(x), label='f(x)')
    plt.plot(x_, f(x_), 'm--', marker='o', markersize=4, label = 'gradient descent')
    plt.grid()
    plt.legend()
    plt.show()

def GradientPlot_2d(x1_range, x2_range, f, results):
    
    plt.figure(figsize=(8,6))
    x1, x2 = np.meshgrid(np.arange(x1_range[0], x1_range[1], 0.1), 
                            np.arange(x2_range[0], x2_range[1], 0.1), indexing='ij' )
    plt.contour(x1, x2, f(x1, x2), colors='green')
    plt.plot(*zip(*results), '-o', color='orange') # 内层*号将列表打开，作为输入参数，zip函数按列重组，外层*号打开为两列输入
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def LossAndAccPlot(epochs, train_loss, test_loss, train_acc, test_acc):
    epoch_list = [i+1 for i in range(epochs)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epoch_list, train_acc, 'red', label='train acc')
    axes[0].plot(epoch_list, test_acc, 'blue', label='test acc')
    axes[0].set_yticks([i/10 for i in range(11)])
    axes[0].set_yticklabels([i/10 for i in range(11)])
    axes[0].legend()
    axes[0].set_ylabel('Accuracy')

    axes[1].plot(epoch_list, train_loss, 'orange', label='train loss')
    axes[1].plot(epoch_list, test_loss, 'skyblue', label='test loss')
    axes[1].legend()
    axes[1].set_ylabel('Loss')
    plt.tight_layout()
    plt.show()
    return 


if __name__ ==  '__main__':
    Xs = np.concatenate((np.random.normal(2,10,(50,1000)) , np.random.normal(-2,10,(50,1000))),axis=0)
    Xt = (Xs[np.r_[0:40,50:90]] **2)/50
    y = np.array([0]*50 + [1]*50 + [0]*40 + [1]*40)
    X = np.concatenate((Xs,Xt),axis=0)
    sample_domain = np.array([1]*100+[-1]*80)

    PCAPlot(X,y,sample_domain)