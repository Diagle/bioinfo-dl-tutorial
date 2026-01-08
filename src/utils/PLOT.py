import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


def Domain_plot(X, y ,sample_domain):
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

def PCA_plot(X, y ,sample_domain):

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

def curve_plot(x, y1, y2, label_list):
    plt.figure(figsize=(8,5))
    plt.plot(x, y1, label=label_list[0])
    plt.plot(x, y2 , 'm--', label=label_list[1])
    plt.grid()
    plt.legend()
    plt.show()


if __name__ ==  '__main__':
    Xs = np.concatenate((np.random.normal(2,10,(50,1000)) , np.random.normal(-2,10,(50,1000))),axis=0)
    Xt = (Xs[np.r_[0:40,50:90]] **2)/50
    y = np.array([0]*50 + [1]*50 + [0]*40 + [1]*40)
    X = np.concatenate((Xs,Xt),axis=0)
    sample_domain = np.array([1]*100+[-1]*80)

    PCA_plot(X,y,sample_domain)