import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


def PCA_plot(X,y,sample_domain,component_order=1,label_names=[0,1]):

    src_idx = np.where(sample_domain == 1)
    tar_idx = np.where(sample_domain == 0)
    Xs = X[src_idx]
    Xt = X[tar_idx]
    y_ = (y - label_names[0]) /(label_names[1] - label_names[0])

    ys = y_[src_idx]
    yt = y_[tar_idx]

    cls = ['g','y','grey','grey']
    labels_s = [cls[int(i)] for i in ys]
    labels_t = [cls[int(i)+2] for i in yt]
    n_points1 = np.sum(y_==0)
    n_points2 = np.sum(y_==1)

    pca = PCA(svd_solver='full',n_components=2)
    pca.fit(X)
    Us = pca.transform(Xs)
    Ut = pca.transform(Xt)
    u = np.concatenate((Us,Ut),axis=0)
    
    nbins = 30

    plt.subplots(1,2,figsize=(10,6))
    plt.subplot(121)
    plt.scatter(Us[:,0],Us[:,1],c=labels_s,marker='*', label='Source', alpha=0.3,s=10)
    plt.scatter(Ut[:,0],Ut[:,1],c=labels_t,marker='d', label='Target', alpha=0.3,s=10)
    plt.legend()
    plt.xlabel('PC1',fontsize=20)
    plt.ylabel('PC2',fontsize=20)
    plt.axis('equal')

    plt.subplot(122)
    rng = (np.min(np.min(u[:,component_order-1])),np.max(np.max(u[:,component_order-1])))
    plt.hist(Us[0:n_points1,component_order-1],bins=nbins,color='g',alpha=0.3,range=rng)
    plt.hist(Us[n_points1:n_points1+n_points2,component_order-1],bins=nbins,color='y',alpha=0.3,range=rng)
    plt.hist(Ut[:,component_order-1],bins=nbins,color='grey',alpha=0.3,range=rng)
    plt.title(f'PC{component_order}',fontsize=20)
    plt.show()

if __name__ ==  '__main__':
    Xs = np.concatenate((np.random.normal(2,10,(50,1000)) , np.random.normal(-2,10,(50,1000))),axis=0)
    Xt = (Xs[np.r_[0:40,50:90]] **2)/50
    y = np.array([0]*50 + [1]*50 + [2]*40 + [3]*40)
    X = np.concatenate((Xs,Xt),axis=0)
    sample_domain = np.array([1]*100+[0]*80)

    PCA_plot(X,y,sample_domain)