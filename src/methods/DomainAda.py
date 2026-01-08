from typing import Self
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ot.da import *


class CORAL(BaseEstimator, TransformerMixin):
    """
    CORAL: CORrelation ALignment for domain adaptation.
    
    Parameters
    ----------
    regularization : float
        加在协方差矩阵对角线上的正则项，默认 1.0 (对应加 I)

    Attributes
    ----------
    Cs_inv_sqrt_ : ndarray
        源域协方差矩阵的 -1/2 幂
    Ct_sqrt_ : ndarray
        目标域协方差矩阵的 +1/2 幂
    """

    def __init__(self, regularization=1.0):
        super().__init__()
        self.regularization = regularization

    def _matrix_power(self, M, power):
        """使用 SVD 计算矩阵的幂 M^power."""
        U, S, Vt = np.linalg.svd(M)
        S_power = np.diag(S ** power)
        return U @ S_power @ U.T

    def fit(self, X, sample_domain):
        """
        Learn the whitening and coloring matrices from source and target.

        Parameters
        ----------
        Xs : source data, shape (n_samples_s, n_features)
        Xt : target data, shape (n_samples_t, n_features)
        """
        Xs = X[sample_domain == 1]
        Xt = X[sample_domain == -1]
        
        # 协方差矩阵
        Cs = np.cov(Xs, rowvar=False) + self.regularization * np.eye(Xs.shape[1])
        Ct = np.cov(Xt, rowvar=False) + self.regularization * np.eye(Xt.shape[1])

        # 计算 Cs^{-1/2} 和 Ct^{1/2}
        self.Cs_inv_sqrt_ = self._matrix_power(Cs, -0.5)
        self.Ct_sqrt_ = self._matrix_power(Ct, 0.5)

        return self

    def transform(self, Xs):
        """
        Transform source data to target domain space.

        Parameters
        ----------
        Xs : source data, shape (n_samples_s, n_features)

        Returns
        -------
        Xs_trans : transformed source data
        """
        # whiten + color
        X_whiten = Xs @ self.Cs_inv_sqrt_
        X_color = X_whiten @ self.Ct_sqrt_
        return X_color

    def fit_transform(self, X, sample_domain):
        """Fit and then transform the source data."""
        Xs = X[sample_domain == 1]
        Xt = X[sample_domain == -1]
        return self.fit(X, sample_domain).transform(Xs)

class TCA(BaseEstimator):
    pass

## 封装ot.da最优运输方法
# OT原始形式最优运输域适应
class OT_Exact(BaseEstimator):
    
    def __init__(self):
        super().__init__()
        self.emd_transport = EMDTransport()

    def fit(self, X, sample_domain):
        Xs = X[sample_domain == 1]
        Xt = X[sample_domain == -1]
        self.emd_transport.fit(Xs=Xs, Xt=Xt)
        return self
    
    def transform(self, Xs):
        return self.emd_transport.transform(Xs)
    
    def fit_transform(self,X,sample_domain):
        Xs = X[sample_domain == 1]
        Xt = X[sample_domain == -1]
        return self.fit(X, sample_domain).transform(Xs)
# OT信息熵形式最优运输域适应    
class OT_IT(BaseEstimator):

    def __init__(self, reg_alpha=1):
        super().__init__()
        self.sinkhorn_transport = SinkhornTransport(reg_e=reg_alpha)
    
    def fit(self, X, sample_domain):
        Xs = X[sample_domain == 1]
        Xt = X[sample_domain == -1]
        return self.sinkhorn_transport.fit(Xs=Xs, Xt=Xt)
    
    def transform(self, Xs):
        return self.sinkhorn_transport.transform(Xs)
    
    def fit_transform(self, X, sample_domain):
        Xs = X[sample_domain == 1]
        Xt = X[sample_domain == -1]
        return self.fit(X, sample_domain).transform(Xs)

# OT Group Lasso
class OT_GL(BaseEstimator):
    def __init__(self, reg_e=1e-1,reg_cl=1e0):
        super().__init__()
        self.sinkhornLpl1_transport = SinkhornLpl1Transport(reg_e=reg_e, reg_cl=reg_cl)

    def fit(self, X, y, sample_domain):
        Xs = X[sample_domain == 1]
        Xt = X[sample_domain == -1]
        ys = y[sample_domain == 1]
        return self.sinkhornLpl1_transport.fit(Xs=Xs, ys=ys, Xt=Xt)
    
    def transform(self, Xs):
        return self.sinkhornLpl1_transport.transform(Xs)
    
    def fit_transform(self, X, y, sample_domain):
        Xs = X[sample_domain == 1]
        Xt = X[sample_domain == -1]
        return self.fit(X, y, sample_domain).transform(Xs)

# OT Laplace
class OT_Laplace():
    def __init__(self, reg_lap=1, reg_src = 1):
        super().__init__()
        self.emdLaplace_transport = EMDLaplaceTransport(reg_lap=reg_lap, reg_src=reg_src)
    
    def fit(self, X, y, sample_domain):
        Xs = X[sample_domain == 1]
        Xt = X[sample_domain == -1]
        ys = y[sample_domain == 1]
        return self.emdLaplace_transport.fit(Xs=Xs, ys=ys, Xt=Xt)
    
    def transform(self, Xs):
        return self.emdLaplace_transport.transform(Xs)
    
    def fit_transform(self, X, y, sample_domain):
        Xs = X[sample_domain == 1]
        return self.fit(X, y, sample_domain).transform(Xs)

class OT_Unbalance(BaseEstimator):
    def __init__(self,reg_e=1, reg_m=0.1):
        super().__init__()
        self.unbalancedSinkhorn_transport = UnbalancedSinkhornTransport(reg_e=reg_e, reg_m=reg_m)

    def fit(self, X, sample_domain):
        Xs = X[sample_domain == 1]
        Xt = X[sample_domain == -1]
        return self.unbalancedSinkhorn_transport.fit(Xs)

    def transform(self, Xs):
        return self.unbalancedSinkhorn_transport.transform(Xs)
        
    def fit_transform(self, X, sample_domain):
        Xs = X[sample_domain == 1]
        return self.unbalancedSinkhorn_transport.fit(X, sample_domain).transform(Xs)
