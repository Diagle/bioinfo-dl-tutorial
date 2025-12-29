import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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
        Xt = X[sample_domain == 0]
        
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
        Xt = X[sample_domain == 0]
        return self.fit(X, sample_domain).transform(Xs)

class TCA(BaseEstimator):
    pass

class OT(BaseEstimator):
    
    def __init__(self):
        super.__init__()

    def fit(self):
        pass
    
    def transform(self):
        pass
    
    def fit_transform(self):
        return self.fit().transform()