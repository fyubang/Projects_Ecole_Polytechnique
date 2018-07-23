from __future__ import division
import numpy as np
from scipy import signal
from scipy.special import gamma, digamma
from sklearn.neighbors import NearestNeighbors
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib.parallel import cpu_count
import bottleneck as bn

NUM_CPU = cpu_count()


def _get_first_mutual_info_unwrap(*arg, **kwarg):
    """
        Parallelize the get_first_mutual_info function
    """
    return FetureSelection_mRmR._get_first_mutual_info(*arg, **kwarg)


def _get_mutual_info_unwrap(*arg, **kwarg):
    """
        Parallelize the get_mutual_info function
    """

    return FetureSelection_mRmR._get_mutual_info(*arg, **kwarg)


class FetureSelection_mRmR(object):
    """
        mRmR represents minimal-redundancy-maximal-relevance, which is an important way to select the most important
    n_features. The method is realized by the mutual information.

    Parameters
    ----------

    n_neighbours: int, optional default 6.
        used to set the number of neighbour which would be applied to the kernel density estimation.

    n_features: int or string 'auto', optional, default 'auto'
        1. int, the number of features that the user would like to select;
        2. string, 'auto' represents the best num of features determined
            automatically by the algorithm.

    classfication: Boolean, optional, default Ture
        1. True for the classification problem;
        2. False for the regression problem.

    n_jobs: int, optional, default 1
        the number of cpus that user wants to apply.

    verbose: int, optional, default 0

    Attributes
    ----------
    n_features_: int, the number of the final selected features


    """
    def __init__(self, n_neighbours=6, n_features='auto', classfication=True, n_jobs=1, verbose=0):
        self.n_neighbours = n_neighbours
        self.n_features = n_features
        self.classification = classfication
        self.n_jobs = n_jobs if n_jobs > 0 else NUM_CPU
        self.verbose = verbose
        self.X = None
        self.y = None
        self.n_features_ = None
        self.ranking_ = None
        self.mi_ = None

    @classmethod
    def _nearest_distances(cls, X, k=1):
        """
            Calculate the distance of the kth nearest point of each point in X

        :param X: 2-D array metrix
        :param k: int, kth neighbor
        :return: 1-D array of the distances of the kth nearest point of each point in X
        """
        knn = NearestNeighbors(n_neighbors=k, metric='chebyshev')
        knn.fit(X)
        return knn.kneighbors(X)[0][:, -1]

    def _entropy(self, X, k=1):
        """
            Calculate the entropy of X

        :param X: 2-D array metrix
        :param k: int, kth neighbor
        :return: entropy of the

        Reference:
        ---------
        Geometric k-nearest neighbor estimation of entropy and mutual information
        Warren M. Lord, Jie Sun, and Erik M. Bollt
        """
        # it works well too for enough data set
        # return np.log((np.pi ** (d/2)) / gamma(d/2 + 1)) + d * np.mean(np.log(distances)) + \
        #         np.log(n) - np.log(k)
        distances = self._nearest_distances(X, k)
        n, d = X.shape
        tmp = (np.pi ** (.5 * d)) / gamma(.5 * d + 1)
        return d * np.mean(np.log(distances + np.finfo(self.X.dtype).eps)) + np.log(tmp) + digamma(n) - digamma(k)

    def _mutual_info_cc(self, variables, k=1):
        """
            Calculate the mutual information of the 1-D arrays

        :param variables: the tuple which contains the 1-D arrays(n*1)
        :param k: the number of neighbours
        :return:
        """
        if len(variables) < 2:
            raise AttributeError(
                "Mutual information must involve at least 2 variables")
        all_vars = np.hstack(variables)
        return sum([self._entropy(X, k) for X in variables]) - \
               self._entropy(all_vars, k)

    def _mutual_info_dc(self, x, y, k):
        """
            Calculates the mututal information between a continuous vector x and a
        disrete class vector y.

        :param x: 2-D continuous array (n*1)
        :param y: 2-D discrete array (n*1)
        :param k: the number of the neighbours
        :return: the mutual information between x and y

        Reference:
        ---------
        Brian C. Ross, 2014, PLOS ONE
        Mutual Information between Discrete and Continuous Data Sets
        """
        y = y.flatten()
        n = x.shape[0]
        classes = np.unique(y)
        knn = NearestNeighbors(n_neighbors=k)
        distances_kth = np.zeros(n)
        Nx = [np.count_nonzero(y == yi) for yi in y]
        for c in classes:
            mask = np.where(y == c)[0]
            knn.fit(x[mask])
            distances_kth[mask] = knn.kneighbors()[0][:, -1]

        knn.fit(x)
        m = knn.radius_neighbors(radius=distances_kth, return_distance=False)
        m = [i.shape[0] for i in m]
        return digamma(n) - np.mean(digamma(Nx)) + digamma(k) - np.mean(digamma(m))

    def _get_mutual_info(self, f, s):
        """
            Calculate the mutual information between f-th and s-th columnof Matrix X(continuous)
        :param f: the column number of matrix X
        :param s: the column number of matrix X
        """
        n, p = self.X.shape
        variables = (self.X[:, s].reshape(n, 1), self.X[:, f].reshape(n, 1))
        MI = self._mutual_info_cc(variables, k=self.n_neighbours)
        return MI if MI > 0 else np.nan

    def _get_mi_vector(self, F, s):
        """
            Calculate the vector of mutual informations between all the unselected features and the feature s.
        :param F: The array of indexes of columns of the matrix X which have not been seleted
        :param s: The index of the last selected column of X
        :return:
        """
        MIs = Parallel(n_jobs=self.n_jobs)(delayed(_get_mutual_info_unwrap)(self, f=f, s=s) for f in F)
        return MIs

    def _get_first_mutual_info(self, i, k=1):
        """
            Calculate the the mutual information between the i-th column of X and y.
        :param i: the index of i-th column of X
        """
        n, p = self.X.shape
        if self.classification:
            x = self.X[:, i].reshape((n, 1))
            MI = self._mutual_info_dc(x, self.y, k)
        else:
            # variables is 2-col 2-D array
            variables = (self.X[:, i].reshape((n, 1)), self.y)
            MI = self._mutual_info_cc(variables, k)
        return MI if MI > 0 else np.nan

    def _get_first_mi_vector(self, k=1):
        """
            Calculates the Mututal Information between each feature in X and y,
            Used when selecting the first feature.
        """
        n, p = self.X.shape
        MIs = Parallel(n_jobs=self.n_jobs)(delayed(_get_first_mutual_info_unwrap)(self, i=i, k=k) for i in range(p))
        return MIs

    def _check_params(self, X, y):
        """
            Check if the input X, y satisfy the requirements of the model.
            For the case that y is continuous, scale X and y.
        :param X: the 2-D array feature matrix X
        :param y: the 1-D label array y
        """
        X, y = check_X_y(X, y)
        if not self.classification:
            scale = StandardScaler()
            X = scale.fit_transform(X)
            y = scale.fit_transform(y.reshape(-1, 1))
        if self.n_jobs > NUM_CPU:
            raise ValueError("n_jobs must be smaller the number of cpu")
        if not isinstance(self.n_neighbours, int):
            raise ValueError("n_neighbours must be an integer.")
        if self.n_neighbours < 1:
            raise ValueError('n_neighbours must be larger than 0.')
        if self.classification and self.n_neighbours > np.min(np.bincount(y)):
            raise ValueError('n_neighbours must be smaller than the occurence of each label.')
        if self.verbose != 1 and self.verbose != 2 and self.verbose != 2:
            raise ValueError('verbose must be 1, 2 or 3.')
        return X, y

    @classmethod
    def _add_remove(cls, S, F, i):
        """
            Move the selected feature from F to S
        :param S: the list of index of selected features
        :param F: the list of index of unselected features
        :param i: index of feature that will be moved
        :return: S, F
        """
        S.append(i)
        F.remove(i)
        return S, F

    def fit(self, X, y):
        X_y = self._check_params(X, y)
        self.X = X_y[0]
        self.y = X_y[1].reshape((-1, 1))
        n, p = X.shape

        S = []    # list of selected features
        F = range(p)    # list of unselected features

        if self.n_features != 'auto':
            feature_mi_matrix = np.zeros((self.n_features, p))
        else:
            feature_mi_matrix = np.zeros((n, p))
        feature_mi_matrix[:] = np.nan
        S_mi = []

        # Find the first feature
        k_min = 3
        range_k = 7
        xy_MI = np.empty((range_k, p))
        for i in range(range_k):
            xy_MI[i, :] = self._get_first_mi_vector(i + k_min)
        xy_MI = bn.nanmedian(xy_MI, axis=0)

        S, F = self._add_remove(S, F, bn.nanargmax(xy_MI))
        S_mi.append(bn.nanmax(xy_MI))

        if self.verbose > 0:
            self._info_print(S, S_mi)

        # Find the next features
        if self.n_features == 'auto':
            n_features = np.inf
        else:
            n_features = self.n_features

        while len(S) < n_features:
            s = len(S) - 1
            feature_mi_matrix[s, F] = self._get_mi_vector(F, S[-1])
            fmm = feature_mi_matrix[:len(S), F]
            if bn.allnan(bn.nanmean(fmm, axis=0)):
                break
            MRMR = xy_MI[F] - bn.nanmean(fmm, axis=0)
            if np.isnan(MRMR).all():
                break
            selected = F[bn.nanargmax(MRMR)]
            S_mi.append(bn.nanmax(bn.nanmin(fmm, axis=0)))
            S, F = self._add_remove(S, F, selected)
            if self.verbose > 0:
                self._info_print(S, S_mi)
            if self.n_features == 'auto' and len(S) > 10:
                MI_dd = signal.savgol_filter(S_mi[1:], 9, 2, 1)
                if np.abs(np.mean(MI_dd[-5:])) < 1e-3:
                    break
        self.n_features_ = len(S)
        self.ranking_ = S
        self.mi_ = S_mi

        return self

    def transform(self, X):
        try:
            self.ranking_
        except AttributeError:
            raise ValueError('You need to call the fit(X, y) method first.')
        X = X[:, self.ranking_]
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _info_print(self, S, MIs):
        if self.n_features == 'auto':
            text = 'Auto model---Selected feature # %d: %d' % (len(S), S[-1])
        else:
            text = 'Manual model---Selected feature # %d/%d: %d' %(len(S), self.n_features,  S[-1])
        if self.verbose == 2:
            text += ', Mutual Information: %.4f' % MIs[-1]
        print (text)

    def getSelectedFeatures(self):
        try:
            self.ranking_
        except AttributeError:
            raise ValueError('You need to call the fit(X, y) method first.')
        return self.ranking_

