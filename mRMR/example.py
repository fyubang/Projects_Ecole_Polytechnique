from mRmR import FetureSelection_mRmR
from sklearn.datasets import make_classification, make_regression

if __name__ == '__main__':
    s = 200
    f = 20
    i = int(0.1 * f)
    r = int(0.05 * f)
    c = 2

    # generate data with discrete labels
    X, y = make_classification(n_samples=s, n_features=f, n_informative=i,
                               n_redundant=r, n_clusters_per_class=c,
                               random_state=18, shuffle=False)
    mRmR = FetureSelection_mRmR(n_jobs=-1, verbose=2)
    mRmR.fit(X, y)
    print (mRmR.getSelectedFeatures())

    # generate data with continuous labels
    X, y = make_regression(n_samples=s, n_features=f, n_informative=i,
                           random_state=18, shuffle=False)
    mRmR = FetureSelection_mRmR(classfication=False, n_jobs=-1, verbose=2)
    mRmR.fit(X, y)
    print (mRmR.getSelectedFeatures())