import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from joblib import Parallel, delayed
from itertools import product, izip, combinations

'''

train_data = pd.read_csv('train_v2.csv')
test_data = pd.read_csv('test_v2.csv')

features = list(test_data.keys())
features.remove('id')
X = np.vstack([np.asarray(np.hstack([train_data[feat].values, test_data[feat].values]), dtype=np.float64) for feat in features]).T
X[np.isinf(X)] = np.nan

loss = train_data['loss'].values
default = np.asarray(train_data['loss'].values > 0, dtype=int)

#imputer = Imputer()
imputer = Imputer()
X = imputer.fit_transform(X)

scalar = StandardScaler()
X = scalar.fit_transform(X)
'''

X = np.load('X2.npy')
default = np.load('default.npy')
loss = np.load('loss.npy')

n_folds = 5
kf = KFold(len(loss), n_folds=n_folds, shuffle=True, random_state=1)

#features = [520, 521, 268, 271]
#features = [520, 521, 268, 1, 331]
#features = [520, 521, 268, 1, 331, 768, 658, 269, 4, 329, 508]
features = [1, 767, 219, 65, 769, 770]


new_features = []
new_features.append(X[:,520][:,np.newaxis] - X[:,521][:,np.newaxis])
new_features.append(X[:,521][:,np.newaxis] - X[:,271][:,np.newaxis])
X = np.hstack([X] + new_features)


def lr_cv(train, val, candidate, weight, thres, C=1e20, penalty='l2'):
    #clf = GradientBoostingClassifier(n_estimators=65, max_depth=6, learning_rate=0.300, max_features='sqrt', verbose=2)
    clf = LogisticRegression(C=C, penalty=penalty, class_weight={0: 1, 1: weight})
    clf.fit(X[train][:,candidate], default[train])
    default_hat = clf.predict_proba(X[val][:,candidate])[:,1]
    return f1_score(default[val], default_hat > thres)

def make_submission():
    print 'here'    
    thres = 0.5
    train = range(len(default))
    test = range(len(default), len(X))
    clf = LogisticRegression(C=1e5, penalty='l2', class_weight={0: 1, 1: 7})
    clf.fit(X[train][:,features], default[train])
    default_hat = clf.predict_proba(X[test][:,features])[:,1]
    print 'here'
    clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, loss='lad')
    clf.fit(X[train][loss[train] > 0], loss[train][loss[train] > 0])
    print 'here'

    loss_hat = np.zeros(default_hat.shape)
    loss_hat[default_hat > thres] = np.clip(np.floor(clf.predict(X[test][default_hat > thres])), 0, 100)
    print 'writing submission...'
    with open('submission.csv', 'w') as fd:
        fd.write('id,loss\n')
        for id, l in zip(range(len(default), len(X)), loss_hat):
            fd.write(str(id+1) + ',' + str(int(l)) + '\n')
    
#make_submission()

            #*********** 0.56563399178 (100, 0.05, 4) ************
    


scores = Parallel(n_jobs=5)(delayed(lr_cv)(train, val, features, 7, 0.5) for train, val in kf)
print np.mean(scores)
assert False
'''
n_folds = 5
kf = KFold(len(loss), n_folds=n_folds, shuffle=True, random_state=1)
for feat in range(20):
    best = -1.0
    best_idx = None
    best_thres = None
    for idx in range(768, X.shape[1]):
        if idx not in features:
            candidate = features + [idx]
            scores = Parallel(n_jobs=5)(delayed(lr_cv)(train, val, candidate, 7, 0.5) for train, val in kf)
            if np.mean(scores) > best:
                best = np.mean(scores)
                best_idx = idx
                print best, best_idx
    features.append(best_idx)
    print best, features
'''

def gbr_cv(train, val, default_hat, n_estimators, learning_rate, max_depth, thres=0.5):
    clf = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, loss='lad')
    clf.fit(X[train][loss[train] > 0], loss[train][loss[train] > 0])

    loss_hat = np.zeros(default_hat.shape)
    loss_hat[default_hat > thres] = np.clip(np.floor(clf.predict(X[val][default_hat > thres])), 0, 100)
    return np.mean(np.abs(loss_hat - loss[val]))
    
#*********** 0.56563399178 (100, 0.05, 4) ************
n_estimators_list = [100]
#n_estimators_list = [10]
learning_rate_list = [0.05]
max_depth_list = [4]
#max_depth_list = [2, 3, 4]

kf = KFold(len(loss), n_folds=n_folds, shuffle=True, random_state=1234)

default_hat_list = []
for train, val in kf:
    clf = LogisticRegression(C=1e5, penalty='l2', class_weight={0: 1, 1: 7})
    clf.fit(X[train][:,features], default[train])
    default_hat_list.append(clf.predict_proba(X[val][:,features])[:,1])

best = 1e3
best_params = None    
for n_estimator, learning_rate, max_depth in product(n_estimators_list, learning_rate_list, max_depth_list):
    scores = Parallel(n_jobs=n_folds)(delayed(gbr_cv)(train, val, default_hat, n_estimator, learning_rate, max_depth) for (train, val), default_hat in izip(kf, default_hat_list))
    print (n_estimator, learning_rate, max_depth), np.mean(scores)
    if np.mean(scores) < best:
        best = np.mean(scores)
        best_params = (n_estimator, learning_rate, max_depth)
        print '***********', best, best_params, '************'

# XXXX can pre-compute logistic regression results for speedup
assert False

scores = []
for train, val in kf:
    clf = LogisticRegression(C=1e20, penalty='l2', class_weight={0: 1, 1: 7})
    clf.fit(X[train][:,features], default[train])
    default_hat = clf.predict_proba(X[val][:,candidate])[:,1]
    clf = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, loss='lad')
    clf.fit(X[train][loss[train] > 0], loss[train][loss[train] > 0])

    loss_hat = np.zeros(default_hat.shape)
    loss_hat[default_hat > 0.5] = np.clip(np.round(clf.predict(X[val][default_hat > 0.5])), 0, 100)

    scores.append(np.mean(np.abs(loss_hat - loss[val])))
    print scores[-1]
    #scores.append(f1_score(default_hat, default[val]))
    #scores.append(roc_auc_score(default[val], default_hat))
    
print np.mean(scores)


def predict(X_train, default_train, loss_train, X_test, features,
            hyperparams={'C': 1e20, 'penalty': 'l2', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.9,
                         'loss': 'lad', 'thres': 0.5},
            default_features=['f527', 'f528', 'f271'],
            loss_features=['f2','f67','f514','f670','f598','f766','f404','f596','f271','f75','f282']):
    features = list(features)
    default_idx = [features.index(f) for f in default_features]
    
    clf = LogisticRegression(C=hyperparams['C'], penalty=hyperparams['penalty'])
    clf.fit(X_train[:, default_idx], default_train)
    default_hat = clf.predict_proba(X_test[:, default_idx])[:,1]
    print roc_auc_score(default_val, default_hat)
    default_mask = default_hat > hyperparams['thres']
    print f1_score(default_val, default_mask)
    
    loss_idx = [features.index(f) for f in loss_features]
    clf = GradientBoostingRegressor(n_estimators=hyperparams['n_estimators'], learning_rate=hyperparams['learning_rate'],
                                    max_depth=hyperparams['max_depth'], subsample=hyperparams['subsample'], loss=hyperparams['loss'])
    clf.fit(X_train[:, loss_idx][loss_train > 0], loss_train[loss_train > 0])
    
    loss_hat = np.zeros(X_test.shape[0])
    loss_hat[default_mask] = np.clip(clf.predict(X_val[:,loss_idx][default_mask]), 0., 100.)
    
    return loss_hat

mae = []
for train, val in kf:
    X_train, X_val = X[train].copy(), X[val].copy()
    default_train, default_val = default[train].copy(), default[val].copy()
    loss_train, loss_val = loss[train].copy(), loss[val].copy()

    hyperparams={'C': 1e20, 'penalty': 'l2', 'n_estimators': 10, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.9, 'loss': 'lad', 'thres': 0.1}
    
    loss_hat = predict(X_train=X_train, default_train=default_train, loss_train=loss_train, hyperparams=hyperparams,
                       X_test=X_val, features=features, loss_features=features)
    mae.append(np.mean(np.abs(loss_hat - loss_val)))
    print mae[-1]
    
print np.mean(mae)

    
class L1Regressor:
    def __init__(self, alpha=0.0):
        self.coef_ = None
        self.alpha = alpha
        assert alpha == 0.0
    
    def fit(self, X, y):
        c = np.array([self.alpha]*X.shape[1] + [1.]*X.shape[0] + [1.]*X.shape[0])
            
        A = np.hstack([X, np.eye(X.shape[0], X.shape[0]), -np.eye(X.shape[0], X.shape[0])])
        b = y[:,np.newaxis]
        
        G = -np.eye(c.shape[0], c.shape[0])
        
        h = -np.array([-10]*X.shape[1] + [0.]*X.shape[0] + [0.]*X.shape[0])
        self.coef_ = cvxopt.solvers.lp(c=matrix(c, tc='d'), A=matrix(A, tc='d'), b=matrix(b, tc='d'), G=matrix(G, tc='d'), h=matrix(h, tc='d'))['x']
            
    def predict(self, X):
        assert self.coef_
            
        return np.dot(X,self.coef_[:X.shape[1]])

kf = KFold(len(loss[loss > 0]), n_folds=10, shuffle=True, random_state=0)

features = list(train_data.keys())
features.remove('id')
features.remove('loss')

X2 = np.vstack([train_data[feat].values for feat in features]).T

imputer = Imputer()
X2 = imputer.fit_transform(X2)

scalar = StandardScaler()
X2 = scalar.fit_transform(X2)

for idx in [1]:
    for train, val in kf:
        #X3 = X2[:, features[:idx]]
        clf = GradientBoostingRegressor(n_estimators=100, loss='lad')
        clf.fit(X[loss > 0][train], loss[loss > 0][train])
    
        loss_hat = clf.predict(X[loss > 0][val])
    
        mae += [np.mean(np.abs(loss_hat - loss[loss > 0][val]))]
    print idx, np.mean(mae)

for learning_rate in [0.1, 0.01]:
    for max_depth in [2,3,4,5,6]:
        shuf = ShuffleSplit(loss.shape[0], 1, 0.5, random_state=0)
        clf = GradientBoostingRegressor(n_estimators=100, learning_rate=learning_rate, max_depth=max_depth, loss='lad', subsample=1.0)
        #clf = GradientBoostingClassifier(n_estimators=500, learning_rate=10**-3, max_depth=3, subsample=0.9)
        for train, test in shuf:
            clf.fit(X[train], loss[train])
            loss_hat = clf.predict(X[test])

            error = []
            for loss_hat in clf.staged_predict(X[test]):
                error.append(np.mean(np.abs(np.round(loss_hat) - loss[test])))

            print max_depth, learning_rate, np.argmin(error), np.min(error)
    

    
    
    
