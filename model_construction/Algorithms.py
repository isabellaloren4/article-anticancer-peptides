#installation
!pip install numpy
!pip install pytoda
!pip install catboost
!pip install scikit-optimize
!pip install xgboost
!pip install lightgbm
!pip install openpyxl

#Import
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#Definition of models and parameters
model_params = {
          'lr': {'model': LogisticRegression(),
                'params': {
                          'C': Real(1e-4, 1e4, prior='log-uniform'),
                          'fit_intercept': Categorical([True, False]),
                          'solver': Categorical(['newton-cg', 'liblinear', 'sag', 'saga'])}},

          'knn': {'model': KNeighborsClassifier(),
                  'params': {
                            'n_neighbors': Integer(1, 50),
                            'weights': Categorical(['uniform', 'distance']),
                            'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
                            'p': Integer(1, 5)}},

          'nb': {'model': GaussianNB(),
                'params': {
                          'var_smoothing': Real(1e-10, 1e-1, prior='log-uniform')}},

          'dt': {'model': DecisionTreeClassifier(),
                'params': {
                          'criterion': Categorical(['gini', 'entropy']),
                          'splitter': Categorical(['best', 'random']),
                          'max_depth': Integer(3, 30),
                          'min_samples_split': Integer(2, 10),
                          'min_samples_leaf': Integer(1, 10),
                          'max_features': Real(0.1, 1.0, prior='uniform')}},

          'svm': {'model': LinearSVC(),
                  'params': {
                            'C': Real(1e-6, 1e+6, prior='log-uniform'),
                            'loss': Categorical(['hinge', 'squared_hinge']),
                            'tol': Real(1e-6, 1e-2, prior='log-uniform')}},

          'gpc': {'model': GaussianProcessClassifier(),
                  'params': {
                            'optimizer': Categorical(['fmin_l_bfgs_b', None]),
                            'n_restarts_optimizer': Integer(0, 10),
                            'max_iter_predict': Integer(100, 1000)}},

          'mlp': {'model': MLPClassifier(),
                  'params': {
                            'hidden_layer_sizes': Integer(10,100),
                            'activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),
                            'solver': Categorical(['sgd', 'adam']),
                            'alpha': Real(1e-5, 1e-1, prior='log-uniform'),
                            'learning_rate': Categorical(['constant', 'invscaling', 'adaptive']),
                            'learning_rate_init': Real(1e-4, 1e-1, prior='log-uniform'),
                            'max_iter': Integer(1000,1001)}},

          'ridge': {'model': RidgeClassifier(),
                    'params': {
                              'alpha': Real(1e-4, 1e4, prior='log-uniform'),
                              'fit_intercept': Categorical([True, False]),
                              'solver': Categorical(['auto', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])}},

          'rf': {'model': RandomForestClassifier(),
                'params': {
                          'n_estimators': Integer(10, 500),
                          'criterion': Categorical(['gini', 'entropy']),
                          'max_depth': Integer(3, 30),
                          'min_samples_split': Integer(2, 10),
                          'min_samples_leaf': Integer(1, 10),
                          'max_features': Real(0.1, 1.0, prior='uniform'),
                          'bootstrap': Categorical([True, False]),
                          'class_weight': Categorical(['balanced', 'balanced_subsample', None])}},

          'qda': {'model': QuadraticDiscriminantAnalysis(),
                  'params': {
                            'reg_param': Real(0, 1, prior='uniform'),
                            'store_covariance': Categorical([True, False]),
                            'tol': Real(1e-5, 1e-1, prior='log-uniform')}},

          'ada': {'model': AdaBoostClassifier(),
                  'params': {
                            'n_estimators': Integer(10, 500),
                            'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                            'algorithm': Categorical(['SAMME', 'SAMME.R'])}},

          'gbc': {'model': GradientBoostingClassifier(),
                  'params': {
                            'n_estimators': Integer(10, 500),
                            'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                            'max_depth': Integer(3, 10),
                            'min_samples_split': Integer(2, 10),
                            'min_samples_leaf': Integer(1, 10),
                            'max_features': Real(0.1, 1.0, prior='uniform'),
                            'subsample': Real(0.1, 1.0, prior='uniform')}},

          'lda': {'model': LinearDiscriminantAnalysis(),
                  'params': {
                            'solver': Categorical(['lsqr', 'eigen']),
                            'shrinkage': Real(0, 1, prior='uniform'),
                            'tol': Real(1e-6, 1e-4, prior='log-uniform')}},

          'et': {'model': ExtraTreesClassifier(),
                'params': {
                          'n_estimators': Integer(10, 500),
                          'criterion': Categorical(['gini', 'entropy']),
                          'max_depth': Integer(3, 30),
                          'min_samples_split': Integer(2, 10),
                          'min_samples_leaf': Integer(1, 10),
                          'max_features': Real(0.1, 1.0, prior='uniform'),
                          'bootstrap': Categorical([True, False]),
                          'class_weight': Categorical(['balanced', 'balanced_subsample', None])}},

          'xgboost': {'model': XGBClassifier(),
                      'params': {
                                'learning_rate': Real(0.01, 0.3, prior='uniform'),
                                'n_estimators': Integer(50, 500),
                                'max_depth': Integer(3, 10),
                                'min_child_weight': Integer(1, 10),
                                'gamma': Real(0, 1, prior='uniform'),
                                'subsample': Real(0.5, 1, prior='uniform'),
                                'colsample_bytree': Real(0.5, 1, prior='uniform'),
                                'reg_alpha': Real(0, 1, prior='uniform'),
                                'reg_lambda': Real(1, 3, prior='uniform'),
                                'scale_pos_weight': Real(1, 5, prior='uniform')}},

          'lightgbm': {'model': LGBMClassifier(verbose=-1),
                     'params': {
                                'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                                'n_estimators': Integer(10, 500),
                                'num_leaves': Integer(2, 100),
                                'max_depth': Integer(3, 10),
                                'min_child_samples': Integer(1, 50),
                                'min_child_weight': Real(1e-5, 1e-3, prior='log-uniform'),
                                'subsample': Real(0.1, 1.0, prior='uniform'),
                                'colsample_bytree': Real(0.1, 1.0, prior='uniform'),
                                'reg_alpha': Real(0, 1, prior='uniform'),
                                'reg_lambda': Real(0, 1, prior='uniform')}},

          'catboost': {'model': CatBoostClassifier(verbose=0),
                      'params': {
                                'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                                'iterations': Integer(10, 500),
                                'depth': Integer(3, 10),
                                'l2_leaf_reg': Real(1, 10, prior='uniform'),
                                'border_count': Integer(1, 255),
                                'bagging_temperature': Real(0, 1, prior='uniform'),
                                'random_strength': Real(1e-9, 10, prior='log-uniform')}}
          
          'mlp': {'model': MLPClassifier(),
                  'params': {
                            'hidden_layer_sizes': Integer(10,100),
                            'activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),
                            'solver': Categorical(['sgd', 'adam']),
                            'alpha': Real(1e-5, 1e-1, prior='log-uniform'),
                            'learning_rate': Categorical(['constant', 'invscaling', 'adaptive']),
                            'learning_rate_init': Real(1e-4, 1e-1, prior='log-uniform'),
                            'max_iter': Integer(1000,1001)}}
}
