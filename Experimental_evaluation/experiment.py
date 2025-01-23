#Install
!pip install numpy
!pip install pytoda
!pip install scikit-optimize
!pip install openpyxl

#import
import pandas as pd
from numpy import mean
from numpy import std
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from threading import Thread
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.base import BaseEstimator, ClassifierMixin
from csv import writer

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

#defining the experiment
def experiment(data_name, model_name, model, params, X_train, y_train, X_test, y_test, i):

    # configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    # define search
    search = BayesSearchCV(model, params, scoring='accuracy', cv=cv_inner, n_iter=10, refit=True, random_state=1, n_jobs=1)

    # execute search
    result = search.fit(X_train, y_train)

    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_

    # evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)

    # evaluate the model
    acc = accuracy_score(y_test, yhat)
    prec = precision_score(y_test, yhat)
    rec = recall_score(y_test, yhat)
    f1 = f1_score(y_test, yhat)
    mcc = matthews_corrcoef(y_test, yhat)

    # store the result
    #results.append([data_name, model_name, i, acc, rec, prec, f1, mcc, result.best_score_, result.best_params_])
    results = [data_name, model_name, i, acc, rec, prec, f1, mcc, result.best_score_, result.best_params_]
    with open('results/'+data_name+'_results.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(results)
        f_object.close()

    # report progress
    print(f"{data_name}, {model_name} {i} > acc={acc:.2f}, est={result.best_score_:.2f}, cfg={result.best_params_}")

# T is a list of transformed data that will be used in experiments
T = []

#load the transformed data from files:

#---1D DeepChem
T.append( ('maccskeys', pd.read_csv('features/maccskeys.csv', header=None).to_numpy()) )
T.append( ('circular', pd.read_csv('features/circular.csv', header=None).to_numpy()) )
T.append( ('mol2vec', pd.read_csv('features/mol2vec.csv', header=None).to_numpy()) )
T.append( ('rdkit', pd.read_csv('features/rdkit.csv', header=None).to_numpy()) )
T.append( ('bpsymmetry', pd.read_csv('features/bpsymmetry.csv', header=None).to_numpy()) )
T.append( ('modlamp', pd.read_csv('features/modlamp.csv', header=None).to_numpy()) )
T.append( ('fastatoseq', pd.read_csv('features/fastatoseq.csv', header=None).to_numpy()) )
T.append( ('smilestoseq', pd.read_csv('features/smilestoseq.csv', header=None).to_numpy()) )
T.append( ('mordred', normalize(pd.read_csv('features/mordred.csv', header=None).to_numpy())) )

#---1D protPy
T.append( ('AAC', pd.read_csv('features/AAC.csv', header=None).to_numpy()) )
T.append( ('PAAC', pd.read_csv('features/PAAC.csv', header=None).to_numpy()) )
T.append( ('APAAC', pd.read_csv('features/APAAC.csv', header=None).to_numpy()) )
T.append( ('CTD', pd.read_csv('features/CTD.csv', header=None).to_numpy()) )
T.append( ('CTriad', pd.read_csv('features/CTriad.csv', header=None).to_numpy()) )
T.append( ('DPC', pd.read_csv('features/DPC.csv', header=None).to_numpy()) )
T.append( ('TPC', pd.read_csv('features/TPC.csv', header=None).to_numpy()) )


#---2D DeepChem
T.append( ('coulombmatrix', pd.read_csv('features/coulombmatrix.csv', low_memory=False, header=None).to_numpy()) )
T.append( ('onehot', pd.read_csv('features/onehot.csv', low_memory=False, header=None).to_numpy()) )
T.append( ('smiles2image', pd.read_csv('features/smiles2image.csv', low_memory=False, header=None).to_numpy()) )

import time

#configure the cross-validation procedure
cv_outer = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)

#list to store thread references
threads = []

#create a thread pool with max worker threads
pool = ThreadPoolExecutor(max_workers=8)

for data_name, X_ in T:

    for i, (train_ix, test_ix) in enumerate(cv_outer.split(X_, y)):

        #split data
        X_train, X_test = X_[train_ix, :], X_[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        for model_name, mp in model_params.items():

            #add experiment to thread list: pool
            exp = pool.submit(experiment, data_name, model_name, mp['model'],mp['params'], X_train, y_train, X_test, y_test, i) # does not block

            #add to the list to save the thread reference
            threads.append(exp)

            time.sleep(0.1)

#waits for threads to finish
for exp in as_completed(threads):
    exp.result()
