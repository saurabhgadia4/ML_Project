import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
from sklearn import metrics as mt
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.ensemble import GradientBoostingRegressor as gbr
import time

TRAIN_SIZE = 10000

def rf_regressor(rf_model, train_x, train_y, valid_x, valid_y, generate_csv=False):
    est = rf_model.fit(train_x, train_y)
    train_y_pred = est.predict(train_x)
    error = mt.mean_absolute_error(train_y, train_y_pred)
    print 'Train Error',error
    valid_y_pred = est.predict(valid_x)
    return mt.mean_absolute_error(valid_y, valid_y_pred), error, est


if __name__=="__main__":
    narray = np.loadtxt('C:\Users\saura\Desktop\ML_Project\data\\rho_gr_0_85_fmat.csv',delimiter=',')
    #n_real = np.loadtxt('C:\Users\saura\Desktop\ML_Project\data\\norm_fmat.csv',delimiter=',')
    print 'read narray.. size:%r'%(len(narray))
    nlabel = np.loadtxt('C:\Users\saura\Desktop\ML_Project\data\\rho_gr_0_85_label.csv',delimiter=',')
    #label_real = np.loadtxt('C:\Users\saura\Desktop\ML_Project\data\\label.csv',delimiter=',')
    print 'read label'
    train_x, test_x, train_y, test_y = cv.train_test_split(narray, nlabel, random_state = 42, test_size=0.2)
    ttrain_x, ex_x, ttrain_y, ex_y = cv.train_test_split(train_x, train_y, test_size=0.5)
    #ttrain_x = train_x
    #ttrain_y = train_y
    trees_array = [10,50,100,200]
    kf = cv.KFold(len(ttrain_x),n_folds=5)
    st_time = time.time()
    for ntrees in trees_array:
        valid_acc = []
        test_acc = []
        train_acc = []
        est_arr = []
        fold = 0

        for train_idx, test_idx in kf:
            print '\nfold: %r'%(fold)
            #rf_model = rfr(n_estimators=ntrees, n_jobs=-1) Random forest regressor
            #rf_model = etr(n_estimators=ntrees, n_jobs=3, bootstrap=False)
            rf_model = gbr(n_estimators=ntrees, loss='lad')
            vacc, tacc, est = rf_regressor(rf_model, ttrain_x[train_idx], ttrain_y[train_idx], ttrain_x[test_idx], ttrain_y[test_idx])
            valid_acc.append(vacc)
            train_acc.append(tacc)
            est_arr.append(est)
            print 'Train Size:%r' % (len(train_idx))
            print 'Validation Size:%r' % (len(test_idx))
            print 'Validation Error: %r' % vacc
            print 'Train Error: %r' % tacc
            print 'Test data size:%r' % (len(test_x))
            test_acc.append(mt.mean_absolute_error(test_y, est.predict(test_x)))
            print 'Test Error for fold: %r' % test_acc[-1]
            fold+=1
        et_time = time.time()
        print '..Statistics..' 
        print 'Trees:%r' % ntrees
        print 'Train Average: %r' % (np.mean(train_acc))
        print 'Validation Average: %r' % (np.mean(valid_acc))
        print 'Test Average: %r' % (np.mean(test_acc))
        print 'Total Time Taken: %r mins' % ((et_time-st_time)/60)