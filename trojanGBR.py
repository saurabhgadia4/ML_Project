import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn import metrics as mt
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.feature_selection import RFE
#from sklearn.feature_selection import SelectFromModel
import time
import random
import trojanClean
import trojanParam as param

fobj = None

def rf_regressor(rf_model, train_x, train_y, valid_x, valid_y, generate_csv=False, f_selection=False, n_features=3):
    global fobj
    est = rf_model.fit(train_x, train_y)
    print 'feature importance: ', est.feature_importances_
    if f_selection:
        print "Feature Selection Enabled- feature count: %r" % (n_features)
        rf_model = RFE(rf_model, n_features, step=1)
    importances = est.feature_importances_
    
    std = np.std([tree.feature_importances_ for tree in est.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(train_x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(train_x.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(train_x.shape[1]), indices)
    plt.xlim([-1, train_x.shape[1]])
    plt.show()


    train_y_pred = est.predict(train_x)
    error = mt.mean_absolute_error(train_y, train_y_pred)
    fobj.write('Train Error: %r\n' % (error))
    valid_y_pred = est.predict(valid_x)
    return mt.mean_absolute_error(valid_y, valid_y_pred), error, est

def driver(comp_mat, pps_mth='ORIGINAL', exp_thresh=[30], regression_mth=[], test_size=0.2, f_selection=False, n_features=3):
    global fobj
    fobj.write('Total Number of records: %r\n' % (len(comp_mat)))
    
    #step 1 split Train and test records
    train_x, u_test_x, train_y, u_test_y = cv.train_test_split(comp_mat, comp_mat[:,-1], random_state = 52, test_size=test_size)
    
    #removing the predictor column from the matrix
    u_test_x = u_test_x[:,:-1]

    for exp in exp_thresh:

        fobj.write('\n\nExpected Threshold Limit: %r\n' % (exp))
        #step 2 prning to required exppected threshold
        c_train_x = train_x[(train_x[:,-1]<=exp)]
        c_train_y = c_train_x[:,-1]
        c_train_x = c_train_x[:,:-1]


        #step1 split Train
        ttrain_x, ttest_x, ttrain_y, ttest_y = cv.train_test_split(c_train_x, c_train_y, random_state = 32, test_size=test_size )

        
        fobj.write('Total Number of Constrained test records: %r\n' % len(ttest_y))
        fobj.write('Total Number of Unconstrained test records: %r\n' % len(u_test_y))

        fobj.write('Total Constrained Training Records: %r\n' % len(ttrain_y))

        print 'Fitting Model Measurements'
        trees_array = [50,100]

        kf = cv.KFold(len(ttrain_x), n_folds=5)
        st_time = time.time()

        for ntrees in trees_array:
            valid_acc = []
            test_acc = []
            train_acc = []
            est_arr = []
            unconst_acc = []
            fold = 0

            for train_idx, test_idx in kf:
                fobj.write('\nfold: %r\n'%(fold))
                #rf_model = rfr(n_estimators=ntrees, n_jobs=-1) Random forest regressor
                #rf_model = etr(n_estimators=ntrees, n_jobs=3, bootstrap=False)
                rf_model = gbr(n_estimators=ntrees, loss='lad')
                vacc, tacc, est = rf_regressor(rf_model, ttrain_x[train_idx], ttrain_y[train_idx], ttrain_x[test_idx], ttrain_y[test_idx], f_selection=f_selection, n_features=n_features)
                valid_acc.append(vacc)
                train_acc.append(tacc)
                est_arr.append(est)
                fobj.write('Train Size:%r\n' % (len(train_idx)))
                fobj.write('Validation Size:%r\n' % (len(test_idx)))
                fobj.write('Validation Error: %r\n' % (vacc))
                fobj.write('Train Error: %r\n' % (tacc))
                fobj.write('Constrained Test data size:%r\n' % (len(ttest_x)))
                test_acc.append(mt.mean_absolute_error(ttest_y, est.predict(ttest_x)))
                fobj.write('Constrained Test Error for fold: %r\n' % (test_acc[-1]))
                unconst_acc.append(mt.mean_absolute_error(u_test_y, est.predict(u_test_x)))
                fobj.write('Complete test accuracy:%r\n' % (unconst_acc[-1]))
                fold+=1
                break

            et_time = time.time()
            fobj.write('..Statistics..\n')
            fobj.write('Expected Threshold Limit: %r\n' % (exp))
            fobj.write('Trees:%r\n' % ntrees)
            fobj.write('Train Average: %r\n' % (np.mean(train_acc)))
            fobj.write('Validation Average: %r\n' % (np.mean(valid_acc)))
            fobj.write('Constrained Test Average: %r\n' % (np.mean(test_acc)))
            fobj.write('Unconstrained Test Avg: %r\n' % (np.mean(unconst_acc)))
            fobj.write('Total Time Taken: %r mins\n' % ((et_time-st_time)/60))

            #Print to console
            print('..Statistics..\n')
            print('Expected Threshold Limit: %r\n' % (exp))
            print('Trees:%r\n' % ntrees)
            print('Train Average: %r\n' % (np.mean(train_acc)))
            print('Validation Average: %r\n' % (np.mean(valid_acc)))
            print('Constrained Test Average: %r\n' % (np.mean(test_acc)))
            print('Unconstrained Test Avg: %r\n' % (np.mean(unconst_acc)))
            print('Total Time Taken: %r mins\n' % ((et_time-st_time)/60))

def preprocess(infile, method='ORIGINAL', mp_transform=True, drop_list=[]):
    '''
        Takes the pre_clean file and creates the feature file and label file 
        based on the required method specified
        Methods = Original, Normalize
    '''


    #1. Normalize the data: this will create a feature matrix csv and label csv
    fmat_file = infile.split('.')[0] + '_' +str(exp_thresh)+'_fmat.csv'
    label_file = infile.split('.')[0] + '_' +str(exp_thresh) + '_label.csv'

    if method==param.PREPROCESS_MTH['ORIGINAL']:
        pass
    elif method==param.PREPROCESS_MTH['NORM']: 
        comp_mat = trojanClean.normalize_train(infile, fmat_file, label_file, mp_transform=False, drop_list=drop_list)
    print 'PreProcessing Done'
    return comp_mat


if __name__=="__main__":
    #Original Data transformation parameters
    global fobj
    r_int = str(random.randint(1,1000))
    infile = 'ensemble_data\\train_cleaned_MP.csv'
    pps_mth = param.PREPROCESS_MTH['NORM']

    exp_thresh = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]
    drop_arr = [
                 u'REF_10', u'REF_50', u'REFC', u'REFC_10', u'REFC_50', u'REFC_90', u'RHO', u'RHO_10', u'RHO_50',
                 u'RHO_90', u'ZDR', u'ZDR_10', u'ZDR_50', u'ZDR_90', u'KDP', u'KDP_10', u'KDP_50', u'KDP_90'
                ]
    
    stat_file = 'ensemble_data\\stat_rf_'+r_int+'.txt'
    fobj = open(stat_file, 'w')
    
    comp_mat = preprocess(infile, pps_mth, drop_list=drop_arr)
    driver(comp_mat, pps_mth=pps_mth, exp_thresh=exp_thresh, f_selection=False, n_features=3)
    fobj.close()

'''
COL_ANAME2ID = {
        2, 3, :4,
       5, 6, 7,8, :9,
       10, 11, 12,
       13, 14, 15, 16,
       17, 18, 19, 20,
       21, :22, u'EXP':23
}

u'MP': 0.07868953  u'RDR_DST':0.18300135  u'REF':0.35802427  u'REF_10':0.03851914  u'REF_50':0.08254207  
u'REF_90':0.165098    u'REFC':0.
  u'REFC_10':0.00151652  u'REFC_50':0.00164687  u'REFC_90':0.02285314  u'RHO':0.          u'RHO_10':0.00394372  
  u'RHO_50':0.00178804    u'RHO_90':0.00874705  u'ZDR':0.00216149  u'ZDR_10':0.01308603  
  u'ZDR_50':0.01386322  u'ZDR_90':0.01014912  u'KDP':0. u'KDP_10':0.00959565  u'KDP_50':0.00477479  u'KDP_90'0.

'''