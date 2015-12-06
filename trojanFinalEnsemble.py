import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn import metrics as mt
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV
#from sklearn.feature_selection import SelectFromModel
import time
import os
import random
import trojanClean
import trojanParam as param

def kaggle_predict(est, test_x):
    print 'Test Size:%r' % len(test_x)
    test_y_pred = est.predict(test_x)
    id_col = np.arange(1.,len(test_y_pred)+1, dtype=np.int)
    all_data = np.column_stack((id_col, test_y_pred))
    np.savetxt(os.path.join(param.CURRENT_FOLDER, param.FINAL_TEST_OUT), all_data, delimiter=',', header='Id,Expected')
    df = pd.read_csv(os.path.join(param.CURRENT_FOLDER, param.FINAL_TEST_OUT))
    df.to_csv(os.path.join(param.CURRENT_FOLDER, param.FINAL_TEST_OUT_1), header=True, index=False)


def train_ensemble(fobj, train_2_x=None, hold_x=None, sub_test=None):
    ens_train2_name = 'TRAIN_2_MIX.csv'
    ens_test_name = 'TEST_FINAL_MIX.csv'
    sub_test_name = 'Submission.csv'
    if train_2_x==None:
        train_2_x = np.loadtxt(os.path.join(param.CURRENT_FOLDER, ens_train2_name), delimiter=',')
        test_x = np.loadtxt(os.path.join(param.CURRENT_FOLDER, ens_test_name), delimiter=',')
        orig_test = np.loadtxt(os.path.join(param.CURRENT_FOLDER, 'Submission.csv'), delimiter=',')
    
    train_x = train_2_x[:,:-1] 
    test_x = train_2_x[:,-1]

    stg2_GBR(train_2_x, test_x, hold_x, sub_test)



def stg2_GBR(train_x, train_y, hold_x, sub_test):
    ntrees=5
    exp = [20, 30]
    rate = [0.1, 0.3]
    hold_test_x = hold_x[:,:-1]
    hold_test_y = hold_x[:,-1]
    
    for e in exp:
        print 'Stage 2: Exp Threshold - %r' % e
        fobj.write('Stage 2: Exp Threshold - %r\n'%e)
        c_train = train_x[(train_x[:,-1]<=exp)]
        c_train_y = c_train[:,-1]
        c_train_x = c_train[:,:-1]
        for r in rate:
            kaggle_file = 'Kaggle_GBR_e'+str(e)+'_r'+str(r)+'.csv'
            df_kaggle_file = 'Kaggle_df_GBR_e'+str(e)+'_r'+str(r)+'.csv'
            print 'Stage 2: Rate - %r' % r
            fobj.write('GBR exp: %r rate: %r'%(e,r))
            rf_model = gbr(n_estimators=ntrees, loss='lad', learning_rate=r, max_depth=6)
            est = rf_model.fit(c_train_x, c_train_y)
            
            train_y_pred = est.predict(c_train_x)
            error = mt.mean_absolute_error(c_train_y, train_y_pred)
            print 'GBR Train Error: %r\n' % (error)
            fobj.write('GBR Train-2 Error: %r\n' % (error))

            train_y_pred = est.predict(hold_test_x)
            error = mt.mean_absolute_error(hold_test_y, train_y_pred)
            print 'GBR 20 percent Hold Error: %r\n' % (error)
            fobj.write('GBR 20 percent Hold Error: %r\n' % (error))


            print 'Test Size:%r' % len(sub_test)
            test_y_pred = est.predict(sub_test)
            id_col = np.arange(1.,len(test_y_pred)+1, dtype=np.int)
            all_data = np.column_stack((id_col, test_y_pred))
            np.savetxt(os.path.join(param.CURRENT_FOLDER, kaggle_file), all_data, delimiter=',', header='Id,Expected')
            df = pd.read_csv(os.path.join(param.CURRENT_FOLDER, kaggle_file))
            df.to_csv(os.path.join(param.CURRENT_FOLDER, df_kaggle_file), header=True, index=False)


def gen_ensemble(train_1_x, train_2_x, hold_out, sub_test, fobj):
    train_2_y = train_2_x[:,-1]
    test_x = train_2_x[:,:-1]
    hold_out_y = hold_out[:,-1]
    hold_out_x = hold_out[:,:-1]

    #Adding GBR Prediction
    gbr_est, gbr_y, hold_gbr_y, test_gbr_y = runGBR(train_1_x, test_x, hold_out_x, sub_test, fobj)
    error = mt.mean_absolute_error(train_2_y, gbr_y)
    print 'GBR Train-2 Error: %r\n' % (error)
    fobj.write('GBR Train-2 Error: %r\n' % (error))
    name = 'GBR_TRAIN_2_Y.csv'
    gbr_y_c = np.column_stack((gbr_y, train_2_y))
    np.savetxt(os.path.join(param.CURRENT_FOLDER, name), gbr_y_c, delimiter=',')

    #Adding XTR Prediction
    xtr_est, xtr_y, hold_xtr_y, test_xtr_y = runXTR(train_1_x, test_x, hold_out_x, sub_test, fobj)
    error = mt.mean_absolute_error(train_2_y, xtr_y)
    print 'XTR Train-2 Error: %r\n' % error
    fobj.write('XTR Train-2 Error: %r\n' % (error))
    name = 'XTR_TRAIN_2_Y.csv'
    xtr_y_c = np.column_stack((xtr_y, train_2_y))
    np.savetxt(os.path.join(param.CURRENT_FOLDER, name), xtr_y_c, delimiter=',')

    #Adding RF prediction
    rf_est, rf_y, hold_rf_y, test_rf_y = runRF(train_1_x, test_x, hold_out_x, sub_test, fobj)
    error = mt.mean_absolute_error(train_2_y, rf_y)
    print 'RF Train-2 Error: %r\n' % error
    fobj.write('RF Train-2 Error: %r\n' % (error))
    name = 'RF_TRAIN_2_Y.csv'
    rf_y_c = np.column_stack((rf_y, train_2_y))
    np.savetxt(os.path.join(param.CURRENT_FOLDER, name), rf_y_c, delimiter=',')


    #Adding XGBoost prediction
    # xgb_est, xgb_y, hold_xgb_y, test_xgb_y = runXGB(train_1_x, test_x, hold_out_x, sub_test, fobj)
    # error = mt.mean_absolute_error(train_2_y, xgb_y)
    # print 'XGB Train-2 Error: %r\n' % error
    # fobj.write('XGB Train-2 Error: %r\n' % (error))
    # error = mt.mean_absolute_error(hold_out_y, hold_xgb_y)
    # print 'XGB 20 percent Test Error: %r\n' % error
    # fobj.write('XGB 20 percent Test Error: %r\n')
    # name = 'XGB_TRAIN_2_Y.csv'
    # xgb_y_c = np.column_stack((xgb_y, train_2_y))
    # np.savetxt(os.path.join(param.CURRENT_FOLDER, name), xgb_y_c, delimiter=',')


    test_x  = np.column_stack((test_x, gbr_y))
    hold_out_x  = np.column_stack((hold_out_x, hold_gbr_y))
    sub_test = np.column_stack((sub_test, test_gbr_y))
    
    test_x  = np.column_stack((test_x, xtr_y))
    hold_out_x  = np.column_stack((hold_out_x, hold_xtr_y))
    sub_test = np.column_stack((sub_test, test_xtr_y))
    
    test_x  = np.column_stack((test_x, rf_y))
    hold_out_x  = np.column_stack((hold_out_x, hold_rf_y))
    sub_test = np.column_stack((sub_test, test_rf_y))

    # test_x  = np.column_stack((test_x, xgb_y))
    # hold_out_x  = np.column_stack((hold_out_x, hold_xgb_y))
    # sub_test = np.column_stack((sub_test, test_xgb_y))

    #appending real label to train2
    test_x  = np.column_stack((test_x, train_2_y))
    hold_x = np.column_stack((hold_out_x, hold_out_y))

    ens_train2_name = 'TRAIN_2_MIX.csv'
    ens_test_name = 'TEST_FINAL_MIX.csv'
    sub_test_name = 'Submission.csv'

    np.savetxt(os.path.join(param.CURRENT_FOLDER, ens_train2_name), test_x, delimiter=',')
    np.savetxt(os.path.join(param.CURRENT_FOLDER, ens_test_name), hold_x, delimiter=',')
    np.savetxt(os.path.join(param.CURRENT_FOLDER, sub_test_name), sub_test, delimiter=',')

def runGBR(train_1_x, test_x, hold_out, sub_test, fobj):
    ntrees = 100
    rate = 0.3
    max_depth = 6
    exp = 30

    c_train_1_x = train_1_x[(train_1_x[:,-1]<=exp)]
    c_train_y = c_train_1_x[:,-1]
    c_train_x = c_train_1_x[:,:-1]

    rf_model = gbr(n_estimators=ntrees, loss='lad', learning_rate=rate, max_depth=max_depth)
    est = rf_model.fit(c_train_x, c_train_y)
    train_y_pred = est.predict(c_train_x)
    error = mt.mean_absolute_error(c_train_y, train_y_pred)
    print 'GBR Train Error: %r\n' % (error)
    fobj.write('GBR Train-1 Error: %r\n' % (error))
    valid_y_pred = est.predict(test_x)
    hold_y = est.predict(hold_out)
    sub_y = est.predict(sub_test)
    return est, valid_y_pred, hold_y, sub_y

def runXTR(train_1_x, test_x, hold_out, sub_test, fobj):
    ntrees = 100
    njobs = 2
    exp = 40

    c_train_1_x = train_1_x[(train_1_x[:,-1]<=exp)]
    c_train_y = c_train_1_x[:,-1]
    c_train_x = c_train_1_x[:,:-1]

    rf_model = etr(n_estimators=ntrees, n_jobs=-1)
    est = rf_model.fit(c_train_x, c_train_y)
    train_y_pred = est.predict(c_train_x)
    error = mt.mean_absolute_error(c_train_y, train_y_pred)
    print 'XTR Train-1 Error: %r\n' % (error)
    fobj.write('XTR Train-1 Error: %r\n' % (error))
    valid_y_pred = est.predict(test_x)
    hold_y = est.predict(hold_out)
    sub_y = est.predict(sub_test)
    return est, valid_y_pred, hold_y, sub_y

def runRF(train_1_x, test_x, hold_out, sub_test, fobj):
    ntrees = 100
    njobs = 2
    exp=30

    c_train_1_x = train_1_x[(train_1_x[:,-1]<=exp)]
    c_train_y = c_train_1_x[:,-1]
    c_train_x = c_train_1_x[:,:-1]

    rf_model = rfr(n_estimators=ntrees, n_jobs=-1)
    est = rf_model.fit(c_train_x, c_train_y)
    train_y_pred = est.predict(c_train_x)
    error = mt.mean_absolute_error(c_train_y, train_y_pred)
    print 'RF Train-1 Error: %r\n' % (error)
    fobj.write('RF Train-1 Error: %r\n' % (error))
    valid_y_pred = est.predict(test_x)
    hold_y = est.predict(hold_out)
    sub_y = est.predict(sub_test)
    return est, valid_y_pred, hold_y, sub_y

# def runXGB(train_1_x,test_x, hold_out, sub_test, fobj):
#     exp=20

#     c_train_1_x = train_1_x[(train_1_x[:,-1]<=exp)]
#     c_train_y = c_train_1_x[:,-1]
#     c_train_x = c_train_1_x[:,:-1]

#     xg_train = xgb.DMatrix(c_train_x,label=c_train_y)
#     param = {'bst:eta':0.3, 'silent':0, 'objective':'count:poisson'}
#     num_round = 10
#     bst = xgb.train(param,xg_train,num_round)
#     predicted_validation = bst.predict(xg_train)
#     error = mt.mean_absolute_error(predicted_validation,c_train_y)
#     fobj.write('XGB train-1 Error: %r \n' % (error))
#     xg_test = xgb.DMatrix(test_x)
#     predicted_test = bst.predict(xg_test)
#     hold_y = bst.predict(hold_out)
#     sub_y = bst.predict(sub_test)
#     return est, valid_y_pred, hold_y, sub_y
