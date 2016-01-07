import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn import metrics as mt
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing as pr
#from sklearn.feature_selection import SelectFromModel
import time
import random
import trojanClean
import trojanParam as param
import trojanUtil as util
import trojanFinalClean as FinalClean
import trojanFinalEnsemble as FinalEnsemble

def split_data(comp_mat, test_size=0.2, train2_size=0.5):

    #Generating 80-20 train and test data
    train_x, u_test_x, train_y, u_test_y = cv.train_test_split(comp_mat, comp_mat[:,-1], random_state = 52, test_size=test_size)

    #step 1

    #ignore labels as X matrix have labels in it already
    #save U_test_x as test file using np.
    np.savetxt(os.path.join(param.CURRENT_FOLDER, param.TEST_FINAL_FILE), u_test_x, delimiter=',')

    #step 2 - Generate two train data set for training individial models and ensemble model
    train_1_x, train_2_x, train_1_y, train_2_y = cv.train_test_split(train_x, train_x[:,-1], random_state = 52, test_size=train2_size)

    #step 3 - write the train data to csv files
    np.savetxt(os.path.join(param.CURRENT_FOLDER, param.TRAIN_1_FINAL_FILE), train_1_x, delimiter=',')
    np.savetxt(os.path.join(param.CURRENT_FOLDER, param.TRAIN_2_FINAL_FILE), train_2_x, delimiter=',')

if __name__=="__main__":
    r_int = str(random.randint(1,10000))
    sname = '100_ank_f_var_stg2_'+ r_int
    stat_file = os.path.join(param.CURRENT_FOLDER, sname)
    fobj = open(stat_file, 'w')
    
    mp_train_mat = FinalClean.get_marshall_result(os.path.join(param.CURRENT_FOLDER, param.MP_TRAIN_FILE))
    mp_test_mat = FinalClean.get_marshall_result(os.path.join(param.CURRENT_FOLDER, param.MP_TEST_FILE))
    drop_list = [u'MP']
    '''
    drop_list = [
                 u'MP', u'REF_10', u'REF_50', u'REFC', u'REFC_10', u'REFC_50', u'REFC_90', u'RHO_10',
                  u'RHO_50', u'RHO_90', u'ZDR', u'ZDR_10', u'ZDR_50', u'ZDR_90', u'KDP', u'KDP_10', u'KDP_50', u'KDP_90'
                ]
    '''
    train_comp_mat, test_comp_mat = FinalClean.normalize(os.path.join(param.CURRENT_FOLDER, param.FINAL_TRAIN_INFILE), os.path.join(param.CURRENT_FOLDER, param.FINAL_TEST_INFILE), mp_train_mat=mp_train_mat, mp_test_mat=mp_test_mat, drop_list=drop_list, add_variance=False)
    split_data(train_comp_mat)
    train_1_x = np.loadtxt(os.path.join(param.CURRENT_FOLDER, param.TRAIN_1_FINAL_FILE), delimiter=',')
    train_2_x = np.loadtxt(os.path.join(param.CURRENT_FOLDER, param.TRAIN_2_FINAL_FILE), delimiter=',')
    test_x = np.loadtxt(os.path.join(param.CURRENT_FOLDER, param.TEST_FINAL_FILE), delimiter=',')
    orig_test = np.loadtxt(os.path.join(param.CURRENT_FOLDER, 'test_final_test_norm_wmp_fmat.csv'), delimiter=',')
    
    FinalEnsemble.gen_ensemble(train_1_x, train_2_x, test_x, orig_test, fobj)

    FinalEnsemble.train_ensemble(fobj)
