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
import copy
import trojanClean
import trojanParam as param
import trojanUtil as util

def get_marshall_result(mp_file):
    mp_df = pd.read_csv(mp_file, index_col='Id')
    stripped_df = pd.DataFrame()
    stripped_df[param.MP_RES] = mp_df['Expected'] 
    #dropping all columns
    mp_mat = stripped_df.as_matrix()
    print 'Shape of MP Result', np.shape(mp_mat)
    return mp_mat
    
def label_to_csv(df, label_file):
    label_mat = df[util.get_aname2rname('EXP')].as_matrix()
    np.savetxt(label_file, label_mat, delimiter=',')
    df = df.drop(util.get_aname2rname('EXP'),axis=1)
    return df, label_mat

def normalize(train_file, test_file, mp_train_mat=None, mp_test_mat=None, add_variace=False, drop_list=[]):
    train_df = pd.read_csv(train_file, index_col='Id')
    print 'Read Train_file columns:',train_df.columns
    test_df = pd.read_csv(test_file, index_col='Id')
    print 'Read Test_file columns:',test_df.columns

    ip_train = train_file.split('.')
    ip_test = test_file.split('.')
    train_fmat_file = ip_train[0] + '_final_train_norm_wmp_fmat.csv'
    train_label_file = ip_train[0]+'_final_train_y.csv'
    test_fmat_file = ip_test[0] + '_final_test_norm_wmp_fmat.csv'

    #drop all unwanted columns
    train_df = util.drop_columns(train_df, drop_list)
    test_df = util.drop_columns(test_df, drop_list)

    #step 1 groupby wrt to Id and take their mean and then seperate the label column.
    if add_variace:
        train_temp = copy.deepcopy(train_df)
        test_temp = copy.deepcopy(test_df)
    train_temp = train_temp.groupby(level='Id').var()
    test_temp = test_temp.groupby(level='Id').var()

    train_df = train_df.groupby(level='Id').mean()
    test_df = test_df.groupby(level='Id').mean()
    if add_variace:
        train_df['Var'] = train_temp['Ref']
        test_df['Var'] = test_temp['Ref']

    train_df, l_mat = label_to_csv(train_df, train_label_file)

    #step2 convert dataframe to numpy array
    train_df_mat = train_df.as_matrix()
    test_df_mat = test_df.as_matrix()

    #step3 impute
    imp = pr.Imputer(missing_values='NaN',strategy='mean')
    temp = imp.fit(train_df_mat)
    train_f_mat = imp.transform(train_df_mat)

    imp = pr.Imputer(missing_values='NaN',strategy='mean')
    temp_test = imp.fit(test_df_mat)
    test_f_mat = imp.transform(test_df_mat)

    print 'Done Imputing'
    #Convert to unit variance and 0 mean
    train_f_mat = pr.scale(train_f_mat)
    test_f_mat = pr.scale(test_f_mat)
    print 'Done Scaling'

    #appending the result of marshall Palmer
    print 'Appending mp_mat result to fmat'
    print 'Size train fmat',np.shape(train_f_mat)
    print 'Size mp_mat',np.shape(mp_train_mat)
    print 'Size test fmat',np.shape(test_f_mat)
    print 'Size mp_mat',np.shape(mp_test_mat)


    train_f_mat = np.column_stack((train_f_mat, mp_train_mat))
    print 'Train Shape after appending',np.shape(train_f_mat)


    test_f_mat = np.column_stack((test_f_mat, mp_test_mat))
    print 'Test Shape after appending',np.shape(test_f_mat)

    #write to csv file by converting to dataframe
    np.savetxt(train_fmat_file, train_f_mat, delimiter=',')
    np.savetxt(test_fmat_file, test_f_mat, delimiter=',')
    print 'Done writing to CSV'
    train_comp_mat = np.column_stack((train_f_mat, l_mat))
    return train_comp_mat, test_f_mat