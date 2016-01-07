import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
from sklearn import metrics as mt
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.ensemble import GradientBoostingRegressor as gbr
import time
import trojanParam as param
import trojanUtil as util

def drop_columns(df, drop_list):
    for col in drop_list:
        print 'dropping column:%r' % col
        df = df.drop(util.get_aname2rname(col),axis=1)
    print df.columns
    return df

def cal_min_intv(minutes_past):
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    #valid_time = valid_time / 60.0
    return valid_time

def min_transform(df):
    df['minutes_past'] = cal_min_intv(df['minutes_past'])    
    return df

def preprocess(df):
    #transforming minutes
    df = df.groupby(level='Id').apply(min_transform)
    
    return df

def driver(df, comp_data):
    exp_range = [10,15,20,25,30]
    data_transform = {'ORIG':1, 'NORM':2}


if __name__=="__main__":
    df = pd.read_csv('ensemble_data\\exp_g_70_80pre_clean.csv',index_col='Id')
    df = preprocess(df)
    #take only records where reflectivity is finite
    df = df[np.isfinite(df['Ref'])]

    #drop columns other than min, radar_dist, ref, refc, Expected
     #dropping ref values
    dropList = [u'REF_10', u'REF_50', u'REF_90', u'REFC_10', u'REFC_50',u'REFC_90',
     u'RHO', u'RHO_10', u'RHO_50', u'RHO_90', u'ZDR', u'ZDR_10', u'ZDR_50', u'ZDR_90', 
     u'KDP', u'KDP_10', u'KDP_50', u'KDP_90']
    df = drop_columns(df, dropList)
    df.to_csv('ensemble_data\\min_test1.csv')





















#print df['minutes_past']
    # min_arr = df['minutes_past']
    # ref_arr = df['Ref']
    # refc_arr = df['RefComposite']
    # print 'min_arr',min_arr
    # print 'ref_arr',ref_arr
    # ref_prod = 0
    # ref_sum = 0
    # refc_prod = 0
    # refc_sum = 0
    # min_p = 0
    # for i in range(len(min_arr)):
    #     if not np.isnan(ref_arr.iloc[i]):
    #         ref_sum = ref_sum + min_arr.iloc[i]
    #         ref_prod = ref_prod + ref_arr.iloc[i]*min_arr.iloc[i]
    #         min_p = min_p + min_arr.iloc[i]
    #     if not np.isnan(refc_arr.iloc[i]):
    #         refc_sum = refc_sum + min_arr.iloc[i]
    #         refc_prod = refc_prod + refc_arr.iloc[i]*min_arr.iloc[i]
        
    # ref = ref_prod/ref_sum
    # refc = refc_prod/refc_sum
    # df = df.mean()

    # df['Ref']= ref
    # #print min_p
    # df['minutes_past'] = min_p
    # #print refc
    # df['RefComposite'] = refc
