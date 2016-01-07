import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import trojanParam as param
import trojanUtil as util

def drop_columns(df_objs, drop_list):
    for col in drop_list:
        for obj in df_objs:
            obj = df.drop(util.get_aname2rname(col),axis=1)
    return df_objs

def pre_process(df):
    '''
        call this before applying any estimator
    '''
    pass

def split(df, splits):
    '''
        Range will be the list of tuples containing 
        the ranges for splitting the data to train.
        All preprocessing of the data should be done
        prior to calling this function
    '''
    df = df.groupby(level='Id').mean()
    df_obj = []
    if not splits:
        return [df]
    for sp in splits:
        lower = sp[0]
        upper = sp[1]
        df_obj.append(df[(df.Expected>=lower) &df[(df.Expected<=upper)])
    return df_obj

def statistics(df_objs, splits):
    if df_objs:
        columns = df_objs[0].columns

    for i in range(len(df_objs)):
        if splits[i]:
            print 'data frame range: %r-%r'%(splits[i][0],splits[i][1]) 
        print 'Dataframe Length: %r'%(len(df_objs[i]))

    for col in columns:
        print '\nColumn:%r\n' % (col)
        for i in range(len(df_objs)):
            print '\ndata frame range: %r-%r\n'%(splits[i][0],splits[i][1])
            print '\nDescription\n'
            df_objs[i][col].describe()
            print '\nCorrelation\n'
            df_objs[i][col].corr()

def gen_dfobjs(filename, drop_arr=[], split_arr=[]):
    '''
        Drop all unnecessary columns. Filename shold be pre cleaned.

    '''
    df = pd.read_csv(filename, index_col='Id')
    df = drop_columns([df],drop_arr)[0]
    df_objs = split(df, split_arr)
    return df_objs



if __name__=="__main__":
    filename = 'data_viz\\train_cleaned.csv'
    drop_arr = [u'MP', u'REF_10', u'REF_50', u'REF_90', u'REFC', u'REFC_10', u'REFC_50',u'REFC_90']
    split_arr = [(0,865), (1065,1470),(1650,2100),(2450,3100),(4360,4780)]
    df_objs = gen_dfobjs(filename, drop_arr=drop_arr, split_arr=split_arr)



