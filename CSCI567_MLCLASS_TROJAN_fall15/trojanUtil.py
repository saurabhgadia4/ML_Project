import trojanParam as param
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def drop_columns(df, drop_list):
    for col in drop_list:
        print 'dropping column:%r' % col
        df = df.drop(get_aname2rname(col),axis=1)
    print df.columns
    return df

def write_df2csv(df, fname):
	df.to_csv(os.path.join(param.DATA_PATH,fname), index=False)

def clean_train(df, write=False, fname=None):
    hlist = df[get_aname2rname('ID')].unique()
    for i in range(len(hlist)):
        idx = hlist[i]
        id_df = df[df[get_aname2rname('ID')]==idx]
        id_df = id_df.iloc[:,get_aname2id('REF')]
        id_df = id_df.dropna()
        print 'idx:',idx
        if id_df.empty:
            # delete that ID rows from
            df = df[df[get_aname2rname('ID')]!=idx]
    if write:
    	write_df2csv(df, fname)
    return df

def get_train_test(orig_file, test_size=0.2):
    u_test_x = np.loadtxt(fmat, delimiter=',')
    u_test_y = np.loadtxt(lmat, delimiter=',')
    train_x, test_x, train_y, test_y = cv.train_test_split(narray, nlabel, random_state = 52, test_size=test_size)
    return train_x, test_x, train_y, test_y

def groupby_id(filename):
    df = pd.read_csv(filename, index_col='Id')
    df = df.drop(get_aname2rname('MP'),axis=1)
    df = df.groupby(level='Id').mean()
    return df

def Expected_stat(df=None, filename=None):
    '''
        Expects a pre clean file
    '''
    
    if df is None:
        df = groupby_id(filename)
    print 'Stats\n',df['Expected'].describe()
    for i in range(10):
        print 'Expected Count from %r-%r:%d' % (i*10,(i+1)*10,len(df[(df.Expected>=i*10) & (df.Expected<(i+1)*10)]))
    for i in range(1,6):
        plt.figure()
        df[(df.Expected>=(i*1000)) & (df.Expected<((i+1)*1000))]['Expected'].plot(kind='hist', bins=15)
    plt.figure()
    df[(df.Expected>=0)&(df.Expected<100)]['Expected'].plot(kind='hist',bins=20)
    plt.figure()
    df[(df.Expected>=100)&(df.Expected<6000)]['Expected'].plot(kind='hist',bins=20)
    return df

def Scatter_Plots(df):
    columns = ['Ref', 'RefComposite','RhoHV','Zdr','Kdp']
    for col in columns:
        plt.figure();
        df.plot(kind = 'scatter',y='Expected',x=col)

def get_aname2rname(aname):
    return param.COL_ID2NAME[param.COL_ANAME2ID[aname]]

def get_id2rname(id):
    return param.COL_ID2NAME[id]

def get_id2aname(id):
    return param.ALIAS_NAME[param.COL_ID2NAME[id]]

def get_aname2id(aname):
    return param.COL_ANAME2ID[aname]

def get_rname2id(rname):
    return param.COL_ANAME2ID[param.ALIAS_NAME[rname]]

def get_rname2aname(rname):
    return ALIAS_NAME[rname]

