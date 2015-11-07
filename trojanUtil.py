import trojanParam as param
import os

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

