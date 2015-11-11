# import trojanParam as param
import pandas as pd
import numpy as np
import trojanParam as param
import trojanUtil as util
import time
from sklearn import preprocessing as pr
infile = "data\\train.csv"
outfile = "data\\train_cleaned.csv"


#It just removes Id where no REF values are present and stores back to outfile csv
def pre_cleaning(infile, outfile):
    st = time.time()
    # read file
    df = pd.read_csv(infile, index_col=0)
    print 'read data'
    # sum Ref values for each Id
    ref_sums = df['Ref'].groupby(level='Id').sum()
    print 'done grouping'
    # get the index of nan sums values
    null_refs_idx = [i for i in ref_sums.index if np.isnan(ref_sums[i])]
    print 'found null recs'
    # Remove this rows from the dataframe and write new csv file
    df.drop(null_refs_idx, axis = 0, inplace = True)
    print 'dropped ids'
    df.to_csv(outfile, header=True)
    et = time.time()
    print 'Processed Time:',et-st


def normalize(infile, fmat_out=param.NORMALIZE_OUT, label_out=param.LABEL_OUT):
    '''
        #takes the above csv after removal of void id's and normalizes it by imputing Nans.
        #infile: valid clean csv absolute path
        #outfile: write the normalized data should be absolute path
        #return: returns Feature matrix(narray) and its label
        os.path.join(param.DATA_PATH, param.TRAIN_TRIAL_FILE)

        illustration:


        >>> df = pd.DataFrame({'A':[1,1,2,3,3],'B':[np.nan,3,4,1,np.nan],'C':[4,5,np.nan,2,3]})
        >>> df.set_index(['A'])
            B   C
        A
        1 NaN   4
        1   3   5
        2   4 NaN
        3   1   2
        3 NaN   3
        >>> df
           A   B   C
        0  1 NaN   4
        1  1   3   5
        2  2   4 NaN
        3  3   1   2
        4  3 NaN   3
        >>> df = df.set_index(['A'])
        >>> df
            B   C
        A
        1 NaN   4
        1   3   5
        2   4 NaN
        3   1   2
        3 NaN   3
        >>> df = df.groupby(level='A').mean()
        >>> df
           B    C
        A
        1  3  4.5
        2  4  NaN
        3  1  2.5
        >>> dmat = df.as_matrix()
        >>> dmat
        array([[ 3. ,  4.5],
               [ 4. ,  nan],
               [ 1. ,  2.5]])
        >>> imp = pr.Imputer(missing_values='NaN',strategy='mean')
        >>> temp = imp.fit(dmat)
        >>> narray = imp.transform(dmat)
        >>> narray
        array([[ 3. ,  4.5],
               [ 4. ,  3.5],
               [ 1. ,  2.5]])
        >>> pr.scale(narray)
        array([[ 0.26726124,  1.22474487],
               [ 1.06904497,  0.        ],
               [-1.33630621, -1.22474487]])

    '''
    df = pd.read_csv(infile, index_col='Id')
    print 'Read the CSV file'
    #step 0 Remove columns minutes and label from the dataframe
    df = df.drop(util.get_aname2rname('MP'),axis=1)
    print df.columns

    #step 1 groupby wrt to Id and take their mean and then seperate the label column.
    df = df.groupby(level='Id').mean()
    print 'Done groupby mean'
    label_mat = df[util.get_aname2rname('EXP')].as_matrix()
    np.savetxt(label_out, label_mat, delimiter=',')
    df = df.drop(util.get_aname2rname('EXP'),axis=1)
    print df.columns

    #step2 convert dataframe to numpy array
    df_mat = df.as_matrix()

    #step3 impute
    imp = pr.Imputer(missing_values='NaN',strategy='mean')
    temp = imp.fit(df_mat)
    narray = imp.transform(df_mat)

    print 'Done Imputing'
    #Convert to unit variance and 0 mean
    narray = pr.scale(narray)
    print 'Done Scaling'

    #write to csv file by converting to dataframe
    np.savetxt(fmat_out, narray, delimiter=',')
    print 'Done writing to CSV'

    return narray, label_mat
