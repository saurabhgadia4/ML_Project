# import trojanParam as param
import pandas as pd
import numpy as np
import time
infile = "data\\train.csv"
outfile = "data\\train_cleaned.csv"

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

