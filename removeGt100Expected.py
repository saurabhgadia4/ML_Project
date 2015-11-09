"""
Remove all the outliers
Outliers: any entry that has expected > 100
"""

import pandas as pd
import numpy as np
import time

infile = 'data\\train_cleaned.csv'
outfile = 'data\\train_outlier_removed.csv'

# read file
df = pd.read_csv(infile, index_col=0)
print 'read data'

# remove the entries with expected > 100
df = df.drop(df[df.Expected > 100].index)

df.to_csv(outfile,header=True)
print 'csv created'