import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
from sklearn import metrics as mt
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV
#from sklearn.feature_selection import SelectFromModel
import time
import random
import trojanClean
import trojanParam as param

if __name__=="__main__":
    drop_arr = [
                 u'MP', u'REF_10', u'REF_50', u'REFC', u'REFC_10', u'REFC_50', u'REFC_90', u'RHO', u'RHO_10', u'RHO_50',
                 u'RHO_90', u'ZDR', u'ZDR_10', u'ZDR_50', u'ZDR_90', u'KDP', u'KDP_10', u'KDP_50', u'KDP_90',
                ]
    infile = 'data\\test.csv'

    #trojanClean.pre_cleaning(infile, outfile)
    fmat_out = 'ensemble_data\\norm_test_fmat.csv'
    trojanClean.normalize_test(infile, fmat_out, drop_list=drop_arr)