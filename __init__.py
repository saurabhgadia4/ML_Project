import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import trojanParam as param
import trojanUtil as util
import trojanClean
import time
from sklearn import preprocessing as pr

if __name__=="__main__":
    trojanClean.normalize(os.path.join(param.DATA_PATH, param.TRAIN_PRECLEAN))
    trojanClean.normalize_test(os.path.join(param.DATA_PATH, param.TEST_FILE))
    # pre_clean_files = ['exp_le_50.csv', 'exp_g_50_60.csv', 'exp_g_60_70.csv', 'exp_g_70_80.csv', 'exp_g_80_90.csv', 'exp_g_90.csv']

    # for f in pre_clean_files:
    #     out_file = f.split('.')[0] + '_sspre_clean.csv'
    #     trojanClean.pre_cleaning(os.path.join(param.DATA_VIZ_PATH, f), os.path.join(param.DATA_VIZ_PATH,out_file))
    


    # #trojanClean.normalize_train(os.path.join(param.DATA_PATH,'rho_gr_0_85_pre_clean.csv'), fmat_out='rho_gr_0_85_fmat.csv', label_out='rho_gr_0_85_label.csv')
    # 