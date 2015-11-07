import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import trojanParam as param
import trojanUtil as util
import time

if __name__=="__main__":
    st_time = time.time()
    df = pd.read_csv(os.path.join(param.DATA_PATH, param.TRAIN_TRIAL_FILE))
    util.clean_train(df, write=True, fname='trial_train_clean.csv')
    end_time = time.time()
    print 'Total Time:',end_time-st_time