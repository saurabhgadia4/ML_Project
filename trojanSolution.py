import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
from sklearn import metrics as mt
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.ensemble import GradientBoostingRegressor as gbr

def rf_regressor(rf_model, train_x, train_y, valid_x, valid_y):
    est = rf_model.fit(train_x, train_y)
    print 'Train Size:%r'%len(train_x)
    train_y_pred = est.predict(train_x)
    error = mt.mean_absolute_error(train_y, train_y_pred)
    print 'Train Error',error
    valid_y_pred = est.predict(valid_x)
    print 'Train Size:%r'%len(valid_x)
    valid_error = mt.mean_absolute_error(valid_y, valid_y_pred)
    print 'valid Error:%r' % valid_error
    #Load Test X matrix CSV
    test_x = np.loadtxt('C:\Users\saura\Desktop\ML_Project\data\\norm_test_fmat.csv',delimiter=',')
    print 'Test Size:%r' % len(test_x)
    test_y_pred = est.predict(test_x)
    id_col = np.arange(1.,len(test_y_pred)+1, dtype=np.int)
    all_data = np.column_stack((id_col, test_y_pred))
    np.savetxt('C:\Users\saura\Desktop\ML_Project\data\\mytest_solution.csv', all_data, delimiter=',', header='Id,Expected')
    df = pd.read_csv('C:\Users\saura\Desktop\ML_Project\data\\mytest_solution.csv')
    df.to_csv('C:\Users\saura\Desktop\ML_Project\\final_solution.csv', header=True, index=False)

if __name__=="__main__":
    narray = np.loadtxt('C:\Users\saura\Desktop\ML_Project\data\\rho_gr_0_85_fmat.csv',delimiter=',')
    print 'read narray.. size:%r'%(len(narray))
    nlabel = np.loadtxt('C:\Users\saura\Desktop\ML_Project\data\\rho_gr_0_85_label.csv',delimiter=',')
    print 'read label'
    train_x, test_x, train_y, test_y = cv.train_test_split(narray, nlabel, random_state = 42, test_size=0.2)
    rf_model = gbr(n_estimators=100, loss='lad')
    rf_regressor(rf_model, train_x, train_y, test_x, test_y)

    #200 0.99157639  19.635566   19.24871943     FALSE


#gBR results
#n_estimators = 100
# read narray.. size:731556
# read label
# Train Size:585244
# Train Error 23.2042661418
# Train Size:146312
# valid Error:23.190855007768523
# Test Size:717625