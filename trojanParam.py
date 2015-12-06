DATA_PATH = 'C:\Users\saura\Desktop\ML_Project\data\\'
DATA_VIZ_PATH = 'C:\Users\saura\Desktop\ML_Project\data_viz\\'
TEST_FILE = 'test.csv'
TRAIN_FILE = 'train.csv'
TRAIN_PRECLEAN = 'train_cleaned.csv'
TRIAL_FILE = 'trial1.csv'
NORMALIZE_OUT = 'norm_train_fmat.csv'
NORMALIZE_TEST_OUT = 'norm_test_fmat.csv'
LABEL_OUT = 'label.csv'
MP_RES = 'MP_RES'
CURRENT_FOLDER='final\\train_test_with_var_stg2_GXR'
MP_TRAIN_FILE = "mp_result.csv"
MP_TEST_FILE = "mp_test_res.csv"
FINAL_TRAIN_INFILE = 'train_cleaned.csv'
FINAL_TEST_INFILE = 'test.csv'

FINAL_TEST_FMAT = 'test_fmat.csv'
FINAL_TEST_OUT = 'kaggle_np.csv'
FINAL_TEST_OUT_1 = 'kaggle_submission.csv'

RHO_INFILE = 'rho_gr_0_85_pre_clean.csv'
FINAL_RHO_INFILE = 'rho_gr_0_85_pre_clean.csv'
LABEL_FILE = 'train_cleaned_y.csv'
TRAIN_1_FINAL_FILE = 'ens_train_1.csv'
TRAIN_2_FINAL_FILE = 'ens_train_2.csv'
TEST_FINAL_FILE = 'ens_test.csv'

EXP_THRESHOLD = {
                     'GBM':30,
                     'XTR':40,
                     'RF':25
}

REG_PARAM = {
                    'GBM':{
                            'n_estimators':100,
                            'max_depth':6,
                            'learning_rate':0.3,
                            'loss':'lad'

                    } ,
                    'RF':{
                            'n_estimators':500,
                            'n_jobs':2 
                    },
                    'XTR':{
                            'n_estimators':100,
                            'n_jobs':2

                    }
}

#To extract the #col from actual column name eg. 'Ref'->3 
#input = u'Ref'
#use: #col = COL_NAME2ID[ALIAS_NAME['Ref']]
#or #col = COL_NAME2ID['REF']

#To extract real column name from #col
#use: COL_ID2NAME[#col]

ALIAS_NAME = {
       u'Id':u'ID', u'minutes_past':u'MP', u'radardist_km':u'RDR_DST', u'Ref':u'REF', u'Ref_5x5_10th':u'REF_10',
       u'Ref_5x5_50th':u'REF_50', u'Ref_5x5_90th':u'REF_90', u'RefComposite':u'REFC',
       u'RefComposite_5x5_10th':u'REFC_10', u'RefComposite_5x5_50th':u'REFC_50',
       u'RefComposite_5x5_90th':u'REFC_90', u'RhoHV':u'RHO', u'RhoHV_5x5_10th':u'RHO_10',
       u'RhoHV_5x5_50th':u'RHO_50', u'RhoHV_5x5_90th':u'RHO_90', u'Zdr':u'ZDR', u'Zdr_5x5_10th':u'ZDR_10',
       u'Zdr_5x5_50th':u'ZDR_50', u'Zdr_5x5_90th':u'ZDR_90', u'Kdp':u'KDP', u'Kdp_5x5_10th':u'KDP_10',
       u'Kdp_5x5_50th':u'KDP_50', u'Kdp_5x5_90th':u'KDP_90', u'Expected':u'EXP'

}
COL_ANAME2ID = {
       u'ID':0, u'MP':1, u'RDR_DST':2, u'REF':3, u'REF_10':4,
       u'REF_50':5, u'REF_90':6, u'REFC':7,u'REFC_10':8, u'REFC_50':9,
       u'REFC_90':10, u'RHO':11, u'RHO_10':12,
       u'RHO_50':13, u'RHO_90':14, u'ZDR':15, u'ZDR_10':16,
       u'ZDR_50':17, u'ZDR_90':18, u'KDP':19, u'KDP_10':20,
       u'KDP_50':21, u'KDP_90':22, u'EXP':23
}

COL_ID2NAME = {
       0:u'Id', 1:u'minutes_past', 2:u'radardist_km', 3:u'Ref', 4:u'Ref_5x5_10th',
       5:u'Ref_5x5_50th', 6:u'Ref_5x5_90th', 7:u'RefComposite',
       8:u'RefComposite_5x5_10th', 9:u'RefComposite_5x5_50th',
       10:u'RefComposite_5x5_90th', 11:u'RhoHV', 12:u'RhoHV_5x5_10th',
       13:u'RhoHV_5x5_50th', 14:u'RhoHV_5x5_90th', 15:u'Zdr', 16:u'Zdr_5x5_10th',
       17:u'Zdr_5x5_50th', 18:u'Zdr_5x5_90th', 19:u'Kdp', 20:u'Kdp_5x5_10th',
       21:u'Kdp_5x5_50th', 22:u'Kdp_5x5_90th', 23:u'Expected'
}

PREPROCESS_MTH = {
       'ORIGINAL':1,
       'NORM':2
}

REGRESSOR = {
       'RFOREST':1,
       'EXTREES':2,
       'GBM':3,
       'LINEAR':4,
       'MARSHALL':5
}