DATA_PATH = 'data\\'
TEST_FILE = 'test.csv'
TRAIN_FILE = 'train.csv'
TRAIN_TRIAL_FILE = 'trial_train.csv'
TRIAL_FILE = 'trial1.csv'

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
