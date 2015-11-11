import numpy as np
import pandas as pd
from sklearn import linear_model

infile = 'data\\train_outlier_removed.csv'
testfile = 'data\\test.csv'
outfile = 'models\\predicted_by_linear_regression.csv'

df = pd.read_csv(infile, index_col=0)
print 'training data read'

df = df.fillna(df.mean())
print 'nan removed'

column_list = df.columns.values.tolist()

column_list.remove('Expected')

training_x = df[column_list]
training_y = df['Expected']

training_x = training_x.values
training_y = training_y.values
print 'converted to numpy array'

reg = linear_model.LinearRegression()

reg.fit(training_x,training_y)
print 'model generated'

test_df = pd.read_csv(testfile,index_col = 0)
print 'test file read'

test_df = test_df.fillna(test_df.mean())
print 'test nan removed'

test_data = test_df.values
print 'test df converted to numpy array'

result = reg.predict(test_data)
print 'predicted labels for test data'

n = len(result);
id = np.linspace(1, n, num=n, dtype='int32');

df = pd.DataFrame({'ID':id, 'Expected':result});
df = df.set_index('ID');
print 'pandas dataframe with id created'

df.to_csv(outfile,header=True);
print 'saved to outfile'
