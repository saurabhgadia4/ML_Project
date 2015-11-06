import trojanParam as param
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__=="__main__":
	df = pd.read_csv(os.path.join(param.DATA_PATH, param.TRAIN_FILE))
	print 'task done'
	print(df.describe())
