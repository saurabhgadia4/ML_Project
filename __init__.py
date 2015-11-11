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
