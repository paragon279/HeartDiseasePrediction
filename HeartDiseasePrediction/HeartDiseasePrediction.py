import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
'exec(% matplotlib inline)'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sn

# Loading the Dataset
## dataset
disease_df = pd.read_csv("../input/framingham.csv")
disease_df.drop(['education'], inplace = True, axis = 1)
disease_df.rename(columns = {'male':'Sex_male'}, inplace = True)

## removing NaN / NULL values
disease_df.dropna(axis = 0, inplace = True)
print(disease_df.head(), disease_df.shape)
print(disease_df.TenYearCHD.value_counts())

# counting number of patients affected with CHD
plt.figure(figsize=(7,5))
sn.countplot(x='TenYearCHD', data=disease_df, palette="BuGn_r")
plt.show()