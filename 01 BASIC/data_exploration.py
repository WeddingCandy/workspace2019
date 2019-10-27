
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')



df_train = pd.read_csv('../01 BASIC/data/train.csv')

print(df_train.columns)


#descriptive statistics summary
print(df_train['SalePrice'].describe())

#histogram
sns.distplot(df_train['SalePrice'])
# plt.show()

#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())