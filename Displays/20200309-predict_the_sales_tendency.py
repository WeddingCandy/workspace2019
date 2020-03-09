import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
import pylab


df = pd.read_excel(r"20200308-predictes.xlsx" , header=0 , )
df['sub'] = df['sales_2'] - df['sales_1']
print(df.info())
print(df.head(5))

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.scatter(df.iloc[:,0],df.iloc[:,3])

plt.xlabel("日期")

plt.ylabel("差值")

pylab.show()


