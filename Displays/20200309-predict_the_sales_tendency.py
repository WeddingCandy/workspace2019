import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
import pylab
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

df = pd.read_excel(r"20200308-predictes.xlsx" , header=0 , )
df['sub'] = df['sales_2'] - df['sales_1']
print(df.info())
print(df.tail(5))


print(df.iloc[:,1:].corr())

# plt.rcParams['font.sans-serif'] = ['SimHei']


model = LinearRegression()

feature_cols = ['dates']
sub_cols = ['sub']

X = df[feature_cols]
y = df[sub_cols]

inputs =  df['sub']
inputs.plot()
plt.show()
# 绘制自相关图
print(plot_acf(inputs).show())


data_diff = inputs.diff()

# 差分后需要排空，
data_diff = data_diff.dropna()

data_diff.plot()
plt.show()

plot_acf(data_diff).show()
plot_pacf(data_diff).show()


arima = ARIMA(inputs, order=(1, 1, 1))
result = arima.fit(disp=False)
print(result.aic, result.bic, result.hqic)

plt.plot(data_diff)
plt.plot(result.fittedvalues, color='red')
# plt.title('ARIMA RSS: %.4f' % sum(result.fittedvalues - data_diff['income']) ** 2)
plt.show()

inputs.index = inputs.index.map(X.values)


pred = result.predict('2020-03-09', '2020-03-10', typ='levels')
print(pred)
x = pd.date_range('2020-02-03', '2020-03-05')
plt.plot(x[:31], inputs['sub'])
# lenth = len()
plt.plot(pred)
plt.show()
print('end')



# model.fit(X,y)
#
# plt.scatter(X.values, y.values)
#
# plt.plot(X, model.predict(X) , color='blue')
#
# plt.xlabel('日期')
#
# plt.ylabel('差值')
#
# plt.show()
#
# print("截距与斜率:",model.intercept_,model.coef_)

