import pandas as pd
df1 = pd.read_csv(r'C:\Users\20261799\Desktop\39W(2).csv' , encoding='utf-8')
df2 = pd.read_csv(r'C:\Users\20261799\Desktop\30W.csv' , encoding='utf-8')
df_join = df1.join(df2 ,how='left')
df_output =df_join[df_join['deviceverK'].isnull().values==True ]
df_join = df_output[['machineidm' ,'deviceverm']]
print(df_join.head())
df_join.to_csv(r'C:\Users\20261799\Desktop\match.csv' ,index=0)