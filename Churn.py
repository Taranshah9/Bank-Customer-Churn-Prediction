import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
df = pd.read_csv('Churn_Modelling.csv')
print(df.head(10))
df = df.drop('RowNumber',axis=1)
print(df.head(10))
print(df.columns)
df = df.drop('CustomerId',axis=1)
df = df.drop('Surname',axis=1)
print(df.head(10))
print(df.shape)
df = pd.get_dummies(df,prefix=['Geography','Gender'])
print(df.head(10))
y = pd.DataFrame(df['Exited'])
x = df.drop('Exited',axis=1)
print(x.head())
print(y.head())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

