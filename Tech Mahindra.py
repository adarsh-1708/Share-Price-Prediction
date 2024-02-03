import pandas as pd
import numpy as np

wipro=pd.read_csv('TECHM.csv')
wipro=wipro.drop(columns=['Trades', 'Deliverable Volume','%Deliverble'])

x=wipro.drop(columns=['Date', 'Symbol', 'Series', 'High', 'Low', 'Last',
       'Close', 'Volume', 'Turnover'])
y=wipro.iloc[:,5:6]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.ensemble import GradientBoostingRegressor
lr = GradientBoostingRegressor()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f'R-squared (R²): {r2}')

import pickle
pickle.dump(lr,open('PredictionTECHM.pkl','wb'))
model=pickle.load(open('PredictionTECHM.pkl','rb'))