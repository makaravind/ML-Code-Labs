#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:32:11 2018

@author: debanjanchakroborty
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as math
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics



linReg = LinearRegression()
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)
x_vars =  ['TV']
x = data[x_vars]

y = data['sales']

print()
#x_train,x_test,y_train,y_test = train_test_split(x,y)
n,m = x.shape
tr_end = math.floor(n*0.9)
print(n, tr_end)
x_train,x_test,y_train,y_test = (x.iloc[0:tr_end-1],
                                x.iloc[tr_end:n-1], 
                                y.iloc[0:tr_end-1], 
                                y.iloc[tr_end: n-1] )
linReg.fit(x_train,y_train)
print(linReg.intercept_)
print (linReg.coef_)
#print(data.head())
#print(y_train)


zip(x_train,y_train,linReg.coef_)

#x_train,x_test,y_train,y_test = (x[10:100],x[:10],y[10:100],y[0:10])
linReg.fit(x_train,y_train)

y_pred = linReg.predict(x_test)
print(y_pred)
print('loss is',metrics.mean_absolute_error(y_test , y_pred))
print('mean squared loss',metrics.mean_squared_error(y_test , y_pred))

#plt.plot(x_train,y_train)

intercept = linReg.intercept_
slope = linReg.coef_

plt.scatter(x_train, y_train,  color='black')
plt.title('Test Data')
plt.xlabel('TV')
plt.ylabel('Sales')
#plt.xticks(np.arange(0, 1000, step = 6))
plt.xticks(())
#plt.yticks(())
x_train = x_train.sort_values(by='TV')

plt.plot([x_train.iloc[0], x_train.iloc[-1]], [slope*x_train.iloc[0]+intercept,slope*x_train.iloc[-1]+intercept], color='red',linewidth=3)
#plt.plot(x_test,y_test,color = 'red',linewidth = 3)
plt.show()

#print(data.shape)
#import seaborn as sns
#_data = pd.concat([x_train, y_train], axis=1)
#sns.pairplot(_data,x_vars = ['TV'],y_vars = ['sales'],kind = 'reg', size = 7)
