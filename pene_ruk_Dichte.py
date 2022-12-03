# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:59:00 2022

@author: Bert
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

#AbhVar  = "Pene"
AbhVar  = "RuK "
Produkt = '10/20'
#Produkt = '70/100'
#Produkt = '160/220'
#Produkt = 'ALL'

if (Produkt  == '10/20'):
    df=pd.read_csv('data/10_20_Dichte.csv', sep=';')
    learning_rate=0.0001
    if (AbhVar == "Pene") :
        a=np.array([80, 0.45, 0.55, 0, 0, 0, 1080, 69])
    else:
        a=np.array([80, 0.45, 0.55, 0, 0, 0, 1080, 14])
elif (Produkt == '70/100'):
    df=pd.read_csv('70_100_Dichte.csv', sep=';')
    learning_rate=0.002
    if (AbhVar == "Pene") :
        a=np.array([80, 1   , 0   , 0, 0, 0, 1040, 45])
    else:
        a=np.array([80, 1   , 0   , 0, 0, 0, 1040, 85])
elif (Produkt == '160/220'):
    df=pd.read_csv('160_220_Dichte.csv', sep=';')
    learning_rate=0.0002
    if (AbhVar == "Pene") :
        a=np.array([80, 1   , 0   , 0, 0, 0, 1030, 40])
    else:
        a=np.array([80, 1   , 0   , 0, 0, 0, 1030, 175])
elif (Produkt == 'ALL'):
    df1=pd.read_csv('10_20_Dichte.csv', sep=';')
    df2=pd.read_csv('70_100_Dichte.csv', sep=';')
    df3=pd.read_csv('160_220_Dichte.csv', sep=';')
    df=pd.concat([df1,df2,df3])
    learning_rate=0.0001
    if (AbhVar == "Pene") :
        a=np.array([80, 1   , 0   , 0, 0, 0, 1030, 40])
    else:
        a=np.array([80, 1   , 0   , 0, 0, 0, 1030, 175])
        
result=([['Ergebnis                    ','lineare Regression','Neuronales Netz'],
        ['Vorhersage                   ',0,0],
        ['mean absolut error auf XTrain',0,0],
        ['R²                 auf XTrain',0,0],
        ['mean absolut error auf XTest ',0,0],
        ['R²                 auf XTest ',0,0],
        ['mean absolut error auf beides',0,0],
        ['R²                 auf beides',0,0] ])


fig1, ax1 = plt.subplots(nrows=3,ncols=3)

fig, ax = plt.subplots(nrows=5,ncols=2)
plt.tight_layout()
plt.style.use('seaborn-deep')
X=df.to_numpy()
XVar = X[:,6:9]
for i in range(len(ax1)):
    for j in range(len(ax1)):
        if i == j:
            ax1[i][i].hist(XVar[:,i], bins=50)
        else :
            ax1[i][j].scatter(XVar[:,i],XVar[:,j])
y=X[:,8]
X=X[:,0:8]
if AbhVar == "Pene" :
    y1 = y
    y = np.copy(X[:,7])
    X[:,7]=y1
    X[:,7]= X[:,7]**0.21 * 118
    ax[4][0].plot(X[:,7],y,'.')
    y=y*y1**(3)/1000000
else :
    ax[4][0].plot(X[:,7],y,'.')
    #y=y*X[:,7]/125
    X[:,7]= X[:,7]**-0.21 * 118
    
ax[0][0].scatter(X[:,0],y,marker='.')
ax[0][1].plot(X[:,1],y,'.')
ax[1][0].plot(X[:,2],y,'.')
ax[1][1].plot(X[:,3],y,'.')
ax[2][0].plot(X[:,4],y,'.')
ax[2][1].plot(X[:,5],y,'.')
ax[3][0].plot(X[:,6],y,'.')
ax[3][1].plot(X[:,6],X[:,7],'.')
ax[4][1].plot(X[:,7],y,'.')
ax[4][1].set_xlabel("transformierte Penetration")
    
plt.xlabel('Penetration')
plt.ylabel('R+K')
plt.show()
#X[:,0] = X[:,0]/100

X1 = X[:,6:8].copy()
a=np.array([1083, 14])
#a=np.array([1040, 85])
#a=np.array([1030, 180])
a[1]=a[1]**-0.21 * 118
X=X1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1,y,test_size=0.1, random_state=None)
#print(X_train.shape,X_test.shape)

# lineare regression

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
intercept = model.intercept_
coef = model.coef_
print(intercept,coef,'\n')
result[1][1] = intercept+coef@a

y_pred = model.predict(X_train)
mae    = mean_absolute_error(y_train, y_pred)
r2     = r2_score(y_train, y_pred)
result[2][1] = mae
result[3][1] = r2

y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)
result[4][1] = mae
result[5][1] = r2

y_pred = model.predict(X)
mae    = mean_absolute_error(y, y_pred)
r2     = r2_score(y, y_pred)
result[6][1] = mae
result[7][1] = r2

#print('lineare Regression : ',AbhVar,(intercept+coef@a)*125/a[len(a)-1])
print('lineare Regression : ',AbhVar,(intercept+coef@a))

# neuronales Netz

autoencoder = Sequential()
autoencoder.add(Dense(len(a),input_dim=len(a),activation='linear'))
#autoencoder.add(Dense(7,activation='linear'))
#autoencoder.add(Dense(7,activation='linear'))
autoencoder.add(Dense(1,activation='linear'))
opt = keras.optimizers.Adam(learning_rate=learning_rate)
autoencoder.compile(loss='mean_squared_error', optimizer=opt)
loss_history = autoencoder.fit(X_train, y_train, epochs=500, verbose=False,validation_data=(X_test, y_test))

lh = loss_history.history['loss']
plt.plot(lh[50:])
plt.show

y_pred = autoencoder.predict(X_train)
mae   = mean_absolute_error(y_train, y_pred)
r2    = r2_score(y_train, y_pred)
result[2][2] = mae
result[3][2] = r2

y_pred = autoencoder.predict(X_test)
mae   = mean_absolute_error(y_test, y_pred)
r2    = r2_score(y_test, y_pred)
result[4][2] = mae
result[5][2] = r2

y_pred = autoencoder.predict(X)
mae   = mean_absolute_error(y, y_pred)
r2    = r2_score(y, y_pred)
result[6][2] = mae
result[7][2] = r2

print("auf die Testmenge: R² = ", r2)

result[1][2] = autoencoder.predict(a.reshape(len(a),1).T)[0][0]

#print('neuronales Netz    : ',AbhVar,autoencoder.predict(a.reshape(len(a),1).T)[0][0]*125/a[len(a)-1])
print('neuronales Netz    : ',AbhVar,autoencoder.predict(a.reshape(len(a),1).T)[0][0])
