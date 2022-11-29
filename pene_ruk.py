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
from keras.layers import Dense
from keras.models import Sequential
from tensorflow import keras

Produkt = '10/20'
#Produkt = '70/100'
#Produkt = '160/220'

if (Produkt  == '10/20'):
    df=pd.read_csv('10_20.csv', sep=';')
    learning_rate=0.01
    a=np.array([80, 0.45, 0.55, 0, 0, 0, 14])
elif (Produkt == '70/100'):
    df=pd.read_csv('70_100.csv', sep=';')
    learning_rate=0.001
    a=np.array([80, 1   , 0   , 0, 0, 0, 85])
elif (Produkt == '160/220'):
    df=pd.read_csv('160_220.csv', sep=';')
    learning_rate=0.001
    a=np.array([80, 1   , 0   , 0, 0, 0, 175])
result=([['Ergebnis für RuK berechnet  ','lineare Regression','Neuronales Netz'],
        ['Vorhersage                   ',0,0],
        ['mean absolut error auf XTrain',0,0],
        ['R²                 auf XTrain',0,0],
        ['mean absolut error auf XTest ',0,0],
        ['R²                 auf XTest ',0,0],
        ['mean absolut error auf beides',0,0],
        ['R²                 auf beides',0,0] ])

X=df.to_numpy()
y=X[:,7]
X=X[:,0:7]
plt.plot(X[:,6],y,'x')
plt.xlabel('Penetration')
plt.ylabel('R+K')
plt.show()
#X[:,0] = X[:,0]/100

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=None)
print(X_train.shape,X_test.shape)

# lineare regression

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
intercept = model.intercept_
coef = model.coef_
print(intercept,coef,'\n')
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

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

print('lineare Regression = ',intercept+coef@a)

y_quer  = intercept+X@coef
delta_y = y - y_quer

# pca mit sklearn, Max, zu Fuß

dim=2
pca = PCA(n_components=dim)
pca_data1 = pca.fit_transform(X)
loadings = pca.components_.T
print('explained variance :',pca.explained_variance_ratio_,pca.explained_variance_ratio_.sum())

from pca import pca
pca_data = pca(X, dimension=dim) 

X_cov = np.cov(X.T)
#print(X_cor)
e_values, e_vectors = np.linalg.eigh(X_cov)

# Sort eigenvalues and their eigenvectors in descending order
e_ind_order = np.flip(e_values.argsort())
e_values = e_values[e_ind_order]
e_vectors = e_vectors[:, e_ind_order] # note that we have to re-order the columns, not rows
#print(e_vectors)
# now we can project the dataset on to the eigen vectors (principal axes)
pca_data2 = X @ e_vectors[: , :dim]

# neuronales Netz

autoencoder = Sequential()
autoencoder.add(Dense(7,input_dim=7,activation='linear'))
#autoencoder.add(Dense(7,activation='linear'))
#autoencoder.add(Dense(7,activation='linear'))
#autoencoder.add(Dense(7,activation='linear'))
#autoencoder.add(Dense(7,activation='linear'))
autoencoder.add(Dense(1,activation='linear'))
opt = keras.optimizers.Adam(learning_rate=learning_rate)
autoencoder.compile(loss='mean_squared_error', optimizer=opt)
loss_history = autoencoder.fit(X_train, y_train, epochs=2000, verbose=False,validation_data=(X_test, y_test))

lh = loss_history.history['loss']
plt.plot(lh[100:])
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

print("aus die Testmenge: R² = ", r2)

print('neuronales Netz    = ',autoencoder.predict(a.reshape(7,1).T))