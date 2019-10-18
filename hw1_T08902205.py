# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 09:40:22 2019

@author: TaiT_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def computePseudoInverse(X):
    U,D,V = np.linalg.svd(X)
    D_plus = np.zeros((X.shape[0], X.shape[1])).T
    D_plus[:D.shape[0], :D.shape[0]] = np.linalg.inv(np.diag(D))
    X_plus = V.T.dot(D_plus).dot(U.T)
    return X_plus

def computeRMSE(y, y_predict):
    Zigma = np.square(y_predict - y).sum()
    return (Zigma/y.size) ** (1/2)
    
def normalizeColumn(column_name):
    global df_train
    mean = df_train[column_name].mean()
    std = df_train[column_name].std()
    df_train[column_name] = df_train[column_name] - mean
    df_train[column_name] = df_train[column_name]/std
    

def onehot_encoder(column_name):
    global df_train
    # Transform binary columns 
    one_hot = pd.get_dummies(df_train[column_name])
    
    # Rename column names using original column_name
    rename_one_hot_columns = []
    for column in list(one_hot.columns):
        rename_one_hot_columns.append(column_name + '_' + str(column))
    one_hot.columns = rename_one_hot_columns
    
    # Transform binary columns to one-hot encoding vectors.
    df_train = pd.concat([df_train.drop(column_name,axis = 1), one_hot], axis = 1)

# Import the training data
df_train = pd.read_csv('train.csv')

pd.set_option('display.max_columns', None)

#===================== Prepare Data =============================#
# Declare predictors
predictors = [
             'ID',
             'school',
             'sex',
             'age',
             'address', # U,R
             'famsize', # GT3 LT3
             'Pstatus', # T,A
             'Medu', # 1 2 3 4
             'Fedu', # 0 2 3 3
             'Mjob', # other services teacher
             'Fjob',
             'reason', #reputation course home other
             'guardian', #mother other father
             'traveltime', # 1 2 3 4
             'studytime', #1 2 3 4
             'failures',# 0 1
             'schoolsup', #no yes
             'famsup', # no yes
             'paid', # no yes
             'activities',
             'nursery',
             'higher',
             'internet',
             'romantic', # no yes
             'famrel',
             'freetime',
             'goout',
             'Dalc',
             'Walc',
             'health',
             'absences'
             ,'G3'
             ]

# Neglect non-predictor columns
df_train = df_train[predictors]

# Declare columns to be coded
binary_val_columns = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian',
                      'schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']

# Encode all columns that needs to be encoded
for column in binary_val_columns:
    onehot_encoder(column)

# Declare columns to be normalized
columns_not_to_be_normalized = binary_val_columns + ['ID','G3']
columns_to_be_normalized = list(set(predictors) - set(columns_not_to_be_normalized))

for column in columns_to_be_normalized:
    normalizeColumn(column)

#================= Separate Train and Test =======================

# Shuffle data, 80% training, 20% testing
df_train = df_train.drop('ID', axis = 1)
zzdf_train = shuffle(df_train)
cutoff = int(0.8 * len(df_train))
df_train_train = df_train.iloc[:cutoff]
df_train_test = df_train.iloc[cutoff:]

# Transform training sets into numpy array
y_train = df_train_train['G3']
print(df_train_train.head(2))
df_train_train = df_train_train.drop('G3',axis = 1)
x_train = df_train_train.values
y_train = y_train.values

# Transform testing sets into numpy array
y_test = df_train_test['G3']
df_train_test = df_train_test.drop('G3',axis = 1)
x_test = df_train_test.values
y_test = y_test.values

#================== Linear Regression ===============#

# Inverse Formula
A = x_train.T.dot(x_train)
A_plus = computePseudoInverse(A)
theta = A_plus.dot(x_train.T).dot(y_train)
# Predicted Y
y_predict = x_test.dot(theta)
# RMSE of 1b
RMSE_linalg_no_reg_test = computeRMSE(y_test, y_predict)
print("RMSE of 1b: " + str(RMSE_linalg_no_reg_test))

#=============== Linear Regression with reg ==========#
# Inverse Formula with regularization
#A = x_train.T.dot(x_train) + (0.5 * np.eye(x_train.shape[1]))
#A_plus = computePseudoInverse(A)
theta_reg = np.linalg.inv(x_train.T.dot(x_train) + 0.5 * np.identity(x_train.shape[1]) ).dot(x_train.T).dot(y_train)
y_predict_reg = x_test.dot(theta_reg)
RMSE_linalg_with_reg_test = computeRMSE(y_test,y_predict_reg)
print("RMSE of 1c: " + str(RMSE_linalg_with_reg_test))
#theta_reg = np.linalg.inv( x_train.T.dot(x_train) + (0.5 * np.eye(x_train.shape[1])) ).dot(x_train.T).dot(y_train)
# Predict Y with regularization
#y_predict_test_with_reg = x_test.dot(theta_reg) 
# y_ predict_test_with_reg2 = x_test.dot(theta) + theta.T.dot(theta)
# RMSE of 1c
#RMSE_linalg_with_reg_test = computeRMSE(y_test,y_predict_test_with_reg)
#RMSE_linalg_with_reg_test2 = computeRMSE(y_test, y_predict_test_with_reg2)

#================ Add Bias Term to x_train and x_test =====#
x_train_with_bias = x_train[:]
x_test_with_bias = x_test[:]

train_bias = np.ones( (x_train_with_bias.shape[0], 1) )
test_bias = np.ones( (x_test_with_bias.shape[0], 1) )

x_train_with_bias = np.concatenate( (train_bias, x_train_with_bias), axis = 1 )
x_test_with_bias = np.concatenate( (test_bias, x_test_with_bias), axis = 1 )

#================ Linear Regression with reg and bias =====#
reg_term = 0.5 * np.eye(x_train_with_bias.shape[1])
reg_term[0][0] = 0

A = x_train_with_bias.T.dot(x_train_with_bias) + reg_term
A_plus = computePseudoInverse(A)
theta_reg_bias = A_plus.dot(x_train_with_bias.T).dot(y_train)
y_predict_test_with_reg_with_bias = x_test_with_bias.dot(theta_reg_bias)
#theta_reg_bias = np.linalg.inv(x_train_with_bias.T.dot(x_train_with_bias) + reg_term).dot(x_train_with_bias.T).dot(y_train)
RMSE_linalg_reg_bias_test = computeRMSE(y_test, y_predict_test_with_reg_with_bias)
print("RMSE of 1d: " + str(RMSE_linalg_reg_bias_test))

#=============== Bayesian Linear Regression ==============#
reg_term = 1 * np.eye(x_train_with_bias.shape[1])
reg_term[0][0] = 0

A = x_train_with_bias.T.dot(x_train_with_bias) + reg_term
A_plus = computePseudoInverse(A)
theta_bayesian = A_plus.dot(x_train_with_bias.T).dot(y_train)
y_predict_bayesian = x_test_with_bias.dot(theta_bayesian)
RMSE_linalg_bayesian = computeRMSE(y_test,y_predict_bayesian)
print("RMSE of 1e: " + str(RMSE_linalg_bayesian))

#============== Plot ====================#

# Plot
X = range(0,y_test.shape[0])
Y = y_test
Y_no_reg = y_predict
Y_with_reg = y_predict_reg
Y_with_reg_and_bias = y_predict_test_with_reg_with_bias
Y_bayesian = y_predict_bayesian

plt.xlabel("Item No.")
plt.ylabel("Values")

plt.plot(X,Y,'red',label="Ground Truth")
plt.plot(X,Y_no_reg,'blue',label="Linear Regression " + str(round(RMSE_linalg_no_reg_test,2)))
plt.plot(X,Y_with_reg,'green',label="Linear Regression (Reg) " + str(round(RMSE_linalg_with_reg_test,2)))
plt.plot(X,Y_with_reg_and_bias,'purple',label = "Linear Regression (r/b) " + str(round(RMSE_linalg_reg_bias_test,2)))
plt.plot(X,Y_bayesian,'orange',label="Bayesian Liinear Regression" + str(round(RMSE_linalg_bayesian,2)))

plt.legend()
plt.show()

#df_test = pd.read_csv('test_no_G3.csv')
#df_test = df_test[predictors]








