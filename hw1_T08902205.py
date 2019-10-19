# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 09:40:22 2019

@author: TaiT_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#def computePseudoInverse(X):
#    U,D,V = np.linalg.svd(X)
#    D_plus = np.zeros((X.shape[0], X.shape[1])).T
#    D_plus[:D.shape[0], :D.shape[0]] = np.linalg.inv(np.diag(D))
#    X_plus = V.T.dot(D_plus).dot(U.T)
#    return X_plus

def computeRMSE(y, y_predict):
    Zigma = np.square(y_predict - y).sum()
    return (Zigma/y.size) ** (1/2)
    
#def normalizeColumn(column_name, target):
#    clone = target.copy()
#    mean = clone[column_name].mean()
#    std = clone[column_name].std()
#    clone[column_name] = clone[column_name] - mean
#    clone[column_name] = clone[column_name]/std
#    return clone
    

def onehot_encoder(column_name, target):
    clone = target.copy()
    # Transform binary columns 
    one_hot = pd.get_dummies(clone[column_name])
    
    # Rename column names using original column_name
    rename_one_hot_columns = []
    for column in list(one_hot.columns):
        rename_one_hot_columns.append(column_name + '_' + str(column))
    one_hot.columns = rename_one_hot_columns
    
    # Transform binary columns to one-hot encoding vectors.
    clone = pd.concat([clone.drop(column_name,axis = 1), one_hot], axis = 1)
    return clone

# Import the training data
df_train = pd.read_csv('train.csv')

pd.set_option('display.max_columns', None)

#===================== Prepare Data =============================#
# Declare predictors
predictors = [
             'school',
             'sex',
             'age',
#             'address', # U,R
             'famsize', # GT3 LT3
#             'Pstatus', # T,A
#             'Medu', # 1 2 3 4
#             'Fedu', # 0 2 3 3
#             'Mjob', # other services teacher
#             'Fjob',
#             'reason', #reputation course home other
#             'guardian', #mother other father
#             'traveltime', # 1 2 3 4
             'studytime', #1 2 3 4
             'failures',# 0 1
#             'schoolsup', #no yes
#             'famsup', # no yes
#             'paid', # no yes
             'activities',
#             'nursery',
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

binary_val_columns = ['school','sex','famsize','activities','higher','internet','romantic']

# Encode all columns that needs to be encoded
for column in binary_val_columns:
    df_train = onehot_encoder(column, df_train)

columns_to_be_normalized = ['age','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences']


#================= Separate Train and Test =======================

# Shuffle data, 80% training, 20% testing
df_train = shuffle(df_train)
cutoff = int(0.8 * len(df_train))
train_set = df_train.iloc[:cutoff]
cv_set = df_train.iloc[cutoff:]

normalizeVal = []

y_train_set = train_set['G3']
y_train = y_train_set.values
y_test_set = cv_set['G3']
y_test = y_test_set.values

train_set = train_set.drop('G3',axis = 1)
cv_set = cv_set.drop('G3',axis = 1)

for column in train_set.columns:
    mean = train_set[column].mean()
    std = train_set[column].std()
    train_set[column] = (train_set[column] - mean) / std
    cv_set[column] = (cv_set[column] - mean) / std

x_train = train_set.values
x_test = cv_set.values

#================== Linear Regression ===============#

# Inverse Formula
theta = np.linalg.pinv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
# Predicted Y
y_predict = x_test.dot(theta)
# RMSE of 1b
RMSE_linalg_no_reg_test = computeRMSE(y_test, y_predict)
print("RMSE of 1b: " + str(RMSE_linalg_no_reg_test))

#=============== Linear Regression with reg ==========#

# Inverse Formula with regularization
theta_reg = np.linalg.pinv(x_train.T.dot(x_train) + 0.5 * np.identity(x_train.shape[1]) ).dot(x_train.T).dot(y_train)
y_predict_reg = x_test.dot(theta_reg)
RMSE_linalg_with_reg_test = computeRMSE(y_test,y_predict_reg)
print("RMSE of 1c: " + str(RMSE_linalg_with_reg_test))

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

theta_reg_bias = np.linalg.pinv(x_train_with_bias.T.dot(x_train_with_bias) + reg_term).dot(x_train_with_bias.T).dot(y_train)
y_predict_test_with_reg_with_bias = x_test_with_bias.dot(theta_reg_bias)
RMSE_linalg_reg_bias_test = computeRMSE(y_test, y_predict_test_with_reg_with_bias)
print("RMSE of 1d: " + str(RMSE_linalg_reg_bias_test))

#=============== Bayesian Linear Regression ==============#
reg_term = 1 * np.eye(x_train_with_bias.shape[1])
reg_term[0][0] = 0

theta_bayesian = np.linalg.pinv(x_train_with_bias.T.dot(x_train_with_bias) + reg_term).dot(x_train_with_bias.T).dot(y_train)
y_predict_bayesian = x_test_with_bias.dot(theta_bayesian)
RMSE_linalg_bayesian = computeRMSE(y_test, y_predict_bayesian)
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

#============== Apply our model on Test Set ====================#

df_test = pd.read_csv('test_no_G3.csv')

predictors_clone = predictors.copy()
predictors_clone.pop() # Remove G3

df_test = df_test[predictors_clone]

for column in binary_val_columns:
    df_test = onehot_encoder(column, df_test)

for column in df_test.columns:
    mean = df_test[column].mean()
    std = df_test[column].std()
    df_test[column] = (df_test[column] - mean) / std    

x_test = df_test.values

#reg_term = 3 * np.eye(x_train_with_bias.shape[1])
#reg_term[0][0] = 0
#
#theta_bayesian_optimize_alpha = np.linalg.pinv(x_train_with_bias.T.dot(x_train_with_bias) + reg_term).dot(x_train_with_bias.T).dot(y_train)
#y_predict_bayesian_test = x_test_with_bias.dot(theta_bayesian_optimize_alpha)
#RMSE_linalg_bayesian_optimize_alpha = computeRMSE(y_test, y_predict_bayesian_test)
#print("RMSE of bayesian with optimize alpha " + str(RMSE_linalg_bayesian_optimize_alpha))


bias_term = np.ones( (x_test.shape[0], 1) )

x_test = np.concatenate( (bias_term, x_test), axis = 1 )

y_predict_test = x_test.dot(theta_bayesian)

print(y_predict_test)








