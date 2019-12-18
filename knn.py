# 2019 WR Yards predictions using K Nearest Neighbors Regression
# authors: Manish Goud, Ram Bala
# 2019 December 18

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from scipy.stats import norm
from operator import itemgetter
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
%matplotlib inline

# Function to display accuracy results of model
def print_accuracy(y, model):
    
    model.fit(xtrain, ytrain.values.ravel())
    y_pred = model.predict(xtest)
    
    print("Mean squared error: %.3f" % mean_squared_error(ytest, y_pred))
    print('R2 score: %.3f' % r2_score(ytest, y_pred))

    cvScore = cross_val_score(model, xtest, ytest.values.ravel(), cv = 3, scoring = 'r2')
    print("R2 cross validation score: %0.2f (+/- %0.2f)" % (cvScore.mean(), cvScore.std() * 2))
    
    for i in y_pred:
        y.append(i)

# Function to read information from csv and return dataframe
def read_file(file_name):
    # Read the csv files with data
    df = pd.read_csv(file_name)
    df.fillna(0, inplace=True) # clean up data
    return df


# Main
# Predicting using KNN
df = read_file("wr_stats.csv")

stats2016to2018 = df[0:603]
midseason2019 = df[603:]

train, test = train_test_split(stats2016to2018, test_size = 0.25, random_state = 10)

# Use variables to train for relationship 
xtrain = train[['Rec','Tgt','TD','G','GS','PPR']] 
# Find relationship between features and Yds variable
ytrain = train[['Yds']]

xtest = test[['Rec','Tgt','TD','G','GS','PPR']]
ytest = test[['Yds']]

# Set up KNN 
knn = neighbors.KNeighborsRegressor(n_neighbors = 10, weights = 'distance')
y_knn = []
print_accuracy(y_knn, knn)

playerNames = midseason2019.iloc[:, 1] # extract player names
predictions = midseason2019[['Rec','Tgt','TD','G','GS','PPR']]

# Use kNN to predict yards for players
knnPredict = knn.predict(predictions)
knnPredict = knnPredict.tolist()

# print values of player and corresponding predicted yards for 2019 season
for (i, j) in zip(playerNames, knnPredict):
    print(i, j)


