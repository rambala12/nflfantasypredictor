# 2019 WR Yards predictions using Linear Regression
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
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import keras
from keras.models import Sequential
import seaborn as sns
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
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


######### Main Program
# Predicting using Linear Regression
df = read_file("wr_stats.csv")

stats2016to2018 = df[0:603]
midseason2019 = df[603:]

train, test = train_test_split(stats2016to2018, test_size = 0.25, random_state = 10)

# Use variables to train for relationship 
xtrain = train[['Rec','Tgt','TD','G','GS']] 
# Find relationship between features and Yds variable
ytrain = train[['Yds']]

xtest = test[['Rec','Tgt','TD','G','GS']]
ytest = test[['Yds']]

# Set up Linear Regression 
lreg = LinearRegression()
y_lreg = []
print_accuracy(y_lreg, lreg)

playerNames = midseason2019.iloc[:, 1] # extract player names
predictions = midseason2019[['Rec','Tgt','TD','G','GS']]

# Use linear regression to predict yards for players
lregPredict = lreg.predict(predictions)
lregPredict = lregPredict.tolist()


######## Graphs for visualizing results
# run the training again to get values for graphing
reg = LinearRegression().fit(xtrain.values, ytrain.values)

# store values in dataframe variables
ygraph = ytrain.copy(deep=True)
xgraph = ytest.copy(deep=True)

# set up values to graph later
# add prediction column to graph variables from training and testing
ygraph['prediction'] = reg.predict(xtrain.values)
xgraph['prediction'] = reg.predict(xtest.values)

ygraph['yds'] = ygraph['Yds']
xgraph['yds'] = xgraph['Yds']


# Accuracy analysis of our model predictions
xgraph = xgraph[xgraph['prediction'] > 0]
# define accuracy constructs
xgraph.loc[xgraph[['prediction','yds']].max(axis=1)/xgraph[['prediction','yds']].min(axis=1) >= 2, 'Pred'] = 'Very Inaccurate'
xgraph.loc[xgraph[['prediction','yds']].max(axis=1)/xgraph[['prediction','yds']].min(axis=1) < 2, 'Pred'] = 'Inaccurate'
xgraph.loc[xgraph[['prediction','yds']].max(axis=1)/xgraph[['prediction','yds']].min(axis=1) < 1.5, 'Pred'] = 'Accurate'
xgraph.loc[xgraph[['prediction','yds']].max(axis=1)/xgraph[['prediction','yds']].min(axis=1) < 1.2, 'Pred'] = 'Very Accurate'
fig, ax1 = plt.subplots(figsize=(15, 7))
ax = sns.scatterplot(x="yds", y="prediction", hue="Pred", data=xgraph)
ax.set_title("Linear Regression Prediction Accuracy", fontsize=24)
ax.set_xlabel("True Value",fontsize=20)
ax.set_ylabel("Prediction",fontsize=20)
ax.tick_params(labelsize=16)
plt.show()


# Point Differential Error within 200 yards graph
from scipy import stats
xgraph['chip'] = (xgraph['yds'] - xgraph['prediction'])
fig, ax1 = plt.subplots(figsize=(15, 7))
x = xgraph['chip'].values
ax = sns.distplot(x, kde=False, fit=stats.gamma);
ax.set_title("Linear Regression Point Differential Error", fontsize=24)
ax.set_xlabel("Error Margin",fontsize=20)
ax.set_ylabel("% of players",fontsize=20)
ax.tick_params(labelsize=16)
plt.show()
print ("This model is able to relatively accurately predict " + str(round(100*len(xgraph[abs(xgraph['chip'])<=200])/len(xgraph),2)) + "% of WR seasonal yards within 25 yards.")



# print values of player and corresponding predicted yards for 2019 season
for (i, j) in zip(playerNames, lregPredict):
    print(j)


