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
%matplotlib inline

def scores(y, model):
    
    model.fit(xtrain, ytrain.values.ravel())
    y_pred = model.predict(xtest)
    
    print("Mean squared error: %.3f" % mean_squared_error(ytest, y_pred))
    print('R2 score: %.3f' % r2_score(ytest, y_pred))

    cvScore = cross_val_score(model, xtest, ytest.values.ravel(), cv = 3, scoring = 'r2')
    print("R2 cross validation score: %0.2f (+/- %0.2f)" % (cvScore.mean(), cvScore.std() * 2))
    
    for i in y_pred:
        y.append(i)

def data_processing(file_name):
	# Read the csv files with data
	df = pd.read_csv(file_name)
	df.fillna(0, inplace=True)
	return df


# Main
# SVR
df = data_processing("wr_stats.csv")

df1 = df[0:402]
df2 = df[402:602]

train, test = train_test_split(df1, test_size = 0.25, random_state = 10)

xtrain = train[['Rec','Tgt','TD','G','GS','PPR']]
ytrain = train[['Yds']]

xtest = test[['Rec','Tgt','TD','G','GS','PPR']]
ytest = test[['Yds']]

svr = SVR(kernel='rbf', gamma=1e-4, C=100, epsilon = .1)
y_svr = []
scores(y_svr, svr)


playerNames = df1.iloc[:, 1]
dfCurrentPredict = df2[['Rec','Tgt','TD','G','GS','PPR']]

svrPredict = svr.predict(dfCurrentPredict)
svrPredict = svrPredict.tolist()

for (i,j) in zip(playerNames, svrPredict):
	print(i,j)