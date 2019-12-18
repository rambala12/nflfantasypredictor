import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline


def data_processing(file_name):
	# Read the csv files with data
	df = pd.read_csv(file_name)
	df.fillna(0, inplace=True)
	return df


# Main
# Linear Regression
df = data_processing("wr_stats.csv")

wr_trainX = df.loc[df["Year"] == 2016]
wr_trainY = df.loc[df["Year"] == 2017]
wr_testY = df.loc[df["Year"] == 2018]
wr_training_set = pd.merge(wr_trainX, wr_trainY, on="Player", how="inner")
wr_testing_set = pd.merge(wr_trainX, wr_trainY, on="Player", how="inner")

wr_lr = LinearRegression()

features = ["Rec_x","PPR_x", "TD_x", "Tgt_x", "G_x", "GS_x"]
wr_lr.fit(wr_training_set[features],wr_training_set["Yds_y"])
prediction = wr_lr.predict(wr_testing_set[features])
wr_lr.score(wr_testing_set[features], wr_testing_set["Yds_y"])

print(prediction,wr_testing_set["Player"])