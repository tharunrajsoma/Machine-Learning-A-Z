# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 00:49:30 2018

@author: tsoma
"""
# Data Preprocessing


#inporting Libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data sets:
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#Taking care of missing data:
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #create a object imputer for class Imputer and use ctrl+i for info abt imputer
imputer = imputer.fit(x[:,1:3]) #Fit imputer obj to our data x(training set) where we take columns which has the missing data (here 1:3 means 1 and 2 columns upper bound not included)
x[:,1:3] = imputer.transform(x[:,1:3]) #this 'trnasform' method replaces the missing data by mean of column

#encoding categorical data:
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()#for taking care of 'Country' categorical variable
x[:,0] = labelencoder_x.fit_transform(x[:,0]) #In above function fit and transform were seprate calls here in one call convert coloumn 0 into num
#Now a new problem occurs that is the Names of Countries becomes 0,1,2......and our machine learning model may think that one country is greater than the other based on their values.
#In order to stop this problem we split this country column having three countries into 3 seperate columns based on their names and make the values in each row of a cloumn as '0' or '1' based on the actual country name of the column 
#This concept is called DUMMY ENCODING
onehotencoder = OneHotEncoder(categorical_features=[0]) #this chooses '0' row to get transformed into 3 seperate columns based on name
x = onehotencoder.fit_transform(x).toarray() #As we already gave input as just 0th column there is no need to say fit_transform emthod for column number

#Now just normal encoding is sufficient for dependent variable because it has just 'yes' or 'no' variables and machine learning model will know that it is a category
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting Dataset into training and testing sets:
#Use of this is to know whether the machine learnning model studied correlations correctly. If it learned the test data output predictions will be matchng
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Feature Scaling:
#This will be useful in situations where the difference in values in one column with the same rows of other column is very high(may be multiples of 100) then effective eucledian distance between two data points from those two columns will be dominated by only single column
#This inturn makes the model show data which more influenced based on the column having difference in the values
#So to avoid this situation we need to scale data such that the variables of both the columns are in same range or same scale(ex: both the columns variables range will be in the range of -1 to 1)
#There are many ways of scaling data: Standardisation(x-mean(x)/standarddeviation(x)) and Normalisation(x-min(x)/max(x)-min(x))
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train) #Here we need to fit our object to our training set and then transform it where formula will be used make all entries in the range of -1 to 1
x_test = sc_x.transform(x_test) #Here there is no need for fit as it is already to fit to the training dataset
#NOTE: Here there is no need to do scaling for dependent variable y because it is just 0 or 1........it holds mutliple values then we may need to scale(Ex:Regression models)
#NOTE2: Even if some machine learning models are not based on feature scaling it is better to do because it will help converging the algorithm faster. 
#Only in the case of decision trees as it is not based on euclidean distances we wont do feature scaling so that it can take more time