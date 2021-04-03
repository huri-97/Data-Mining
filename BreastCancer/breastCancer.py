# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:09:19 2020

@author: Lenovo
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import neighbors 
from sklearn import preprocessing
#read data
breastCancer_df=pd.read_csv("wdbc.data")
print(breastCancer_df.shape)
print(breastCancer_df.head())

#------------------VISUALIZATION TECHNIQUES---------------------------------------
def techniques_visualization():
    #box plot
    print("Box Plot")
    sns.boxplot(x='diagnosis', y='radius_mean', data=breastCancer_df)
    plt.show()
    #histogram
    print("Histogram")
    breastCancer_df.pivot(columns='diagnosis', values='perimeter_mean').plot(kind='hist', subplots=True, rwidth=0.9)
    plt.xlabel('Perimeter_mean')
    plt.show()
    #Also, the code below can be used for histogram
    breastCancer_df.groupby('diagnosis').smoothness_mean.hist(stacked=True)
    plt.xlabel('Smoothness_mean')
    plt.ylabel('Count')
    plt.show()
    #scatter plot
    print("Scatter plot")
    plt.scatter(x=breastCancer_df['diagnosis'], y=breastCancer_df['area_mean'], alpha=0.5, edgecolors='none', s=50)
    plt.xlabel('Diagnosis')
    plt.ylabel('Area_mean')
    plt.show()
    sns.lmplot(x = 'area_mean', y = 'smoothness_mean', hue = 'diagnosis', data = breastCancer_df)
    plt.show()
    #correlation matrix
    print("Correlation Matrix")
    plt.figure(figsize=(20,10)) 
    sns.heatmap(breastCancer_df.corr(), annot=True) 
    plt.show()
   
    
#techniques_visualization()

#---------------------------------OUTLIER (INTERQUARTILE RULE)---------------------------
def calculate_bounds(df,feature,constant):
    # calculate interquartile range
    Q1 = np.percentile(df[feature],25)
    Q3 = np.percentile(df[feature],75)
    interQR = Q3 - Q1
    #calculate bounds
    lower_bound = Q1 - (constant*interQR)
    upper_bound = Q3 + (constant*interQR)
    list=[lower_bound,upper_bound]
    return list

def find_outliers(df,feature,constant):
    bound=calculate_bounds(df,feature,constant)
    outliersVector = (df[feature] < bound[0]) | (df[feature] > bound[1]) #index of outliers in data
    outliers = df[feature][outliersVector]
    return outliers
  
def remove_outlier(df,feature,constant):
    bound=calculate_bounds(df,feature,constant)
    #if a value is greater than lower bound and less than upper bound,
    #then the value is kept in the data frame and return new data frame
    df_out = df[~((df[feature] < bound[0]) | (df[feature] > bound[1]))]
    return df_out 

def remove_outliers_all_column(dataFrame):
    lower = .05
    upper = .96
    quantile = dataFrame.quantile([lower, upper])#the first and third quartile from the data
    for feature in dataFrame.columns:
        if is_numeric_dtype(dataFrame[feature]):#controls whether the feature is numeric or not
 #controls whether each value is greater than lower bound and is less than upper bound, then create new data frame without outliers
           dataFrame = dataFrame[(dataFrame[feature] > quantile.loc[lower, feature]) 
               & (dataFrame[feature] < quantile.loc[upper, feature])] 
    return dataFrame
"""    
print("The indexes and values of the outliers")
print(find_outliers(breastCancer_df,'radius_mean',1.2))
print("Before removing outliers from specific column in data")
breastCancer_df.boxplot('radius_mean', widths=0.5)
plt.show()
print("After removing outliers from specific column in data")
remove_outlier(breastCancer_df,'radius_mean',1.2).boxplot('radius_mean', widths=0.5)
plt.show()
print("After removing outliers from all columns in data")
remove_outliers_all_column(breastCancer_df).boxplot('radius_mean', widths=0.5)
plt.show()
"""
#-----------------------PREPROCESSING-------------------------------
#------------------- MIN_MAX NORMALIZATION ------------------------
breastCancer_df=breastCancer_df.drop(['id'], axis = 1) #data cleaning
dataSubset = breastCancer_df.iloc[:, 1:] #takes data from the second column(no id and diagnosis) 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(dataSubset)
print("Before normalization")
#img=plt.boxplot(dataSubset)
#plt.show()
print("After normalization")
#img=plt.boxplot(x_scaled)
#plt.show()
#----------------------------------------------------------------
#------------------DATA MINING - CLASSIFICATION------------------
y = np.array(breastCancer_df['diagnosis']) 
x = np.array(x_scaled)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=(0.25), random_state=123)
print(x_train)
print(y_train)

print("\nData Mining Classification")
print("\nx_train: " + str(x_train.shape))
print("x_test: "+ str(x_test.shape))
classifier=neighbors.KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)
wdbc_ped = classifier.predict(x_test)                             
print("\nAccuracy score: " + str(accuracy_score(y_test,wdbc_ped)))
print("\nConfusion matrix: ")
print(confusion_matrix(y_test,wdbc_ped))
#---------------------------------------------------------------
