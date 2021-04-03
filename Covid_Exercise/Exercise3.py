# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:51:37 2020

@author: Huri
"""

import pandas as pd
import math
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

covid_df=pd.read_csv("covid.csv")
print(covid_df.head())

def min_max_normalization_train(dataframe):
    new_df = dataframe.copy()
    for feature_name in dataframe.columns:
        value_max = dataframe[feature_name].max()
        value_min = dataframe[feature_name].min()
        new_df[feature_name] = (dataframe[feature_name] - value_min) / (value_max - value_min)
    return new_df

def min_max_normalization_test(test_data):
    new_test = np.array(test_data)
    for column in range(np.size(new_test,1)):
        new_test[:,column] = (new_test[:,column] - new_test[:,column].min()) / (new_test[:,column].max() - new_test[:,column].min())
    return new_test.tolist()
            
def euclidean_distance(row_test,row_train):
    distance = 0.0
    for i in range(len(row_test)-1):
        distance += math.pow((row_test[i] - row_train[i]),2)
    distance_euclidean = math.sqrt(distance)
    return distance_euclidean

def manhattan_distance(train_row,test_row):
    distance_manhattan = 0.0
    for i in range(len(test_row)-1):
        distance_manhattan += abs((test_row[i] - train_row[i]))
    return distance_manhattan

def take_second(element):
    return element[1]

def helper_find_neighbours(train_data,test_row, distance_type):
    distances=[]
    train_rows=[]
    for i in range(len(train_data)):
        if(distance_type == 'euclidean'):   
            distance = euclidean_distance(test_row,train_data[i])
            train_rows.append(train_data[i])
            distances.append(distance)
        if(distance_type == 'manhattan'): 
            distance = manhattan_distance(test_row,train_data[i]) 
            train_rows.append(train_data[i])
            distances.append(distance)
    distance_list = list(zip(train_rows,distances))
    sorted_distances = sorted(distance_list, key = take_second)
    return sorted_distances
    
def find_neighbours(train_data,test_row, distance_type, k):
    neighbour_list = list()
    sorted_distances = helper_find_neighbours(train_data,test_row, distance_type)
    for j in range(k):
        neighbour_list.append(sorted_distances[j][0])
    return neighbour_list

def output_values(neighbours):
    outputs = []
    for row in neighbours:
        outputs.append(row[2])#the last element
    return outputs
        
def KNN(train_data, test_row,distance_type, k):
    list_train_data = train_data.values
    neighbours = find_neighbours(list_train_data,test_row,distance_type, k)
    output = output_values(neighbours)
    prediction_value = max(set(output), key = output.count)
    return prediction_value
  
def prediction_values(train_data, test_data, distance_type, k):
    predict_list=[]
    for i in range(len(test_data)):
        prediction = KNN(train_data, test_data[i], distance_type, k)
        predict_list.append(prediction)
    return predict_list

def draw_accuracy_graph(norm_train, norm_test, distance_type):
    y =[]
    accuracy_scores=[]
    k_range = range(1, 26)
    for i in range(len(norm_test)):
        y.append(norm_test[i][2]) 
    for j in k_range:
        y_pred = prediction_values(norm_train,norm_test,distance_type,j)
        accuracy_scores.append(metrics.accuracy_score(y,y_pred))
    plt.plot(k_range, accuracy_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    plt.show()

    
    
test_set = [[5,39.0,1], [4,35.0,0], [3,38.0,0], [2,39.0,1], [1,35.0,0], [0,36.2,0], [5,39.0,1], [2,35.0,0], [3,38.9,1], [0,35.6,0]] 

test_set_2 = [[5, 39.0, 1], [4, 35.0, 0], [3, 38.0, 0],
              [2, 39.0, 1], [1, 35.0, 0], [0, 36.2, 0],
              [5, 39.0, 1], [2, 35.0, 0], [3, 38.9, 1],
              [0, 35.6, 0], [4, 37.0, 0], [4, 36.0, 1],
              [3, 36.6, 0], [3, 36.6, 1], [4, 36.6, 1]]

a = min_max_normalization_train(covid_df) 
b = min_max_normalization_test(test_set)
c = min_max_normalization_test(test_set_2)
# e) Calculate output for test_set input: 
print("\nThe prediction values for k=3:",prediction_values(a,b,'euclidean',3))
# f) Try different ​ k values and draw accuracy graph according to k parameter. 
draw_accuracy_graph(a,b,'manhattan')
draw_accuracy_graph(a,c,'euclidean')
# g) What is the best k value according to accuracy? Write as a comment in your source code. 
# k = 17 is the best according to accuracy (manhattan and euclidean)

#References:
#https://medium.com/@cbedirhan41/s%C4%B1f%C4%B1rdan-k-nearest-neighbors-algoritmas%C4%B1-python-4080f6b8dd3d
