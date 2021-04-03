# -*- coding: utf-8 -*-
"""
Created on Mon May 11 23:18:07 2020

@author: Huri
"""

import pandas as pd
import math

heart_data = pd.read_csv('heart_summary.csv')
print(heart_data.head())

def frequency(array):
    counter1 = 0
    counter2 = 0
    for i in array:
        if(i == array[0]):
            counter1+=1
        else:
            counter2+=1
    return counter1,counter2
            
def gini(data,feature_name,target_class):
    listLeft = [] #True
    listRight = [] #False
    for i in range(len(data)):
        if(data[feature_name][i] == data[feature_name][0]):
            listLeft.append(data[target_class][i])
        else:
            listRight.append(data[target_class][i])
    count1,count2 = frequency(listLeft) #the number of 1 and 0 in the target for left node list
    count3,count4 = frequency(listRight) #the number of 1 and 0 in the target for right node list
    prob1 = count1/len(listLeft) # the probability of target 1 in the left node list
    prob2 = count2/len(listLeft) # the probability of target 0 in the left node list
    prob3 = count3/len(listRight) # the probability of target 1 in the right node list
    prob4 = count4/len(listRight) # the probability of target 0 in the right node list
    total_prob1 = len(listLeft)/(len(listLeft)+len(listRight)) #total probability of left node list
    total_prob2 = len(listRight)/(len(listLeft)+len(listRight)) #total probability of right node list
    gini_left = 1-((prob1**2)+(prob2**2))
    gini_right = 1-((prob3**2)+(prob4**2))
    total_gini = (total_prob1 * gini_left) + (total_prob2 * gini_right)
    return total_gini

def entropy(data,feature_name,target_class):
    listLeft = [] #If true, Node left list
    listRight = [] #If false, Node right list
    for i in range(len(data)):
        if(data[feature_name][i] == data[feature_name][0]):
            listLeft.append(data[target_class][i])
        else:
            listRight.append(data[target_class][i])
    count1,count2 = frequency(listLeft)
    count3,count4 = frequency(listRight)
    prob1 = count1/len(listLeft) # the probability of target 1 in the left node list 
    prob2 = count2/len(listLeft) # the probability of target 0 in the left node list
    prob3 = count3/len(listRight) # the probability of target 1 in the right node list
    prob4 = count4/len(listRight) # the probability of target 0 in the right node list
    total_prob1 = len(listLeft) / (len(listLeft) + len(listRight)) #total probability of left node list
    total_prob2 = len(listRight) / (len(listLeft) + len(listRight))#total probability of right node list
    entropy_left = -((prob1*math.log2(prob1)) + (prob2*math.log2(prob2)))
    entropy_right = -((prob3*math.log2(prob3)) + (prob4*math.log2(prob4)))
    total_entropy = (total_prob1 * entropy_left) + (total_prob2 * entropy_right)
    return total_entropy

def overAllCollectionGini(data,target_class):
    count1,count2 = frequency(data[target_class])
    prob1 = count1/len(data[target_class])
    prob2 = count2/len(data[target_class])
    gini = 1-( prob1**2 + prob2**2)
    return gini

def overAllCollectionEntropy(data,target_class):
    count1,count2 = frequency(data[target_class])
    prob1 = count1/len(data[target_class])
    prob2 = count2/len(data[target_class])
    entropy = -((prob1*math.log2(prob1))+(prob2*math.log2(prob2)))
    return entropy

print("-------------------------------------------------------------------")
print("Gini Index for the overall collection of training examples:",overAllCollectionGini(heart_data,'target'))
print("Entropy for the overall collection of training examples:",overAllCollectionEntropy(heart_data,'target'))

print("-------------------------------------------------------------------")
print("Gini Index​ for the age​ attribute:",gini(heart_data,'age','target'))    
print("Gini Index​ for the cp​ attribute:",gini(heart_data,'cp','target'))  
print("Gini Index​ for the trestbps​ attribute:",gini(heart_data,'trestbps','target'))  

print("-------------------------------------------------------------------")
print("Entropy​ for the age​ attribute:",entropy(heart_data,'age','target'))    
print("Entropy​ for the cp​ attribute:",entropy(heart_data,'cp','target'))  
print("Entropy for the trestbps​ attribute:",entropy(heart_data,'trestbps','target'))  
        
    
    