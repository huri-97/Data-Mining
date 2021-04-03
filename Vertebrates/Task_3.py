# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:04:47 2020

@author: Huri
"""

import pandas as pd
from itertools import combinations
import re

vertebrates_data = pd.read_csv('vertebrates.csv')

def all_combinations(original_data):
    comb_list = []
    all_comb_list = []
    data = original_data.copy()
    data.drop(["Name", "Class"], axis = 1, inplace = True) 
    comb_one = data.columns
    comb_twice = list(combinations(data.columns,2))
    comb_triple = list(combinations(data.columns,3))
    comb_list.append(comb_one)
    comb_list.append(comb_twice)
    comb_list.append(comb_triple)
    for i in comb_list:
        for j in i:
            all_comb_list.append(j)
    return data,comb_list, all_comb_list
            
def rule_single_attribute(att,original_data):
    data = original_data.copy()
    data = data.drop_duplicates(subset=[att,'Class'],keep='first')
    data = pd.DataFrame(data,columns=[att,'Class'])
    data = data.reset_index(drop=True)
    created_rule=[]
    for i in range (len(data)):
        combine_for_rule = data.columns[0]+str("=")+data[data.columns[0]][i]+str("->")+data[data.columns[1]][i]
        created_rule.append(combine_for_rule)
    return created_rule

def rule_two_attribute(att,att2,original_data):
    data=original_data.copy()
    data = data.drop_duplicates(subset=[att,att2,'Class'],keep='first')
    data = pd.DataFrame(data,columns=[att,att2,'Class'])
    data = data.reset_index(drop=True)
    created_rule = []
    for i in range (len(data)):
        combine_for_rule = data.columns[0]+str("=")+data[data.columns[0]][i]+str("^")+data.columns[1]+str("=")+data[data.columns[1]][i]+str("->")+data[data.columns[2]][i]
        created_rule.append(combine_for_rule)
    return created_rule

def rule_three_attribute(att,att2,att3,original_data):
    data=original_data.copy()
    data = data.drop_duplicates(subset=[att,att2,att3,'Class'],keep='first')
    data = pd.DataFrame(data,columns=[att,att2,att3,'Class'])
    data = data.reset_index(drop=True)
    created_rule = []
    for i in range (len(data)):
        combine_for_rule = data.columns[0]+str("=")+data[data.columns[0]][i]+str("^")+data.columns[1]+str("=")+data[data.columns[1]][i]+str("^")+data.columns[2]+str("=")+data[data.columns[2]][i]+str("->")+data[data.columns[3]][i]
        created_rule.append(combine_for_rule)
    return created_rule

#b)obtaining rules with single attribute
def all_rule_single_attribute(comb_one,original_data):
    rule_with_one = []
    for i in comb_one:
        combine_1 = rule_single_attribute(i,original_data)
        for j in combine_1:
            rule_with_one.append(j)
    return rule_with_one

#c)creating rules with two attributes   
def all_rule_two_attribute(comb_twice,original_data):
    rule_with_two = []
    for i in comb_twice:
        combine_2 = rule_two_attribute(i[0],i[1],original_data)
        for j in combine_2:
            rule_with_two.append(j)
    return rule_with_two

#d)creating rules with three attributes
def all_rule_three_attribute(comb_triple,original_data):
    rule_with_three = []
    for i in comb_triple:
        combine_3 = rule_three_attribute(i[0],i[1],i[2],original_data)
        for j in combine_3:
            rule_with_three.append(j)
    return rule_with_three

#e)to store all rules with single, two and three attributes
def all_rules(comb_list, original_data):
    rules_list = []
    rules_all = []
    all_1_att = all_rule_single_attribute(comb_list[0], original_data)
    all_2_att = all_rule_two_attribute(comb_list[1], original_data)
    all_3_att = all_rule_three_attribute(comb_list[2], original_data)
    rules_list.append(all_1_att)
    rules_list.append(all_2_att)
    rules_list.append(all_3_att)
    for i in rules_list:
        for j in i:
            rules_all.append(j)
    return rules_all,len(all_1_att),len(all_2_att),len(all_3_att)

def print_list(any_list):
    for i in any_list:
        print(i)

# f) Implement​ coverage ​ and ​ accuracy formula for all rules
def rule_and_coverage_accuracy(all_rules_list,df,range1,range2,range3):
    all_list =[]
    acc_cov_list = []
    for i in range(range1): # for rules with single attribute by computing coverage and accuracy values
        splitting = re.split('=|->',all_rules_list[i])
        A = len(df.loc[df[splitting[0]]==splitting[1]])
        D = len(df)
        accuracy_numerator = len(df.loc[(df[splitting[0]]==splitting[1]) & (df['Class']==splitting[2])]) #condition for computing accuracy
        all_list.append(all_rules_list[i]+str(" {")+str("coverage:")+str(A/D)+str(", accuracy: ")+str(accuracy_numerator/A)+str("}"))
        acc_cov_list.append([i,A/D,accuracy_numerator/A])
    for i in range(range1,range1+range2): # for rules with two attribute by computing coverage and accuracy values
        splitting = re.split('=|\^|->',all_rules_list[i])
        A = len(df.loc[(df[splitting[0]]==splitting[1])&(df[splitting[2]]==splitting[3])])
        D = len(df)
        accuracy_numerator = len(df.loc[(df[splitting[0]]==splitting[1])&(df[splitting[2]]==splitting[3]) & (df['Class']==splitting[4])])
        all_list.append(all_rules_list[i]+str(" {")+str("coverage:")+str(A/D)+str(", accuracy: ")+str(accuracy_numerator/A)+str("}"))
        acc_cov_list.append([i,A/D,accuracy_numerator/A])
    for i in range(range1+range2,range1+range2+range3): # for rules with three attribute by computing coverage and accuracy values
        splitting = re.split('=|\^|->',all_rules_list[i])
        A = len(df.loc[(df[splitting[0]]==splitting[1])&(df[splitting[2]]==splitting[3])&(df[splitting[4]]==splitting[5])])
        D = len(df)
        accuracy_numerator = len(df.loc[(df[splitting[0]]==splitting[1])&(df[splitting[2]]==splitting[3])&(df[splitting[4]]==splitting[5]) & (df['Class']==splitting[6])])
        all_list.append(all_rules_list[i]+str(" {")+str("coverage:")+str(A/D)+str(", accuracy: ")+str(accuracy_numerator/A)+str("}"))
        acc_cov_list.append([i,A/D,accuracy_numerator/A])
    return all_list,acc_cov_list

def find_index(list1):
    y= []
    for i in range(len(list1)):
        y.append(list1[i][0])
    return y
    
def find_and_print(index_list,all_list):
    count=1
    for i in index_list:
        for j in range(len(all_list)):
            if(i==j):
                print(str(count) + str("-"),all_list[j])
                count+=1
    
def ten_rules(all_list,accuracy_coverage_list):
    sorting_coverage = sorted(accuracy_coverage_list, key = lambda x: (x[1], x[2]),reverse=True)[0:10]
    sorting_accuracy = sorted(accuracy_coverage_list, key = lambda x: (x[2], x[1]),reverse=True)[0:10]
    print("--------------------------------------------")
    print("Ten Rules According to Coverage")
    find_and_print(find_index(sorting_coverage),all_list)
    print("--------------------------------------------")
    print("Ten Rules According to Accuracy")
    find_and_print(find_index(sorting_accuracy),all_list)
#-----------------------------------------------------------------------------------------------------
# a) Print number of classes​, number of attributes ​and generate all possible combinations of attributes​
print("The classes:",vertebrates_data['Class'].unique())
dropped_class_name_data,comb_list, all_comb_list = all_combinations(vertebrates_data)
all_rules_list,range1,range2,range3 = all_rules(comb_list,vertebrates_data)
print("The attributes:",dropped_class_name_data.columns)
print("The number of classes:",len(vertebrates_data['Class'].unique()))
print("The number of attributes:",len(dropped_class_name_data.columns))
print("---------------------------------------------")
print("---All Possible Combinations of Attributes---")
print_list(all_comb_list)
#---------------------------------------------------------------------------------------------------
# e) Store them in a rule set variable, print ​total rule count and​ all rules
print("---------------------------------------------")
print("Total Rule Count: ",len(all_rules_list))
print("---------------------------------------------")
print("-----------------ALL RULES-------------------")
print_list(all_rules_list)
#---------------------------------------------------------------------------------------------------
#f) All rules with coverage and accuracy values
rules_with_cov_and_acc,coverage_accuracy = rule_and_coverage_accuracy(all_rules_list, vertebrates_data,range1,range2,range3)
# Sorted Ten Rules with coverage and accuracy values 
ten_rules(rules_with_cov_and_acc,coverage_accuracy)






        