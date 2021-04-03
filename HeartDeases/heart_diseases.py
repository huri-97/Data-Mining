# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:11:06 2020

@author: Huri
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import wittgenstein as lw
import time
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

heart = pd.read_csv("heart_data.csv")

print(heart.head())
#--------------------------------------------------------------------
df_heart = pd.DataFrame(heart, columns = ['age','cp','trestbps','thalach','chol'])

print(df_heart.head())
#----------------------Replace with younger and older----------------------------
df_heart['age'] = df_heart['age'].apply(str)

for i in df_heart['age']:
    if (int(i) < 55):
        df_heart['age'].replace({i:"younger person"}, inplace=True) 
        
    else:
        df_heart['age'].replace({i:"older person"}, inplace=True)
        
print(df_heart.head())

 #----------------------Numerical Values------------------------------------------       
label_encoder = preprocessing.LabelEncoder()
df_heart['age']=label_encoder.fit_transform(df_heart['age'])

#---------------------Split Train and Test %20---------------------------------------
label_target = heart['target']
x_train, x_test, y_train, y_test = train_test_split(df_heart, label_target, test_size = 0.2, random_state = 100)

#------------------------Decision tree classification-------------------------
print("\n--------------Decision Tree Classifier-----------------------")
tree_start = int(round(time.time() * 1000))
tree_clf = DecisionTreeClassifier()
tree_clf = tree_clf.fit(x_train,y_train)
print("\nThe running time for the tree: %s milliseconds" % (int(round(time.time() * 1000)) - tree_start))
print("The score for the tree: ",tree_clf.score(x_test, y_test))
#------------------------------AUC for the tree---------------------------
tree_probs = tree_clf.predict_proba(x_test)
tree_probs = tree_probs[:,1]
tree_auc = roc_auc_score(y_test, tree_probs)
tree_clf_fpr, tree_clf_tpr, _ = roc_curve(y_test, tree_probs)
print("AUC value for the tree:" , tree_auc)
plt.plot(tree_clf_fpr, tree_clf_tpr, marker='.', label='Tree ' % tree_auc)

#----------------------Ripper algorithm----------------------------------
print("\n---------------Ripper Algorithm-------------------------------")
ripper_start = int(round(time.time() * 1000))
ripper_clf = lw.RIPPER()
ripper_clf.fit(x_train,y_train)
print("\nThe running time for the ripper algorithm: %s milliseconds " % (int(round(time.time() * 1000)) - ripper_start))
print("The score for the ripper algorithm: ",ripper_clf.score(x_test, y_test))

#--------------------AUC for the ripper algorithm--------------------------
ripper_probs = ripper_clf.predict_proba(x_test)
ripper_probs = ripper_probs[:,1]
ripper_auc = roc_auc_score(y_test, ripper_probs)
ripper_clf_fpr, ripper_clf_tpr, _ = roc_curve(y_test, ripper_probs)
print("AUC value for the ripper: ",ripper_auc)
print("------------------------------------------------------------------")
plt.plot(ripper_clf_fpr, ripper_clf_tpr, marker='.', label='Ripper ' % ripper_auc)

