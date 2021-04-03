# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:58:28 2020

@author: Huri
"""

import glob
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt 

#-------------------------------TASK 1 ----------------------------------
list_vector_img = []
filename_img = []

for filename in glob.glob('Car_Data/*.png'): 
    img = mpimg.imread(filename)
    filename_img.append(filename[9:])
    vector_img = img.flatten()
    list_vector_img.append(vector_img)#vector
    
def show_image(similar3,img_in):
    plt.title("Input")
    plt.imshow(img_in)
    plt.show()
    for i in range(len(similar3)):
        img = mpimg.imread("Car_Data/"+str(similar3[i][1]))
        plt.title(similar3[i][0])
        plt.imshow(img)
        plt.show()
        
def calculate_cosine_similarity(vector1,vector2):
    dot_product = np.dot(vector1,vector2)
    euclidean_operation = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos_similarity = dot_product / euclidean_operation
    return cos_similarity
    
def most_similar_three_images(input_image):
    cos_similarity = []
    remov_vector_list = list_vector_img.copy() #local variable instead of global for removing vector
    remov_name_list = filename_img.copy() #local variable for removing png name
    img2 = mpimg.imread("Car_Data/"+str(input_image))
    vector_input = img2.flatten()
    remov_vector_list.pop(remov_name_list.index(input_image))
    remov_name_list.remove(input_image) 
    for i in remov_vector_list:
        cos = calculate_cosine_similarity(vector_input,i)
        cos_similarity.append(cos) #cos list       
    max3_values = sorted(zip(cos_similarity,remov_name_list), reverse=True)[0:3] #combining two list and select the max three values   
    show_image(max3_values,img2)
    return max3_values        

#print("The most similar three images with similarities:",most_similar_three_images('3952.png'))
print("The most similar three images with similarities:",most_similar_three_images('4228.png'))
print("The most similar three images with similarities:",most_similar_three_images('3861.png'))   
       
#------------------------------------------------------------------------------

