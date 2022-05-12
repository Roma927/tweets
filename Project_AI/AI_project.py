# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import os
import re
import string

data=open(r"C:\Users\20115\Downloads\Health-News-Tweets\Health-Tweets\bbchealth.txt","r") 
for line in data:
    #remove  id and timestamp
    data = line.split("|")
    data=data[2]
    #remove hash
    data=data.replace("#","")
    #remove url
    data = re.sub(r"http\S+", "", data)
    data = re.sub(r"www\S+", "", data)
    #convert to lowercase
    data=data.lower()
    #remove words that starts with @
    data = re.sub(r"@\S+", "", data)
    #remove @
    data=data.replace("@","")

    print(data)
    

def jaccard(a , b):
    #calculating the intersection bet elements of set a amd set b
    intersection = list(set(a) & set(b))
    #grouping both elements of set a amd set b without repetition
    union = list(set(a) | set(b))
    #calc. jaccard distance 
    distance = len(intersection)/len(union)
    return distance


#we need to get the nearest centroid to each training example
def assign_clusters(centroids, cluster_array):
#value of each centroid (K , n)
  ExamplesCentroids = []

  for i in range(len(cluster_array)):
    z=10000000
    index = -1
    for j in range(len(centroids)): 
      dis = jaccard(cluster_array[i],centroids[j]) 
      if dis < z :
        z = dis
        index = j
    ExamplesCentroids.append(index)

  return ExamplesCentroids 

def calc_centroids(X, ExamplesCentroids, K):
    m, n = X.shape
    Centroids = np.zeros((K, n))
    #select the centriod for cluster
    for i in range(K) : 
      min_dis_sum = 0
      Centroids_indx=-1
      #store distance 
      min_dis_dp=[]
      for p1 in range(K[i]):
          min_dis_dp.append([])
          dis_sum=0
          # sum for every distances of tweet p1 with every tweet p2 in the same cluster
          for p2 in range(K[i]):
              if p1 != p2:
                  if p1 < p2:
                      cent_dis = min_dis_dp[p1][p2]
                  else:
                      cent_dis= jaccard(K[i][p2][0], K[i][p1][0])
                  min_dis_dp[p2].append(cent_dis)
                  dis_sum += cent_dis
              else:
                    min_dis_dp.append(0)
          # select the minimum sum of distance as a centroid 
          if dis_sum < min_dis_sum:
              min_dis_sum = dis_sum
              Centroids_indx=p2
              
       # append the selected tweet to the centroid list
      Centroids.append(K[i][Centroids_indx][0])
               
    return Centroids

