from math import *
import cv2
import os
import random
#from numpy import random
import pandas as pd
import numpy as np
#from sklearn.decomposition import FastICA
from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
path1 = 'C:/Users/Rachana Singh/PycharmProjects/directory/directory_1/'
path2 = 'C:/Users/Rachana Singh/PycharmProjects/directory/directory_2/'
path3 = 'C:/Users/Rachana Singh/PycharmProjects/directory/directory_3/'
path4 = 'C:/Users/Rachana Singh/PycharmProjects/pos osu/'
path5 = 'C:/Users/Rachana Singh/PycharmProjects/neg osu/'
labels = []
train_data = []
model = GaussianNB()
predictions = []
sum1 = 0
sum2 = 0
p=0

#matrix = np.full((64,32),255)
#print(matrix)

def dcttransform(matrix,m,n):
    dct = [[0 for x in range(n)] for y in range(m)]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == 0:
                ci = 1/sqrt(m)
            else:
                ci = sqrt(2)/sqrt(m)
            if j == 0:
                cj = 1/sqrt(n)
            else:
                cj = sqrt(2)/sqrt(n)
            s = 0
            for k in range(matrix.shape[0]):
                for l in range(matrix.shape[1]):
                    dct1 = matrix[k][l] * cos((2*k+1) * i * pi / (2*m)) * cos((2*l+1) * j * pi / (2*n))
                    s += dct1
            dct[i][j] = ci*cj*s
    return dct
#descriptor function below is only for square blocks
def descriptor(matrix,block_size,block_stride,col_max,no_of_funccall):
    n = 0
    a = 0
    b = block_size
    c = 0
    d = block_size
    D = []
    while n != no_of_funccall:
        m = matrix[a:b, c:d]
        A = np.array(dcttransform(m, block_size, block_size)).flatten()[:21]
        D.extend(A)
        if d < col_max:
            d += block_stride
            c += block_stride
        else:
            a += block_stride
            b += block_stride
            c = 0
            d = block_size
        n += 1
    return D

"""c = 0
for filename in os.listdir(path1):
    image = cv2.imread(path1+filename, 0)
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    train_data.append(descriptor(small_image))
    c += 1
    print(c)
    labels.append(1)"""
c = 0
for filename in os.listdir(path4):
    image = cv2.imread(path4+filename, 0)
    #small_image = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
    train_data.append(descriptor(image,8,4,16,21))
    c += 1
    print(c)
    labels.append(1)
for filename in os.listdir(path5):
    image = cv2.imread(path5+filename, 0)
    #small_image = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
    train_data.append(descriptor(image,8,4,16,21))
    c += 1
    print(c)
    labels.append(0)
arr_pos = np.zeros(c)
arr_neg = np.zeros(c)
train_data = np.array(train_data)
labels = np.array(labels)
print(train_data.shape)
X_train = train_data
y_train = labels
#X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=109)

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
for i in range(1000):
    bootstrap_samples = np.random.randint(low=0, high=len(X_train), size=100)
    df_bootstrapped = X_train.iloc[bootstrap_samples]
    new_trainSetY = y_train.iloc[bootstrap_samples]
    df_bootstrapped = np.array(df_bootstrapped)
    new_trainSetX = []
    index = 0
    for j in range(100):
        bootstrap_features = random.sample(list(df_bootstrapped[index]), 300)
        index+=1
        new_trainSetX.append(bootstrap_features)
    new_trainSetX = np.array(new_trainSetX)
    #print(new_trainSetX.shape)
    """ica = FastICA(n_components=10)
    S = ica.fit_transform(new_trainSetX)
    #print(S.shape)
    A_= np.array(ica.components_)
    print(A_.shape)"""
    #print(new_trainSetX)
    new_trainSetY = np.array(new_trainSetY)
    model.fit(new_trainSetX, np.ravel(new_trainSetY))
    predicted = model.predict(train_data[:,:300])
    score = accuracy_score(labels, predicted)
    error = 1-score
    print(score)
    if score>0.40:
        wb = (1 / 2 * log(score / error, e))
        post_prob_neg = np.array(model.predict_log_proba(train_data[:,:300])[:,:1]).flatten()
        post_prob_pos = np.array(model.predict_log_proba(train_data[:,:300])[:,1:2]).flatten()
        arr1 = np.multiply(wb, post_prob_neg)
        arr2 = np.multiply(wb, post_prob_pos)
        arr_neg = np.add(arr_neg,arr1)
        arr_pos = np.add(arr_pos,arr2)
        p = p+1
print(p)

for k in range(train_data.shape[0]):
    if arr_neg[k]>=arr_pos[k]:
        predictions.append(0)
    else:
        predictions.append(1)


predictions = np.array(predictions)
#print(predictions.shape)
print(predictions)
print(np.sum(predictions==labels)/float(len(predictions)))
