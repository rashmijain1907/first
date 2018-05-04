
# coding: utf-8

# In[12]:


from sklearn import svm, metrics
#import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
#import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os
from random import shuffle,sample
from sklearn.model_selection import GridSearchCV


# In[2]:


data_path_train = '/home/rashmi/train/'


# In[3]:


img_size = 64

# Images are stored in one-dimensional arrays of this length.

img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.

img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.

num_channels = 1

# Number of classes, one class for each of 10 digits.

num_classes = 2



# In[4]:


def create_training_data():
    training_data = []
    for img in tqdm(os.listdir(data_path_train)):
        label = 0 if 'cat' in img else 1
        path = os.path.join(data_path_train, img)
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (img_size, img_size))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('training_data_cat_dog_1.npy', training_data)
    return training_data


# In[5]:


train_data = create_training_data()


# In[2]:


winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True


# In[3]:


hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,
                       L2HysThreshold,gammaCorrection,nlevels, signedGradients)


# In[4]:


train_data = np.load('training_data_cat_dog_1.npy')


# In[5]:


#Compute hog features from the matrix of image
train_data_1 = train_data[-25000:]
for i in range(0,25000):
    train_data[i,0] = hog.compute(train_data_1[i,0]).T
#    if train_data_1[i,1][0] == 1:
#        train_data[i,1] = 0
#    else:
#        train_data[i,1] = 1


# In[6]:


l=len(train_data[0,0][0])


# In[7]:


#Create the array of features and the category with the help of hog vector
data = np.ndarray(shape=(25000,l+1))
for i in range(0,25000):
    for j in range(0,l):
        data[i][j]=train_data[i,0][0][j]
    data[i,l] = train_data[i,1]


# In[8]:


X = data[:,0:l-1]
y = data[:,l]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#X_test


# In[ ]:

#grid search for finding the best parameters 

train_size=[20000]
result = [['Training Size', 'Accuracy']]
for size in train_size:
    X_train_2 = X_train[0:size+1]
    y_train_2 = y_train[0:size+1]
    parameters = {'kernel':('linear','rbf'), 'C':[0.01,0.03,0.1,0.3,1,3,10,30]}
    classifier = svm.SVC()
    clf = GridSearchCV(classifier, parameters)
    clf.fit(X_train_2, y_train_2)
    y_pred_2 = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_2)
    result.append([size, (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[1,0]+cm[0,1])])
    print(result)

# In[9]:

# Final experiment with multiple datasizes and iterations

train_size = [1000, 3000, 5000, 10000, 15000, 20000]
result = [['Training Size', 'Accuracy', 'Rerun Count']]
for size in train_size:
    n_reps = int(np.floor(20000/size))
    for i in range(1,n_reps+1):
        X_train_i = X_train[(i-1)*size:(i*size)+1]
        y_train_i = y_train[(i-1)*size:(i*size)+1]
        classifier = svm.SVC(best_parameters that are returned from grid search)
        classifier.fit(X_train_i, y_train_i)
        y_pred_i = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_i)
        result.append([size, (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[1,0]+cm[0,1]), i])
    print(result)