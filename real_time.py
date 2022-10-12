import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def reduceDimension(originalX):
  pca_dims = PCA()
  pca_dims.fit(originalX)
  cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
  d = np.argmax(cumsum >= 0.95) + 1
  print("d:",d)
  pca = PCA(n_components = d)
  reducedX = pca.fit_transform(originalX)
  recoveredX = pca.inverse_transform(reducedX)
  return reducedX, recoveredX, pca

#read data
df = pd.read_csv('./sign_mnist_whole.csv')
X = df.iloc[:,1:]
Y = df.iloc[:,0:1]
X.head()
Y.head()
#reduce vectors
reducedX, recoveredX, pca = reduceDimension(X)
reduced = reducedX
print(reducedX[2].shape)

#create model
neigh = KNeighborsClassifier(n_neighbors=10, metric="l1")
neigh.fit(reducedX, Y)

#initiate class for video recording
cap = cv2.VideoCapture(0)

width = 28
height = 28
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 1)
detector = HandDetector(detectionCon=1, maxHands=1)

while True:
    # Get image frame
    success, img = cap.read()   
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frame rate: ", int(fps), "FPS")  
    #resize of image in order to reduce later
    result = img[:, :, 0]
    small_img= cv2.resize(img, (28, 28)) 
    print(small_img.shape)
    #Pass to grayscale
    result = small_img[:, :, 0]
    vector = np.array(result.reshape(1,28*28))
    print(vector.shape)
   #vector reduction
    vector = pca.transform(vector)
    print("vector after",vector.shape)
    predicted = neigh.predict(vector)[0]
    #prediction
    print(predicted)
    # Display
    cv2.imshow("Image Reduce", result)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

