# import the necessary packages
import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
from preprocessing import foreground,cropping
from feature_extraction import *

# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=9)))

# variables to hold the results and names
results = []
names = []
scoring = "accuracy"
predictarr = []

# import the feature vector and trained labels
h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print ("[STATUS] features shape: {}".format(global_features.shape))
print ("[STATUS] labels shape: {}".format(global_labels.shape))

print ("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print ("[STATUS] splitted train and test data...")
print ("Train data  : {}".format(trainDataGlobal.shape))
print ("Test data   : {}".format(testDataGlobal.shape))
print ("Train labels: {}".format(trainLabelsGlobal.shape))
print ("Test labels : {}".format(testLabelsGlobal.shape))
print("\n")
# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

# 10-fold cross validation
'''for name, model in models:
    kfold = KFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)'''


import matplotlib.pyplot as plt

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=40, random_state=18)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

# path to test data
test_path = "../\Image_Classification/test"
fixed_size = tuple((500, 500))

Expected = [2,2,2,2,2,1,1,1,1,1,7,7,7,7,7,3,3,3,3,3,4,4,4,4,4,6,6,6,6,6,5,5,5,5,5]


# loop through the test images
test=36
print("Initiated Testing....")
for x in range(1,test):
    
    image_path = "test/" + str(x) + ".jpg"
    # load the image
    img = cv2.imread(image_path)
   
    # resize the image
    img = cv2.resize(img, fixed_size)
    
    output=foreground(img)
    image,length,breadth= cropping(img,output)
    #print("Cropped: " + str(x) + ".jpg")

    fv_shape = shape(image)
    fv_texture   = texture(image)
    fv_color  = color(image)
    
    #fv_margin = margin(image)


    global_feature = np.hstack([fv_shape,fv_texture,fv_color])

    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    predictor=prediction+1
    predictarr.append(predictor)
    
    a=train_labels[prediction]
    
    # show predicted label on image
    cv2.putText(image, a, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)
    
    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    print(str(x) + ".jpg")
    print("Prediction: " + str(prediction))
    
    
results = confusion_matrix(Expected, predictarr)
print(results)
plt.clf()
plt.imshow(results, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Leaf_1','Leaf_2','Leaf_3','Leaf_4','Leaf_5','Unknown','Leaf_7']
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
plt.show()

count=0