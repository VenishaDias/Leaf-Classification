from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
from preprocessing import foreground,cropping
import time

start = time.time()
# fixed-sizes for image
fixed_size = tuple((500, 500))

# path to training data
train_path = "../\Image_Classification/train"

# no.of.trees for Random Forests
num_trees = 10

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.20

# seed for reproducing same results
seed = 9




# Feature-Descriptor to find 'Shape'
def shape(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Feature-Descriptor to find 'Texture'
def texture(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# Feature-Descriptor to find 'Color'
def color(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# Feature-Descriptor to find 'Margin'
def margin(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,16)
    
    return edged

##def venation(image):
#def SIFT(image):
#	orb = cv2.ORB_create()
#	kp1, des1 = orb.detectAndCompute(image, None)
#	des1=des1.flatten()
#	print (des1.shape)
#	return des1
	
    

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

# num of images per class
images_per_class = 50


for training_name in train_labels:
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name
    
    k = 1
    # loop over the images in each sub-folder
    for x in range(1,images_per_class+1):
        # get the image file name
        file = dir + "/" + str(x) + ".jpg"
        
        # read the image and resize it to a fixed-size
        img = cv2.imread(file)
        img = cv2.resize(img, fixed_size)
        
        output = foreground(img)
        image, length, breadth= cropping(img,output)
        print("PRE-PROCESSED: " + str(x)+".jpg")
        # Global Feature extraction
        fv_shape = shape(image)
        fv_texture = texture(image)
        fv_color  = color(image)
        #fv_margin = margin(image)
        #fv_SIFT = SIFT(image)
		

       
        # Concatenate global features
        global_feature = np.hstack([fv_shape,fv_texture,fv_color])

         # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

        i += 1
        k += 1
    print ("Processed folder: {}".format(current_label))
    j += 1

print ("Completed Global Feature Extraction...")
print(global_feature)

# get the overall feature vector size
print ("Feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print ("Training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print ("Training labels encoded...")

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print ("Feature vector normalized...")

print ("Target labels: {}".format(target))
print ("Target labels shape: {}".format(target.shape))
# save the feature vector using HDF5
h5f_data = h5py.File('output/data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('output/labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print ("End of training..")

end = time.time()
print(end - start , "seconds")
