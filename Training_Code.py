import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
#from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
 
 
# Parameters
path = "C:/Users/Avinash/Desktop/myData"  # folder with all the class folders
labelFile = "C:/Users/Avinash/Desktop/labels.csv"  # file with all names of classes
batch_size_val = 50  # how many to process together
steps_per_epoch_val = 2000
epochs_val = 15
imageDimesions = (32, 32, 3)
testRatio = 1  # if 1000 images split will 200 for testing
validationRatio = 1  # if 1000 images 20% of remaining 800 will be 160 for validation
selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # List of class indices you want to consider


 
 
# Importing of the Images
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))


print("Importing Classes.....")
for x in selected_classes:  # Only consider selected classes
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")
print(" ")
images = np.array(images)
classNo = np.array(classNo)
 
# Split Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
noOfClasses = 12
steps_per_epoch_val = len(X_train)//batch_size_val
# X_train = ARRAY OF IMAGES TO TRAIN
# y_train = CORRESPONDING CLASS ID
 
# TO CHECK IF NUMBER OF IMAGES MATCHES TO NUMBER OF LABELS FOR EACH DATA SET
print("Data Shapes")
print("Train",end = "");print(X_train.shape,y_train.shape)
print("Validation",end = "");print(X_validation.shape,y_validation.shape)
print("Test",end = "");print(X_test.shape,y_test.shape)
assert(X_train.shape[0]==y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
assert(X_validation.shape[0]==y_validation.shape[0]), "The number of images in not equal to the number of lables in validation set"
assert(X_test.shape[0]==y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert(X_train.shape[1:]==(imageDimesions))," The dimesions of the Training images are wrong "
assert(X_validation.shape[1:]==(imageDimesions))," The dimesionas of the Validation images are wrong "
assert(X_test.shape[1:]==(imageDimesions))," The dimesionas of the Test images are wrong"
 
 
# READ CSV FILE
data=pd.read_csv(labelFile)
print("data shape ",data.shape,type(data))
 
# DISPLAY SOME SAMPLES IMAGES  OF ALL THE CLASSES
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()

for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]

        if len(x_selected) == 0:
            continue  # Skip empty classes

        random_image_index = random.randint(0, len(x_selected) - 1)
        axs[j][i].imshow(x_selected[random_image_index, :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))
 
 

 
# PREPROCESSING THE IMAGES
 
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)     # CONVERT TO GRAYSCALE
    img = equalize(img)      # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img
 
X_train=np.array(list(map(preprocessing,X_train)))  # TO IRETATE AND PREPROCESS ALL IMAGES
X_validation=np.array(list(map(preprocessing,X_validation)))
X_test=np.array(list(map(preprocessing,X_test)))
#cv2.imshow("GrayScale Images",X_train[random.randint(0,len(X_train)-1)]) # TO CHECK IF THE TRAINING IS DONE PROPERLY
 
# ADD A DEPTH OF 1
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
 
 
# AUGMENTATAION OF IMAGES: TO MAKEIT MORE GENERIC
dataGen= ImageDataGenerator(width_shift_range=0.1,   # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                            height_shift_range=0.1,
                            zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                            shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                            rotation_range=10)  # DEGREES
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20)  # REQUESTING DATA GENRATOR TO GENERATE IMAGES  BATCH SIZE = NO. OF IMAGES CREAED EACH TIME ITS CALLED
X_batch,y_batch = next(batches)
 
# TO SHOW AGMENTED IMAGE SAMPLES
fig,axs=plt.subplots(1,15,figsize=(20,5))
fig.tight_layout()
 
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0],imageDimesions[1]))
    axs[i].axis('off')
plt.show()
 
 
y_train = to_categorical(y_train,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
 
# CONVOLUTION NEURAL NETWORK MODEL
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5) 
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500
    
    # Creating a sequential model (a linear stack of layers)
    model = Sequential()
    
    # Adding the first convolutional layer
    #Convolutional Layers: These layers scan the input image using small filters to detect patterns and features. 
    # Think of them as looking for simple shapes like edges or curves.
    model.add(Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu'))
    
    # Adding another convolutional layer
    
    model.add(Conv2D(no_Of_Filters, size_of_Filter, activation='relu'))
    
    # Adding a max pooling layer to reduce the spatial dimensions of the output
    #Max Pooling Layers: These layers reduce the size of the image and focus on the most important information, like zooming out.
    model.add(MaxPooling2D(pool_size=size_of_pool))
    
    # Adding more convolutional layers, but with smaller filter size
    
    model.add(Conv2D(no_Of_Filters//2, size_of_Filter2, activation='relu'))
    model.add(Conv2D(no_Of_Filters//2, size_of_Filter2, activation='relu'))
    
    # Another max pooling layer and dropout layer to prevent overfitting
    model.add(MaxPooling2D(pool_size=size_of_pool))
    #Dropout Layers: These layers help prevent the model from memorizing the training data and make it more robust.
    model.add(Dropout(0.5))
    
    # Flattening the output to a one-dimensional array
    #Flatten Layer: This layer takes the output from previous layers and turns it into a one-dimensional array,
    #getting it ready for the fully connected layers.
    
    model.add(Flatten())
    
    # Adding a fully connected (dense) layer with ReLU activation
    #Fully Connected Layers: These layers take the flattened array and try to understand the overall structure of the image.
    # They contain nodes that learn different aspects of the data.
    model.add(Dense(no_Of_Nodes, activation='relu'))
    
    # Another dropout layer
    model.add(Dropout(0.5))
    
    # Adding the output layer with softmax activation for multi-class classification
    #Output Layer: This layer produces the final results. In this case, it uses softmax activation for multi-class classification.
    model.add(Dense(noOfClasses, activation='softmax'))
    
    # Compiling the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
    #Compilation: The model is configured for training with the Adam optimizer, 
    # categorical crossentropy loss (useful for classification problems), and accuracy as the metric to measure performance.
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
 
# TRAIN
model = myModel()
print(model.summary())
history=model.fit(dataGen.flow(X_train,y_train,batch_size=batch_size_val),steps_per_epoch=steps_per_epoch_val,epochs=epochs_val,validation_data=(X_validation,y_validation),shuffle=1)
 
# PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score =model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])
 
 
# STORE THE MODEL AS A PICKLE OBJECT
pickle_out= open("model_trainedxx.p","wb")  # wb = WRITE BYTE
pickle.dump(model,pickle_out)
pickle_out.close()
#cv2.waitKey(1)
#Test Code

 
