# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 20:31:13 2022

@author: shawon
"""


import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D , BatchNormalization, Dropout,Convolution2D,ZeroPadding2D
#from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16


os.getcwd()
data_path = "CNN_RF/data/"

SIZE = 128

dim1 = []
dim2 = []
images = []
labels = [] 
label_name = []
number_of_samples = []
for directory_path in glob.glob(data_path+"/train/*"):
    label = directory_path.split("\\")[-1]
    label_name.append(label)
    number_of_samples.append(len(os.listdir(directory_path)))
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)     
        dim1.append(img.shape[0])
        dim2.append(img.shape[1])
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
        labels.append(label)
        
images = np.array(images)
labels = np.array(labels)
print(number_of_samples)
sns.jointplot(dim1,dim2)
plt.show()


plt.figure(figsize=(12, 8))
plt.bar(label_name, number_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class name")
plt.ylabel("Number of images")
plt.show()


plt.figure(1 , figsize = (100 , 50))
plt.suptitle("Some sample images",fontsize=20)


n = 0 
for i in range(24):
    color = "black"
    n += 1 
    types = ""
    r = np.random.randint(3119)
    img = images[r]
    plt.subplot(4 , 6 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    plt.imshow(img)
    plt.xlabel(labels[r],color=color,fontsize=10)
    plt.xticks([]) , plt.yticks([])
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_test)
test_labels_encoded = le.transform(y_test)
le.fit(y_train)
train_labels_encoded = le.transform(y_train)


X_train, X_test = X_train / 255.0, X_test / 255.0

from tensorflow.keras.utils import to_categorical
y_train_one_hot = to_categorical(train_labels_encoded)
y_test_one_hot = to_categorical(test_labels_encoded)

from keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D,BatchNormalization

activation = 'relu'
feature_extractor = Sequential()


SIZE = 128

feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (SIZE, SIZE, 3)))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Flatten())

#Add layers for deep learning prediction
x = feature_extractor.output  
x = Dense(128, activation = activation, kernel_initializer = 'he_uniform')(x)
x = Dropout(.5)(x)
prediction_layer = Dense(5, activation = 'softmax')(x)

# Make a new model combining both feature extractor and x
cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
cnn_model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(cnn_model.summary()) 
print(feature_extractor.summary())

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
early_stop = EarlyStopping(monitor='val_loss',patience=2)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
##########################################
#Train the CNN model
import time
start = time.time()
history = cnn_model.fit(X_train, y_train_one_hot,
                        batch_size = 16, 
                        epochs=50, validation_data = (X_test, y_test_one_hot),callbacks=[early_stop])
stop = time.time()

print("Training time: "+str(stop-start)+"s")

cnn_model.save('model_output/VGG16_pretrained_modelALLNew.h5')

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(cnn_model, X_test, y_test, cv=5)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
start = time.time()
prediction_NN = cnn_model.predict(X_test)
stop = time.time()
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)
conf_matrix = confusion_matrix(y_test, prediction_NN)
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
 

def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax,cmap=plt.cm.Blues,linewidths=.5)
    #plt.savefig(filename)
    plt.show()


cm_analysis(y_test, prediction_NN, label_name, ymap=None, figsize=(10,10))


start = time.time()
X_for_RF = feature_extractor.predict(X_train) #This is out X input to RF
stop = time.time()

print("Feature extraction time: "+str(stop-start))
#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 1000, random_state = 42)

# Train the model on training data
start = time.time()
rf_history =  RF_model.fit(X_for_RF, y_train) #For sklearn no one hot encoding
stop = time.time()
print("RF model train time: "+str(stop-start))


#Send test data through same feature extractor process
start = time.time()
X_test_feature = feature_extractor.predict(X_test)
stop = time.time()
print("RF model test time: "+str(stop-start))

#Now predict using the trained RF model. 
prediction_RF = RF_model.predict(X_test_feature)
#Inverse le transform to get original label back. 
#prediction_RF = le.inverse_transform(prediction_RF)

#Print overall accuracy
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_RF))

#Confusion Matrix - verify accuracy of each class
cm_analysis(y_test, prediction_RF, label_name, ymap=None, figsize=(10,10))


#Check results on a few select images
#n=5 #dog park. RF works better than CNN
n=500 #Select the index of image to be loaded for testing
img = X_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_features=feature_extractor.predict(input_img)
prediction_RF = RF_model.predict(input_img_features)[0] 
#prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", y_test[n])

from sklearn.metrics import precision_score, recall_score
precision_score(y_train, prediction_RF,average=None)

recall_score(y_train, prediction_RF,average=None)

from sklearn.metrics import f1_score

f1_score(y_test, prediction_RF,average=None)
ax = plt.gca()

rf_disp = RocCurveDisplay.from_estimator(RF_model, X_test, y_test, ax=ax)
plt.show()

from sklearn.model_selection import cross_val_score

cross_val_score(RF_model, X_test_feature, y_test, cv=10, scoring="accuracy")


from sklearn.metrics import roc_curve, roc_auc_score

y_prob_pred_cnb = RF_model.predict_proba(X_test_feature)
y_prob_pred_cnb
roc_auc_score(y_test, y_prob_pred_cnb, multi_class='ovr', average='weighted')

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
y = label_binarize(y_test, classes=[0, 1, 2,3,4])
n_classes = 5

fpr = {}
tpr = {}
thresh ={}
roc_auc = dict()
n_class = 5
for i in range (n_class):
   fpr[i], tpr[i], thresh[i] = roc_curve(test_labels_encoded, y_prob_pred_cnb[:,i], pos_label=i)
   roc_auc[i] = auc(fpr[i], tpr[i])
   # plotting
   plt.plot(fpr[i], tpr[i], linestyle='--',
           label='%s vs Rest (AUC=%0.2f)'%(label_name[i],roc_auc[i]))
   
plt.plot([0,1],[0,1],'b--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.title('Multiclass ROC curve')
plt.xlabel ('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend (loc='lower right')
plt.show()

########################################################
#VGG16+SVG

X_for_SV = X_for_RF #This is our X input to RF

#SVG
from sklearn.svm import LinearSVC
svg = LinearSVC()
# Train the model on training data
start = time.time()
svg.fit(X_for_SV, y_train) #For sklearn no one hot encoding
stop = time.time()
print("SVC model fit time: "+str(stop-start)+"s")
X_for_SV.shape

X_test.shape
start = time.time()
X_test_feature = feature_extractor.predict(X_test)
stop = time.time()

print("SVM model test feature extract time: "+str(stop-start))

X_test_feature.shape
#Send test data through same feature extractor process
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)
#Now predict using the trained RF model.
prediction_SVG = svg.predict(X_test_features)
#Inverse le transform to get original label back. 
prediction_SVG = le.inverse_transform(prediction_SVG)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_SVG))
cross_val_score(svg, X_test_feature, y_test, cv=10, scoring="accuracy")

cm_analysis(y_test, prediction_SVG, label_name, ymap=None, figsize=(10,10))


#####################
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
start = time.time()
clf.fit(X_for_RF,y_train)
stop = time.time()
print(stop-start)
prediction_DT = clf.predict(X_test_features)
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_DT))
cm_analysis(y_test, prediction_DT, label_name, ymap=None, figsize=(10,10))
cross_val_score(clf, X_test_feature, y_test, cv=10, scoring="accuracy")

###############################
X_train_norm = X_train.reshape(2495,49152)
X_test_norm = X_test.reshape(624,49152)

clf = DecisionTreeClassifier(
    max_depth = 5,
    random_state=42)
start = time.time()
clf.fit(X_train_norm,y_train)
stop = time.time()
stop-start
prediction_DT = clf.predict(X_test_norm)
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_DT))
cm_analysis(y_test, prediction_DT, label_name, ymap=None, figsize=(10,10))
