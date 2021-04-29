from create_model import createModel
from plot_matrix import plot_confusion_matrix
import numpy as np
from numpy import array
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import random
from keras import models as models
import os
from tumor import Tumor
from sklearn.metrics import classification_report
from image_preprocessing import loadImages,crop_image,fileToClass,divideImages,augmentAllImages,getRandomLists,getTumorsList
from test_time_augmentation import predictClasses,hardVoting,softVoting


def learnModelWithDataset(glioma,meningioma,pituitary,no,testing,training,model):
    random.shuffle(training)
    random.shuffle(testing)

    train = []
    label = []
    for element in training:
        train.append(element.tumor)
        label.append(element.tumorType)

    test = []
    label_test = []
    for element in testing:
        test.append(element.tumor)
        label_test.append(element.tumorType)

    train= array(train).reshape(array(train).shape[0],224,224,1)
    train = np.array(train).astype('float32')

    test= array(test).reshape(array(test).shape[0],224,224,1)
    test = array(test).astype('float32')

    label = np.array(label).astype('float32')
    label = np_utils.to_categorical(label,4)

    label_test = np.array(label_test).astype('float32')
    label_test = np_utils.to_categorical(label_test,4)


    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])



    history = model.fit(train, label,
                        batch_size=32, epochs=10)

    model.save('tumorClassifier.h5')

# plt.plot(history.history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['loss', 'val'], loc='upper left')
# plt.show()

    _, test_acc = model.evaluate(test, label_test)
    print('Test accuracy without voting:', test_acc)

    mod = models.load_model('tumorClassifier.h5')

    glioma_acc_ind = hardVoting(glioma,0, mod)
    meningioma_acc_ind = hardVoting(meningioma,1, mod)
    pituitary_acc_ind = hardVoting(pituitary,2, mod)
    no_acc_ind = hardVoting(no,3, mod)
    hard_acc = (glioma_acc_ind+meningioma_acc_ind+pituitary_acc_ind+no_acc_ind)/4
    print('Test accuracy with hard voting:', hard_acc)

    _,soft_glioma_acc_ind = softVoting(glioma,0, mod)
    _,soft_meningioma_acc_ind = softVoting(meningioma,1, mod)
    _,soft_pituitary_acc_ind = softVoting(pituitary,2, mod)
    _,soft_no_acc_ind = softVoting(no,3, mod)
    soft_acc = (soft_glioma_acc_ind+soft_meningioma_acc_ind+soft_pituitary_acc_ind+soft_no_acc_ind)/4
    print('Test accuracy with soft voting:', soft_acc)


# categorical_test_labels = pd.DataFrame(label_test).idxmax(axis=1)
# preds = np.round(model.predict(test),0) 
# categorical_preds = pd.DataFrame(preds).idxmax(axis=1)

# Tumors = ['No', 'Glioma', 'Meningioma', 'Pituitary']
# classification_metrics = metrics.classification_report(label_test, preds, target_names=Tumors )
# print(classification_metrics)

# confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)

# plot_confusion_matrix(confusion_matrix, Tumors)


# LOAD FILES
loadedImages=loadImages("ImagesOut")
gliomasTogether=loadedImages['glioma']
meningiomaTogether=loadedImages['meningioma']
pituaryTogether=loadedImages['pituitary']
noTumorTogether=loadedImages['no']

gliomaObject = fileToClass(gliomasTogether,0)
meningiomaObject = fileToClass(meningiomaTogether,1)
pituaryObject = fileToClass(pituaryTogether,2)
noObject = fileToClass(noTumorTogether,3)

numOfImages = 200

# Getting group 1
g1Glioma, restGliomag1 = getRandomLists(gliomaObject,numberOfImages=numOfImages)
g1Meningioma, restMeningiomag1 = getRandomLists(meningiomaObject,numberOfImages=numOfImages)
g1Pituary, restPituaryg1 = getRandomLists(pituaryObject,numberOfImages=numOfImages)
g1No, restNog1 = getRandomLists(noObject,numberOfImages=numOfImages)
group1 = getTumorsList(glioma=g1Glioma,meningioma=g1Meningioma,pituary=g1Pituary,no=g1No)

# Getting group 2
g2Glioma, restGliomag2 = getRandomLists(restGliomag1,numberOfImages=numOfImages)
g2Meningioma, restMeningiomag2 = getRandomLists(restMeningiomag1,numberOfImages=numOfImages)
g2Pituary, restPituaryg2 = getRandomLists(restPituaryg1,numberOfImages=numOfImages)
g2No, restNog2 = getRandomLists(restNog1,numberOfImages=numOfImages)
group2 = getTumorsList(glioma=g2Glioma,meningioma=g2Meningioma,pituary=g2Pituary,no=g2No)

# Getting group 3
g3Glioma, restGliomag3 = getRandomLists(restGliomag2,numberOfImages=numOfImages)
g3Meningioma, restMeningiomag3 = getRandomLists(restMeningiomag2,numberOfImages=numOfImages)
g3Pituary, restPituaryg3 = getRandomLists(restPituaryg2,numberOfImages=numOfImages)
g3No, restNog3 = getRandomLists(restNog2,numberOfImages=numOfImages)
group3 = getTumorsList(glioma=g3Glioma,meningioma=g3Meningioma,pituary=g3Pituary,no=g3No)

# Getting group 4
g4Glioma, restGliomag4 = getRandomLists(restGliomag3,numberOfImages=numOfImages)
g4Meningioma, restMeningiomag4 = getRandomLists(restMeningiomag3,numberOfImages=numOfImages)
g4Pituary, restPituaryg4 = getRandomLists(restPituaryg3,numberOfImages=numOfImages)
g4No, restNog4 = getRandomLists(restNog3,numberOfImages=numOfImages)
group4 = getTumorsList(glioma=g4Glioma,meningioma=g4Meningioma,pituary=g4Pituary,no=g4No)

# Getting group 5
g5Glioma, restGliomag5 = getRandomLists(restGliomag4,numberOfImages=numOfImages)
g5Meningioma, restMeningiomag5 = getRandomLists(restMeningiomag4,numberOfImages=numOfImages)
g5Pituary, restPituaryg5 = getRandomLists(restPituaryg4,numberOfImages=numOfImages)
g5No, restNog5 = getRandomLists(restNog4,numberOfImages=numOfImages)
group5 = getTumorsList(glioma=g5Glioma,meningioma=g5Meningioma,pituary=g5Pituary,no=g5No)


# Tworzenie modelu
model=createModel()

g1Training = []
g1Training.extend(group2)
g1Training.extend(group3)
g1Training.extend(group4)
g1Training.extend(group5)
learnModelWithDataset(glioma=g1Glioma,meningioma=g1Meningioma,pituitary=g1Pituary,no=g1No,testing=group1,training=g1Training,model=model)
print('FIRST GROUP RESULTS')

model=createModel()

g2Training = []
g2Training.extend(group1)
g2Training.extend(group3)
g2Training.extend(group4)
g2Training.extend(group5)
learnModelWithDataset(glioma=g2Glioma,meningioma=g2Meningioma,pituitary=g2Pituary,no=g2No,testing=group2,training=g2Training,model=model)
print('SECOND GROUP RESULTS')

model=createModel()

g3Training = []
g3Training.extend(group1)
g3Training.extend(group2)
g3Training.extend(group4)
g3Training.extend(group5)
learnModelWithDataset(glioma=g3Glioma,meningioma=g3Meningioma,pituitary=g3Pituary,no=g3No,testing=group3,training=g3Training,model=model)
print('THIRD GROUP RESULTS')

model=createModel()

g4Training = []
g4Training.extend(group1)
g4Training.extend(group3)
g4Training.extend(group2)
g4Training.extend(group5)
learnModelWithDataset(glioma=g4Glioma,meningioma=g4Meningioma,pituitary=g4Pituary,no=g4No,testing=group4,training=g4Training,model=model)
print('FOURTH GROUP RESULTS')

model=createModel()

g5Training = []
g5Training.extend(group1)
g5Training.extend(group3)
g5Training.extend(group4)
g5Training.extend(group2)
learnModelWithDataset(glioma=g5Glioma,meningioma=g5Meningioma,pituitary=g5Pituary,no=g5No,testing=group5,training=g5Training,model=model)
print('FIFTH GROUP RESULTS')

