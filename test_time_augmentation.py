import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from keras import models as models
from numpy import array
import numpy as np
import os
from skimage.color import rgb2gray
import scipy as sp     
from PIL import Image
from scipy import stats
import os

from image_preprocessing import loadImages,crop_image

def predictClasses(inputImage):
    tumors = [0,1,2,3]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    images = []
    images.append(inputImage)
    images= array(images).reshape(array(images).shape[0],224,224,1)
    images = np.array(images).astype('float32')
    img = images[0]
    image = np.expand_dims(img, axis=0)
    model = models.load_model('tumorClassifier.h5')
    prediction = model.predict(image)
    gliomaResult = round(prediction[0][0],3)
    meningiomaResult = round(prediction[0][1],3)
    pituaryResult = round(prediction[0][2],3)
    noTumorResult = round(prediction[0][3],3)
    preds = [gliomaResult,meningiomaResult,pituaryResult,noTumorResult]
    ind = np.argmax(preds)
    return preds,tumors[ind]

def flip_lr(img):
    return np.flip(img, axis=1)

def shift(images, shift, axis):
    return np.roll(images, shift, axis=axis)

def rotate(images, angle):
    return sp.ndimage.rotate(
        images, angle)

def hardVoting(images,knownClass):
    tumorHits=0
    for img in images:
        fliped = flip_lr(img)
        shifted = shift(img,-3, axis=0)
        shifted2 = shift(img,-5, axis=1)
        shifted3 = shift(img,-1, axis=0)
        classes = []
        _,c = predictClasses(img)
        _,cF = predictClasses(fliped)
        _,cS = predictClasses(shifted)
        _,cS2 = predictClasses(shifted2)
        _,cS3 = predictClasses(shifted3)
        classes.append(c)
        classes.append(cF)
        classes.append(cS)
        classes.append(cS2)
        classes.append(cS3)
        modeResult = stats.mode(classes)
        mode = modeResult.mode[0]
        if mode == knownClass:
            tumorHits+=1
    acc = tumorHits/len(images)
    return acc


def softVoting(images,knownClass):
    imageHitForInd = 0
    percentegSum = 0
    for img in images:
        fliped = flip_lr(img)
        shifted = shift(img,-3, axis=0)
        shifted2 = shift(img,-5, axis=1)
        shifted3 = shift(img,-1, axis=0)
        p,_ = predictClasses(img)
        pF,_ = predictClasses(fliped)
        pS,_ = predictClasses(shifted)
        pS2,_ = predictClasses(shifted2)
        pS3,_ = predictClasses(shifted3)
        predictions = [p,pF,pS,pS2,pS3]
        predictions = np.array(predictions)
        c1 = np.array(predictions[:,0])
        c2 = np.array(predictions[:,1])
        c3 = np.array(predictions[:,2])
        c4 = np.array(predictions[:,3])
        meanC1 = np.mean(c1)
        meanC2 = np.mean(c2)
        meanC3 = np.mean(c3)
        meanC4 = np.mean(c4)
        mean = np.array([meanC1,meanC2,meanC3,meanC4])
        maxInd = np.argmax(mean)
        if maxInd == knownClass:
            imageHitForInd+=1    
            percentegSum += mean[maxInd]
        meanPcc = percentegSum/len(images)
        meanInd = imageHitForInd/len(images)
    return meanPcc, meanInd


directory = os.getcwd()
folder=loadImages(directory+"/ImagesOther/Few")
glioma=folder["test_glioma"]
meningioma=folder["test_meningioma"]
pituitary=folder["test_pituitary"]
no=folder["test_no"]


glioma_acc_pcc,glioma_acc_ind = softVoting(images=glioma,knownClass=0)
meningioma_acc_pcc,meningioma_acc_ind = softVoting(images=meningioma,knownClass=1)
pituitary_acc_pcc,pituitary_acc_ind = softVoting(images=pituitary,knownClass=2)
no_acc_pcc,no_acc_ind = softVoting(images=no,knownClass=3)
print(glioma_acc_pcc, meningioma_acc_pcc, pituitary_acc_pcc,no_acc_pcc)
print(glioma_acc_ind, meningioma_acc_ind, pituitary_acc_ind,no_acc_ind)


glioma_acc_hard = hardVoting(images=glioma,knownClass=0)
meningioma_acc_hard = hardVoting(images=meningioma,knownClass=1)
pituitary_acc_hard = hardVoting(images=pituitary,knownClass=2)
no_acc_hard= hardVoting(images=no,knownClass=3)
print(glioma_acc_hard, meningioma_acc_hard, pituitary_acc_hard,no_acc_hard)

