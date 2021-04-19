
import imutils
import cv2
import os
from skimage.color import rgb2gray
import random
import numpy as np
from tumor import Tumor

def divideImages(percent,typeOfCancer):
    test = []
    rangeType = int(len(typeOfCancer)*percent)

    for i in range(rangeType):
        move = random.randrange(len(typeOfCancer)-i)
        element = typeOfCancer[move]
        typeOfCancer = np.delete(typeOfCancer,move,0)
        test.append(element)

    return typeOfCancer,test


def fileToClass(files,typeOfCancer):
    cancerClass = []
    for cancer in files:
        cancerClass.append(Tumor(cancer,typeOfCancer))
    return cancerClass

def crop_image(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
   
    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # add contour on the image
    img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)
    
    # add extreme points
    img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
    img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
    img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)
    
    # crop
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    
    return new_img


def loadImages(foldername):

    image_dict={}
    
    for root, _, files in os.walk(foldername):  
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        images=[]
        image_dict[os.path.basename(root)]=images
        
        for file in files:
            
            img_data =cv2.imread(root+"/"+ file)
            imageCropped=crop_image(img_data)
            imageResized=cv2.resize(imageCropped,(224,224),interpolation=cv2.INTER_CUBIC)    
            imageGrey=rgb2gray(imageResized)
            images.append(imageGrey)
            
    return image_dict