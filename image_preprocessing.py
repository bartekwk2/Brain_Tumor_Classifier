
import imutils
import cv2
import os
from skimage.color import rgb2gray
import random
import numpy as np
from tumor import Tumor
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from PIL import Image
import shutil


def augmentAllImages(foldername,datasetIn,datasetOut):

    counter = 0
    for root, _, _ in os.walk(foldername):  
        if counter> 0 :
            pathName = os.path.basename(root)
            augmentImagesInFolder(datasetIn+pathName+"/",datasetOut+pathName+"/",4)
        counter+=1


def augmentImagesInFolder(image_directory, save_dir, nr_of_copies):

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant')

    dataset =[]
    shutil.rmtree(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    my_images = os.listdir(image_directory)
    for i, image_name in enumerate(my_images):
        if (image_name.split('.')[1]== 'jpg'):
            image = io.imread(image_directory + image_name)
            image = Image.fromarray(image)
            image = image.resize((224,224))
            dataset.append(np.array(image))

    x = np.array(dataset)

    i = 0
    for _ in datagen.flow(x, batch_size=len(x), save_to_dir=save_dir, save_prefix='aug', save_format='jpg'):
        i += 1
        if i > nr_of_copies-1:
            break

def divideImages(percent,typeOfCancer):
    test = []
    rangeType = int(len(typeOfCancer)*percent)

    for i in range(rangeType):
        move = random.randrange(len(typeOfCancer)-i)
        element = typeOfCancer[move]
        typeOfCancer = np.delete(typeOfCancer,move,0)
        test.append(element)

    return typeOfCancer,test

def getRandomLists(data,numberOfImages):
    testing = []
    training = []
    #rangeType = int(len(data)*percent)



    for i in range(numberOfImages):
        move = random.randrange(numberOfImages-i)
        element = data[move]
        data = np.delete(data,move,0)
        testing.append(element)

    training = data

    return testing,training


def fileToClass(files,typeOfCancer):
    cancerClass = []
    for cancer in files:
        cancerClass.append(Tumor(cancer,typeOfCancer))
    return cancerClass

def getTumorsList(glioma,meningioma,pituary,no):
    tumors = []

    tumors.extend(glioma)
    tumors.extend(meningioma)
    tumors.extend(pituary)
    tumors.extend(no)

    return tumors

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