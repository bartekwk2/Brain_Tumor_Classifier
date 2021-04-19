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
from tumor import Tumor
from sklearn.metrics import classification_report
from image_preprocessing import loadImages,crop_image,fileToClass,divideImages



Images = loadImages("Images")
glioma = fileToClass(Images["glioma"],0)
meningioma =fileToClass(Images["meningioma"],1)
no = fileToClass(Images["no"],2)
pituitary = fileToClass(Images["pituitary"],3)

GliomaTumorImages,testGliomaTumorImages = divideImages(0.2,glioma)
MeningiomaTumorImages,testMeningiomaTumorImages = divideImages(0.2,meningioma)
NoTumorImages,testNoTumorImages = divideImages(0.2,no)
PituitaryTumorImages,testPituitaryTumorImages = divideImages(0.2,pituitary)

AllImagesTEST = []
AllImagesTEST.extend(testGliomaTumorImages)
AllImagesTEST.extend(testMeningiomaTumorImages)
AllImagesTEST.extend(testNoTumorImages)
AllImagesTEST.extend(testPituitaryTumorImages)

AllImagesTRAIN = []
AllImagesTRAIN.extend(GliomaTumorImages)
AllImagesTRAIN.extend(MeningiomaTumorImages)
AllImagesTRAIN.extend(NoTumorImages)
AllImagesTRAIN.extend(PituitaryTumorImages)

random.shuffle(AllImagesTEST)
random.shuffle(AllImagesTRAIN)


AllImages = []
AllImagesTest = []

label_test = []
label = []


for cancer in(AllImagesTRAIN):
    AllImages.append(cancer.tumor)
    label.append(cancer.tumorType)
    
for cancer in(AllImagesTEST):
    AllImagesTest.append(cancer.tumor)
    label_test.append(cancer.tumorType)
    

model=createModel()

train = AllImages
train= array(train).reshape(array(train).shape[0],224,224,1)
train = np.array(train).astype('float32')

test = AllImagesTest
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
                    validation_split=0.03,
                    shuffle=True,
                    batch_size=32, epochs=10)



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val'], loc='upper left')
plt.show()


test_loss, test_acc = model.evaluate(test, label_test)
print('Test accuracy:', test_acc)

categorical_test_labels = pd.DataFrame(label_test).idxmax(axis=1)
preds = np.round(model.predict(test),0) 
categorical_preds = pd.DataFrame(preds).idxmax(axis=1)
Tumors = ['No', 'Glioma', 'Meningioma', 'Pituitary']
classification_metrics = metrics.classification_report(label_test, preds, target_names=Tumors )
print(classification_metrics)
confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)
plot_confusion_matrix(confusion_matrix, Tumors)

model.save('tumorClassifier.h5')



