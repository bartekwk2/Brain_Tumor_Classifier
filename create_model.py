import keras as keras
import keras.layers as layers

def createModel():
    
    model = keras.Sequential()
    
    model.add(layers.Conv2D(4,(5,5),activation='relu', input_shape=(224,224,1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4,activation='softmax'))

    model.summary()
    
    return model
    