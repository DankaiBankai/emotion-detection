
import numpy as np  
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau



class NeuralNetwork:
    def __init__(self):
        #Input/output arrays
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []


    def read_fer2013(self):
        print("------------Begin data read-------------")
        df=pd.read_csv('fer2013.csv')
        
        #Load training and test data
        for index, row in df.iterrows():
            if 'Training' in row['Usage']:       
                val = row['pixels'].split(" ")
                self.x_train.append(np.array(val, 'float32'))
                self.y_train.append(row['emotion'])
        
            elif 'PublicTest' in row['Usage']:
                val = row['pixels'].split(" ")
                self.x_test.append(np.array(val, 'float32'))
                self.y_test.append(row['emotion'])

        #Format data
        self.x_train = np.array(self.x_train, 'float32')
        self.x_test = np.array(self.x_test, 'float32')

        self.y_train = np.array(self.y_train, 'float32')
        self.y_test = np.array(self.y_test, 'float32')

        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes = 7)      
        self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes = 7)

        #Normalize data between 0 and 1
        self.x_train -= np.mean(self.x_train, axis = 0)
        self.x_train /= np.std(self.x_train, axis = 0)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 48, 48, 1)

        self.x_test -= np.mean(self.x_test, axis = 0)
        self.x_test /= np.std(self.x_test, axis = 0)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 48, 48, 1)
        print("------------End data read------------")

    
    def create_model(self):
        print("------------Begin model create-------------")
        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48,48,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(1024, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(1024))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(1024))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(7))
        model.add(Activation('softmax'))

        #Compile with loss and optimizer  
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("------------End model create-------------")
        return model



def main():
    nn = NeuralNetwork()
    nn.read_fer2013()
    model = nn.create_model()

    #Data augmentation. Randomly apply transformations to input images. Allows for better generalization
    data_augment = ImageDataGenerator(
        zoom_range=0.12,          
        rotation_range=9, 
        height_shift_range=0.1,      
        width_shift_range=0.1,     
        horizontal_flip=True,    
        vertical_flip=False)

    #Stop training if val_loss has stopped improving after 100 epochs
    early_stop = EarlyStopping('val_loss', patience=100)

    #Reduce learning rate if val_loss stops improving after 20 epochs
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=40, min_lr=0.00001, model='auto')

    #Save the model periodically
    model_checkpoint = ModelCheckpoint('CNN', 'val_loss', verbose=1, save_best_only=True)

    callbacks = [model_checkpoint, early_stop, reduce_lr]

    #Train the model
    model.fit(data_augment.flow(nn.x_train, nn.y_train, 16),                  
                  epochs=256, 
                  verbose=1,
                  callbacks = callbacks, 
                  validation_data=(nn.x_test, nn.y_test), 
                  shuffle=True)






if __name__ == "__main__":
    main()


        

