import os
import keras
# importing the checkpoint
from keras.callbacks import ModelCheckpoint
# Here sequential model from keras.models is imported
from keras.models import load_model
# model sequential is imported
from keras.models import Sequential
# 2D convolution from keras'layers is imported,which is for extracting automating features of images
from keras.layers import Conv2D
# The primary aim of pooling operation to reduce the size of the image
# without pixel loss. It can be either average pooling or max pooling.
from keras.layers import MaxPooling2D
# Flatten function is used to convert array in 1D. Since the weights in the fully connected layers are 1-D, matrix comming from max pool needs to be converted into 1-d so that multiplication can take place.
from keras.layers import Flatten
# Dense is the output layer. If one is doing Binary classification, then last activation layer is sigmoid whereas is one is doing Multi-class, single-label classification then the last layer activation function is softmax.
from keras.layers import Dense
# Dropout helps in preventing the overfitting
from keras.layers import Dropout
# Visualize Model.
from keras.utils.vis_utils import plot_model
# Keras has this Image Data Generator class which allows the users to perform image augmentation
from keras.preprocessing.image import ImageDataGenerator

# Here object of sequential model is made
classifier = Sequential()
# classifier.add is use to add layers in our model
# 2D Convolution is used for features extraction such as edges, facial features or in short mid level low level and high level features
# Result of convolutional layer is the activation map
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# activation function relu tends to elimate the negative value from the convoluted output.
# dropout is used to prevent the overfitting and removed the percentage of neuron randomnly
classifier.add(Dropout(0.2))
# reduced the size of the features maps by sliding over the image
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(16, (3, 3), activation='relu'))
classifier.add(Dropout(0.2))
# reduced the size of the features maps
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(8, (3, 3), activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(4, (3, 3), activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# convert the n-dimensional into one dimensional.
classifier.add(Flatten())
# fully connected layer
# multiplication is possible when the order of the matrix is same in 1-d
# FC1
classifier.add(Dense(units=100, activation='relu'))
# output layer
# since the problem is of the binary classifier hence in last layer sigmoid is used as activation function
# only one neuron is used for the prediction
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

plot_model(classifier, to_file='model_plot_binary.png',
           show_shapes=True, show_layer_names=True)

# get the current directory of the working
# this helps to find the path of the image in an simplified way.
current_cwd = os.getcwd()

# directory name where model will be saved
dirName = 'save_keras_model'

try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
except FileExistsError:
    print("Directory " , dirName ,  " already exists")

# creating the checkpoint directory. This is same where we need to save the model
checkpoint = dirName
# join the filename to the checkpoint directory
file_path = os.path.join(checkpoint,
                    "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")

# model saving provided only best model in the desired epochs are saved.
checkpoint = ModelCheckpoint(file_path, monitor='val_acc',
                             verbose=1, save_best_only=True, mode='max')
# this will be passed during training
callbacks_list = [checkpoint]

# scaling the training images
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
# scaling the testing images
test_datagen = ImageDataGenerator(rescale=1./255)

# loading the training_set
training_set = train_datagen.flow_from_directory('training_images',
                                                 target_size=(64, 64),
                                                 batch_size=2,
                                                 class_mode='binary')
# loading the testing images
test_set = test_datagen.flow_from_directory('testing_images',
                                            target_size=(64, 64),
                                            batch_size=2,
                                            class_mode='binary')

# to display the class as well as integer assigned to them'
index_class = training_set.class_indices
print("Index of each class:", index_class)

Length_training = training_set.samples
Length_testing = test_set.samples
print("Total Samples of training images are:", Length_training)
print("Total Samples of testing imagesare:", Length_testing)


# calculating steps_per_epochs
# Since we have two labels hence class_model is binary in nature
# for large dataset use fit_generator whereas for smaller dataset fit is used.

batch_size =  training_set.batch_size

classifier.fit_generator(training_set,
                         steps_per_epoch=round(Length_training/batch_size),
                         epochs=50,
                         validation_data=test_set,
                         validation_steps=round(Length_testing/batch_size),
                         callbacks=callbacks_list)
