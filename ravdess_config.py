import glob
import os
import keras
import pickle
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import save_model, load_model

#training data pickled
pickle_in = open("ravdess_db.pickle", "rb")
training_set = np.array(pickle.load(pickle_in))

pickle_in_lab = open("ravdess_db_labels.pickle", "rb")
training_labels = np.array(pickle.load(pickle_in_lab))
print(training_labels.size)
print(training_labels)

#Emotions dict
emotions_rav = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprised'}

#reshaping
#in_shape = training_set.shape
#training_set = training_set.reshape(in_shape[0], in_shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(training_set, training_labels , test_size=0.2, random_state=42, shuffle=True)


#keras config
model = keras.models.Sequential()
model.add(keras.layers.Conv1D(64, 5, padding='same', input_shape=X_train[0].shape, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.MaxPool1D(pool_size=2))

model.add(keras.layers.Conv1D(64, 5, padding='same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Dense(32, activation = 'relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(9, activation = 'softmax'))


model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32)

valoss, valacc = model.evaluate(X_test, y_test)
print(valoss, valacc)

#save the model
#model.save('60ish.models')


