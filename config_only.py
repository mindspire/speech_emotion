import glob
import os
import keras
import pickle
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import save_model, load_model

#training data pickled
pickle_in = open("berlin_db.pickle", "rb")
training_set = np.array(pickle.load(pickle_in))

pickle_in_lab = open("berlin_db_labels.pickle", "rb")
training_labels = np.array(pickle.load(pickle_in_lab))

#Emotions dict
emotions =  {'W':1, 'L':2, 'E':3, 'A':4, 'F':5, 'T':6, 'N':7}
emotions_english  ={1:'Anger', 2:'Boredom', 3:'Disgust', 4:'Fear', 5: 'Happy', 6:'Sadness', 7:'Neutral'}

#in_shape = training_set.shape
#training_set = training_set.reshape(in_shape[0], in_shape[1], 1)


X_train, X_test, y_train, y_test = train_test_split(training_set, training_labels, test_size=0.2, random_state=42)


#keras config
model = keras.models.Sequential()
model.add(keras.layers.Dense(256, activation = 'relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(8, activation = 'softmax'))
model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])


model.fit(X_train, y_train, epochs=100, batch_size=32)

valoss, valacc = model.evaluate(X_test, y_test)
print(valoss, valacc)

#save the model
model.save('80ish.models')

