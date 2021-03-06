import glob
import os
import keras
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import save_model, load_model
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

emotions =  {'W':1, 'E':3, 'A':4, 'F':5, 'T':6, 'N':7}
emotions_english  ={1:'Anger', 3:'Disgust', 4:'Fear', 5: 'Happy', 6:'Sadness', 7:'Neutral'}


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    #print(X)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
    i = 0
    features = []
    labels = []
    keylabel = []
    #features, labels = np.empty(0,193), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mfccs, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print ("Error encountered while parsing file: ", fn)
              continue
            bob = fn.split('\\')[2][5]
            if bob != 'L':
                i=i+1
                labels.append(bob)
                print (labels)
                keylabel.append(emotions[labels[i - 1]])
                ext_features = np.hstack([mfccs,contrast,tonnetz])
                #print(np.array(ext_features))
                features.append(ext_features)
            #features = np.column_stack(())

            #print (labels)
    return np.array(features), np.array(keylabel)

parent_dir = 'download'
tr_sub_dirs = ["wav"]
pred_sub_dirs = ["pred"]



x, y = parse_audio_files(parent_dir,tr_sub_dirs)

#pickling
pickle_out = open("berlin_db.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("berlin_db_labels.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


X_train, X_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=42, shuffle=True)


#keras part
model = keras.models.Sequential()
model.add(keras.layers.Dense(250, activation = 'relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(8, activation = 'softmax'))

model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs=50)

valoss, valacc = model.evaluate(X_test, y_test)
print(valoss, valacc)

#save the model
#model.save('CNN_4.models')

#load the model
#new_model = tf.keras.models.load_model('CNN.models')
