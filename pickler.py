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

emotions_rav = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprised'}
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
              i = i+1
              print(i)
              mfccs, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print ("Error encountered while parsing file: ", fn)
              continue

            ext_features = np.hstack([mfccs,contrast,tonnetz])
            features.append(ext_features)
            labels.append(fn.split('\\')[2][7])

    return np.array(features), np.array(labels)


parent_dir = 'download'
tr_sub_dirs = ["ravdess_train"]
ts_sub_dirs = ["ravdess_test"]
pred_sub_dirs = ["pred"]

tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)

pickle_out = open("ravdess_db.pickle", "wb")
pickle.dump(tr_features, pickle_out)
pickle_out.close()

pickle_out = open("ravdess_db_labels.pickle", "wb")
pickle.dump(tr_labels, pickle_out)
pickle_out.close()

X_train, X_test, y_train, y_test = train_test_split(tr_features, tr_labels , test_size=0.2, random_state=42, shuffle=True)



#keras part
model = keras.models.Sequential()
model.add(keras.layers.Dense(250, activation = 'relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(8, activation = 'softmax'))

model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs=25)

valoss, valacc = model.evaluate(X_test, y_test)
print(valoss, valacc)