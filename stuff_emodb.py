import glob
import os
import keras
import pickle
from keras.models import save_model, load_model
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

emotions =  {'W':1, 'L':2, 'E':3, 'A':4, 'F':5, 'T':6, 'N':7}
emotions_english  ={1:'Anger', 2:'Boredom', 3:'Disgust', 4:'Fear', 5: 'Happy', 6:'Sadness', 7:'Neutral'}


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
              mfccs, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print ("Error encountered while parsing file: ", fn)
              continue

            ext_features = np.hstack([mfccs,contrast,tonnetz])
            #print(np.array(ext_features))
            features.append(ext_features)
            #features = np.column_stack(())
            labels.append(fn.split('\\')[2][5])
            keylabel.append(emotions[labels[i-1]])
            #print (labels)
    return np.array(features), np.array(keylabel)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels-1] = 1
    return one_hot_encode   

parent_dir = 'download'
tr_sub_dirs = ["wav"]
ts_sub_dirs = ["test"]
pred_sub_dirs = ["pred"]

tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
print(tr_labels)
print(tr_features[0])
#tr_labels = one_hot_encode(tr_labels)

ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)
#ts_labels = one_hot_encode(ts_labels)


#keras part
model = keras.models.Sequential()
model.add(keras.layers.Dense(250, activation = 'relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(8, activation = 'softmax'))

model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy']) 

model.fit(tr_features, tr_labels, epochs=50)

valoss, valacc = model.evaluate(ts_features, ts_labels)
print(valoss, valacc)

#save the model
model.save('CNN_4.models')

#load the model
#new_model = tf.keras.models.load_model('CNN.models')

pr_features, pr_labels = parse_audio_files(parent_dir, pred_sub_dirs)
predictions = model.predict([pr_features])
print(np.argmax(predictions))
