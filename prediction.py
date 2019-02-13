import keras
import glob
from collections import OrderedDict
from operator import itemgetter
import os
import shutil
import librosa
import pandas as pd
from keras.models import save_model, load_model
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
from keras.models import model_from_json
from keras.utils import np_utils
import pickle

emotion_profile = []
emotions = {'W':1, 'L':2, 'E':3, 'A':4, 'F':5, 'T':6, 'N':7}
emotions_english = {1:'Anger', 2:'Boredom', 3:'Disgust', 4:'Fear', 5: 'Happy', 6:'Sadness', 7:'Neutral'}
emotions_rav = {1:' Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Anger', 6:'Fear', 7:'Disgust', 8:'Surprised'}
emotions_nw = {0: 'Anger', 1: 'Calm', 2:'Fear', 3:'Happy', 4:'Sad',
               5: 'Anger', 6: 'Calm', 7:'Fear', 8:'Happy', 9:'Sad'}

def split_audio(audiofilename):
    myaudio = AudioSegment.from_file(audiofilename , "wav")
    chunk_length_ms = 5000 # pyub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        print ("exporting", chunk_name)

        chunk.export(os.path.join(os.path.normpath("C:\Me\Machine Learning\mindspire_emotion_recognition\download\pred"), chunk_name), format="wav")

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
    i = 0
    features = []
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

    return np.array(features)

def clear():
    folder = 'C:\Me\Machine Learning\mindspire_emotion_recognition\download\pred'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

parent_dir = 'download'
tr_sub_dirs = ["new"]
pred_sub_dirs = ["pred"]

split_audio("untitled.wav")
tr_features = parse_audio_files(parent_dir,pred_sub_dirs)


#Berlin Db model
new_model = load_model('80ish.models')
predictions = new_model.predict_classes(tr_features)
predlist = predictions.astype(int).flatten()
emo = predlist.tolist()
print(emo)
for i in range(len(emo)):
    print(emotions_english[emo[i]])
    emotion_profile.append(emotions_english[emo[i]])

#Ravdess DB Model
new_model = load_model('60ish.models')
tr_features = tr_features.reshape(tr_features.shape[0], tr_features.shape[1], 1)
predictions = new_model.predict_classes([tr_features])
predlist = predictions.astype(int).flatten()
emo = predlist.tolist()
print(emo)
for i in range(len(emo)):
    print(emotions_rav[emo[i]])
    emotion_profile.append(emotions_rav[emo[i]])

#Northeastern University
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
X, sample_rate = librosa.load('untitled.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
features = mfccs
pred_1 = features
pred_1= pd.DataFrame(data=pred_1)
pred_1 = pred_1.stack().to_frame().T

pred_2= np.expand_dims(pred_1, axis=2)
predictions = loaded_model.predict(pred_2,
                         batch_size=32,
                         verbose=0)

predictions =predictions.argmax(axis=1)
emo = predictions.astype(int).flatten()
emo = emo.tolist()
emotion_profile.append(emotions_nw[emo[0]])

#print(emotions_nw[emo[i]])
#emotion_profile.append(emotions_nw[emo[i]])

#Percentage breakdown of analysed emotions
emotion_distribution = {i:round(((emotion_profile.count(i)/emotion_profile.__len__()) * 100), 2) for i in emotion_profile}
print(OrderedDict(sorted(emotion_distribution.items(), key=lambda kv:kv[1], reverse=True)))


#housekeeping
clear()



