import soundfile
import numpy as np
import librosa
import glob
import os
import librosa.display
from sklearn.model_selection import train_test_split
import keras
import pandas as pd
from keras.utils import np_utils

# all emotions on RAVDESS dataset
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# we allow only these emotions
AVAILABLE_EMOTIONS = [
    "angry",
    "happy",
    "sad"
]
    


def extract_feature(file_name, **kwargs):
    
    #Extract feature from audio file `file_name`
       
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    
    X, sample_rate = librosa.load(file_name, sr = 16000)
    #print("for new file , X.shape = ", X.shape)
    #librosa.display.waveplot(X, sample_rate)
    if chroma:
        #Use an energy (magnitude) spectrum instead of power spectrogram
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        #print("mfccs.shape = ", mfccs.shape)
        result = np.hstack((result, mfccs))
        #print(result.shape)
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        #print("chroma.shape = ", chroma.shape)
        result = np.hstack((result, chroma))
        #print(result.shape)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
        #print(result.shape)
   
    #print(result.shape)
    return result


def load_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob("data/Actor_*/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
   
    # what are the values for each class
    y= pd.DataFrame(y)
    y = pd.get_dummies(y)
    #print(y)
    y = y.iloc[:].values
    print("X.shape ", np.array(X).shape)
     # split the data to training and testing and return it
    return train_test_split(np.array(X),y, test_size=test_size, random_state=7)