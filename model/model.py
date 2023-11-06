from fileinput import filename
import pandas as pad
import numpy as nup
import glob
import soundfile
import os
import sys
import scipy
import imp
import joblib
import librosa
import librosa.display
import seaborn as sbn
import matplotlib.pyplot as mplt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import collections

from IPython.display import Audio

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


RavdessData = "../dataset/Actor_01"


ravdessDirectoryList = os.listdir(RavdessData)
fileEmotion = []
filePath = []
for dir in ravdessDirectoryList:
    actor = os.listdir(RavdessData + '\\' + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        fileEmotion.append(int(part[2]))
        filePath.append(RavdessData+ '\\' + dir + '\\' + file)

emotion_df = pad.DataFrame(fileEmotion, columns=['Emotions'])
path_df = pad.DataFrame(filePath, columns = ['Path'])
Ravdess_df = pad.concat([emotion_df, path_df], axis =1)

emotion_dict = {1:'neutral', 2: 'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}

Ravdess_df.Emotions.replace(emotion_dict, inplace=True)
Ravdess_df.to_csv('Ravdess.csv')


print(Ravdess_df.head())

dataPath = pad.concat([Ravdess_df], axis = 0)
dataPath.to_csv("data_path.csv", index = False)
print('\n\n')
print(dataPath.head())

mplt.title('Count of Emotions', size = 16)
df = pad.read_csv('data_path.csv')
df = list(df['Emotions'])
frequency = collections.Counter(df)
print(df)
df = pad.DataFrame(list(dataPath['Emotions']))
sbn.countplot(data = df)
sbn.barplot(x=list(frequency.keys()),y=list(frequency.values()))
sbn.countplot(data=list(dataPath['Emotions']))
mplt.ylabel('Count', size = 12)
mplt.xlabel('Emotions', size = 12)
sbn.despine(top = True, right = True, left = False, bottom = False)
mplt.show()


def createWaveplot(data, sr, e):
    mplt.figure(figsize =(10,3))
    mplt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    mplt.show()

def createSpectrogram(data, sr, e):
    X= librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    mplt.figure(figsize=(12,3))
    mplt.title('Spectrogram for audio with {} emotion'.format(e), size = 15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    mplt.colorbar()
    mplt.show()


def noise(data):
    noiseAmp = 0.035*nup.random.uniform()*nup.amax(data)
    data = data + noiseAmp*nup.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shiftRange = int(nup.random.uniform(low=-5, high=5)*1000)
    return nup.roll(data, shiftRange)

def pitch(data, samplingRate, pitchFactor=0.7):
    return librosa.effects.pitch_shift(data, samplingRate, pitchFactor)

emotion= emotion_dict[i]
print(emotion)
path = nup.array(dataPath.Path[dataPath.Emotions==emotion])[1]
data, samplingRate = librosa.load(path)
createWaveplot(data, samplingRate, emotion)
createSpectrogram(data, samplingRate, emotion)
Audio(path)
x = noise(data)
mplt.figure(figsize=(14,4))
mplt.title('Noise in audio with {} emotion'.format(emotion), size = 15)
librosa.display.waveshow(y=x, sr=samplingRate)
mplt.show()
Audio(x, rate=samplingRate)

def extractFeature(filename, mfcc, chroma, mel):
    with soundfile.SoundFile(filename)  as soundFile:
        X = soundFile.read()
        # print(X)
        sampleRate = soundFile.samplerate
        if chroma:
            stft=nup.abs(librosa.stft(X))
        result=nup.array([][:])
        if mfcc:
            mfccs = nup.mean(librosa.feature.mfcc(y=X, sr=sampleRate, n_mfcc=40).T, axis=0)
            # print(result.ndim, mfccs.ndim)
            result=nup.hstack((result, mfccs))
        if chroma:
            chroma=nup.mean(librosa.feature.chroma_stft(S=stft, sr= sampleRate).T, axis =0)
            result=nup.hstack((result, chroma))
        if mel:
            mel=nup.mean(librosa.feature.melspectrogram(X, sr=sampleRate).T, axis=0)
            result=nup.hstack((result, mel))
    # print(result)
    return result


emotions = {'01':'neutral', '02': 'calm', '03':'happy', '04':'sad', '05':'angry', '06':'fear', '07':'disgust', '08':'surprise'}
observedEmotions =['calm','happy', 'fear', 'angry', 'sad', 'neutral']


def loadData(test_size=0.2):
    x, y = [], []
    for file in glob.glob("C:\\Users\\Happy\\Desktop\\sem 5\\AI\\project\\audio_speech_actors_01-24\\Actor_*\\*.wav"):
        fileName = os.path.basename(file)
        emotion1 = emotions[fileName.split('-')[2]]
        if emotion1 not in observedEmotions:
            continue
        # print(file)
        feature = extractFeature(file, mfcc=True, chroma=True, mel=True)
        # print(feature.shape, feature.ndim)
        x.append(feature)
        y.append(emotion1)

    return train_test_split(nup.array(x), y, test_size=test_size, random_state=9)

xTrain, xTest, yTrain, yTest = loadData(test_size=0.23)
print(xTrain.shape[0], xTest.shape[0])
print(f'Features extracted: {xTrain.shape[1]}')

model = MLPClassifier(alpha=0.01, batch_size = 256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model = joblib.load('finalized_model.sav')
model.fit(xTrain,yTrain)
joblib.dump(model,'finalized_model.sav')

expected_Of_y= yTest
yPred = model.predict(xTest)
print(yPred, expected_Of_y)

print(metrics.confusion_matrix(expected_Of_y, yPred))

print(classification_report(yTest, yPred))

accuracy = accuracy_score(y_true=yTest, y_pred=yPred)

print("Accuracy: {}".format(accuracy*100))
