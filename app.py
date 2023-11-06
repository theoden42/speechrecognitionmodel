from flask import Flask, render_template, request, redirect, send_file
import joblib
import speech_recognition as sr
import numpy as nup
import librosa
import soundfile
import os
import glob
import matplotlib.pyplot as mplt
from IPython.display import Audio
import sys
import librosa.display
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

model = joblib.load('finalized_model.sav')

def extractFeature(filename, mfcc, chroma, mel):
    with soundfile.SoundFile(filename)  as soundFile:
        X = soundFile.read()
        sampleRate = soundFile.samplerate
        if chroma:
            stft=nup.abs(librosa.stft(X))
        result=nup.array([][:])
        if mfcc:
            mfccs = nup.mean(librosa.feature.mfcc(y=X, sr=sampleRate, n_mfcc=40).T, axis=0)
            result=nup.hstack((result, mfccs))
        if chroma:
            chroma=nup.mean(librosa.feature.chroma_stft(S=stft, sr= sampleRate).T, axis =0)
            result=nup.hstack((result, chroma))
        if mel:
            mel=nup.mean(librosa.feature.melspectrogram(X, sr=sampleRate).T, axis=0)
            result=nup.hstack((result, mel))
    return result

def loadData(file):
    x=[]
    for file in glob.glob("*.wav"):
        fileName = os.path.basename(file)
        feature=extractFeature(fileName, mfcc=True, chroma=True, mel=True)
        x.append(feature)
    return x

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" in request.files:
            file = request.files["file"]
        else:
            file = None
            
        if file:
            if file.filename == "":
                return redirect(request.url)
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                recorded_audio = recognizer.record(source)
            with open("microphone-results.wav", "wb") as f:
                                f.write(recorded_audio.get_wav_data())
        path = 'microphone-results.wav'
        xTest = loadData(recorded_audio)
        yPred = model.predict(xTest)
        print(xTest, yPred)
        transcript = yPred[0]

    return render_template('index.html', transcript=transcript)

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
