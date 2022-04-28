from __future__ import division, print_function
from flask import Flask,render_template,Response,request
from werkzeug.utils import secure_filename
import statistics as st

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import os 
import shutil
import librosa

app=Flask(__name__)

model = tf.keras.models.load_model('model4.h5')
speech_model = tf.keras.models.load_model("speech_model.h5")

final_results={1:'mask',0:'without mask'}
GR_dict={1:(0,255,0),0:(0,0,255)}

dict = {0:'Fear',1:'Happiness',2:'Sadness',3:'Neutral'}
dict1 = {0:'fear',1:'Happiness',2:'Neutral',3:'Sadness'}
# a=model.predict(X_test[i].reshape(1,48,48,1))
# print("Predicted:",dict[np.argmax(a)]," Actual:",dict[np.argmax(y_test[i])])

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

try:
    shutil.rmtree('songs')
except:
    print("unable to delete previous audio data or no song folder is present")

try: 
    os.mkdir("songs")
except: 
    print("directry is already present")

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


def emotions():
    cap = cv2.VideoCapture(0)

    count = 0

    while True:
           
        
        ret, img = cap.read()
    #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img,1.05,5)
        
        for x,y,w,h in faces:
            
            face_img = img[y:y+h,x:x+w ]
    #         resized = np.array(face_img,target_size=(128,128))
            resized = cv2.resize(face_img,(48,48))
            resized = np.array(tf.image.rgb_to_grayscale(resized,name = None)/255)
            reshaped=resized.reshape(1, 48, 48, 1)
            result = model.predict(reshaped)
            a = dict[np.argmax(result)]
            print(result)
            # a=model.predict(X_test[i].reshape(1,48,48,1))
            # print("Predicted:",dict[np.argmax(a)]," Actual:",dict[np.argmax(y_test[i])])
            
    #         label = np.argmax(result,axis=1)[0]
            
            cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
            cv2.putText(img, a, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
            # Sampling frequency
            freq = 44100

            # Recording duration
            duration = 2

            # Start recorder with the given values 
            # of duration and sample frequency
            recording = sd.rec(int(duration * freq), 
                            samplerate=freq, channels=1)

            # Record audio for the given number of seconds
            sd.wait()

            # This will convert the NumPy array to an audio
            # file with the given sampling frequency
            # write("recording0.wav", freq, recording)

            # Convert the NumPy array to audio file
            filename = "songs/recording"+str(count)+".wav"
            wv.write(filename, recording, freq, sampwidth=2)
            
            feature = extract_mfcc(filename)
            
            feature = np.array([feature])
            feature = feature.reshape(1, 40, 1)
            
            audio_result = speech_model.predict(feature)
            audio_result = np.argmax(audio_result)
            audio_result = dict1[audio_result]
            print(audio_result)
            
            count += 1
            
            cv2.putText(img, audio_result, (50,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0))
                

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        
        if key == 27: 
            break
            
    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    emotions()
    return render_template('index.html')


if __name__=='__main__':
    app.run(debug=True)