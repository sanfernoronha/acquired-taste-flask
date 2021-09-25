from flask import Flask,request,jsonify,render_template
import os

# notebook imports
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import joblib
import keras
from keras import layers
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# load scaler
# scaler = joblib.load("scaler.save") 
# load model
model = load_model('./saved_model',compile= True)

mean_ = [ 3.79641731e-01 , 1.30854785e-01,  2.20021624e+03,  2.24184502e+03,
  4.56812566e+03 , 1.02638643e-01 ,-1.45381779e+02,  1.00941477e+02,
 -1.00045686e+01 , 3.72483486e+01 ,-2.00631455e+00  ,1.53978442e+01,
 -5.81720674e+00,  1.07710795e+01, -7.56917792e+00,  8.28651212e+00,
 -6.50166730e+00 , 4.94216369e+00 ,-5.18337891e+00 , 2.16398008e+00,
 -4.17695826e+00 , 1.44967174e+00 ,-4.19766479e+00 , 7.41688320e-01,
 -2.49768380e+00, -9.17650136e-01]

var_ = [8.17598438e-03, 4.70087347e-03, 5.65188084e+05, 2.95847921e+05,
 2.69583539e+06, 2.08308435e-03 ,1.13400824e+04, 1.20227510e+03,
 5.74855896e+02, 3.16912420e+02, 1.84148110e+02, 1.60078992e+02,
 1.23028998e+02, 1.23663550e+02, 8.77658499e+01 ,7.81749126e+01,
 6.12072863e+01, 5.71910780e+01, 5.08874088e+01, 3.70327651e+01,
 3.51428255e+01, 3.28911150e+01, 3.22273108e+01, 2.68266210e+01,
 2.61238868e+01, 2.75707945e+01]

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
       'metal', 'pop', 'reggae', 'rock']

def power(my_list):
    return [ x**0.5 for x in my_list]
def scale_data(array,means=mean_,stds=power(var_)):
    return (array-means)/stds

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "<p>Hello world!</p>"

@app.route('/classify',methods =['POST'])
def classify():

    data = request.files["song"]
    y, sr = librosa.load(data, mono=True, duration=30)
    
    # check file extention and type
    #convert to .wav
    #extract features and form array
    rmse = librosa.feature.rms(y=y)[0]
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    old_array = [np.mean(chroma_stft),np.mean(rmse),np.mean(spec_cent),np.mean(spec_bw),np.mean(rolloff),np.mean(zcr)]
    for e in mfcc:
        old_array.append(np.mean(e))
    input_array = np.array(old_array,dtype= float)

    #scale
    # scaler.transform(input_array.reshape(1,26))
    new_scaled_data = scale_data(input_array)
    #predict
    print(new_scaled_data)
    predictions = model.predict(new_scaled_data.reshape(1,-1))
    classes = np.argmax(predictions,axis=1)
    
    #return 
    return jsonify({'result': genres[classes[0]]})



if __name__ == '__main__':
    port = int(os.environ.get("PORT",5000))
    app.run(host='0.0.0.0',port=port)