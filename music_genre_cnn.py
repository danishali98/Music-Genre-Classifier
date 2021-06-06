import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras
from keras import models
from keras import layers
from keras import regularizers
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

print("\n\n\nInitializing program...")

cmap = plt.get_cmap('inferno')

##Spectrogram for every Audio in the GTZAN dataset is saved for future use

print("Saving the spectogram from every file in dataset...")
plt.figure(figsize=(10,10))

## Un-comment the following lines when running the program for the first time
# genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
# for g in genres:
#     pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)     
#     for filename in os.listdir(f'C:/Users/Danish Ali/Python/genres/{g}'):
#         songname = f'C:/Users/Danish Ali/Python/genres/{g}/{filename}'
#         y, sr = librosa.load(songname, mono=True, duration=5)
#         plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
#         plt.axis('off');
#         plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
#         plt.clf()
print("Saving spectogram finished...")

##Getting features from the Spectrogram saved in the previous step

print("Writing features from spectogram to a CSV file...")
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

##Writing data to a file for reusability


## Un-comment the following lines when running the program for the first time
# file = open('data.csv', 'w', newline='')
# with file:
#     writer = csv.writer(file)
#     writer.writerow(header)
# genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
# for g in genres:
#     for filename in os.listdir(f'C:/Users/Danish Ali/Python/genres/{g}'):
#         songname = f'C:/Users/Danish Ali/Python/genres/{g}/{filename}'
#         y, sr = librosa.load(songname, mono=True, duration=30)
#         chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#         rmse = librosa.feature.rms(y=y)
#         spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#         rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#         zcr = librosa.feature.zero_crossing_rate(y)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr)
#         to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} 
#             {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
#         for e in mfcc:
#             to_append += f' {np.mean(e)}'
#         to_append += f' {g}'
#         file = open('data.csv', 'a', newline='')
#         with file:
#             writer = csv.writer(file)
#             writer.writerow(to_append.split())
print("Finished writing data...")

data = pd.read_csv('data.csv')

print(data.shape)

##Dropping the coulmn not needed
data = data.drop(['filename'],axis=1)

##Encoding labels

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

##Scaling the features

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

##Dividing data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Train data size: " + str(len(y_train)))
print("Test data size: " + str(len(y_test)))

##Building the network

print("Building model...")
model = models.Sequential([Dropout(0.25)])
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],),
                bias_regularizer=regularizers.l2(1e-4),
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Fitting the model...\n\n")
history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=128,
                    callbacks = [EarlyStopping(monitor='val_acc', patience=5)])

test_loss, test_acc = model.evaluate(X_test,y_test)
print('\n\nTest Accuracy: ' + str(test_acc))