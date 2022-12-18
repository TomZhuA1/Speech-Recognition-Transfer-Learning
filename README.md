# Speech-Recognition-Transfer-Learning
In this project, we built and fine-tuned multiple CNN models to test their performance on a kaggle speech command audio dataset.

***Goal:*** Explore how different architectures/inputs performed on speech recognition. Select the model with the best performance and operate the hyperparameters optimizations technique on it.

***Approach:*** Download the speech dataset from Kaggle speech recognition challenge. Converting Wav files into a spectrogram and removing silence. Feed the images of these spectrograms into models and evaluate their performances.

***Benefit/Values：*** Evaluate the influence of architectures and inputs on the prediction performance of speech cognition models.

***Download Data:***

You need to first import Kaggle package
```
!pip install -q kaggle
```

Then, create a json file to store the api key
```
!mkdir ~/.kaggle
!touch ~/.kaggle/kaggle.json

api_token = {"username":"username","key":"key"}

import json

with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

!chmod 600 ~/.kaggle/kaggle.json
```

You can then use your user name and api key to download data from Kaggle
```
!kaggle competitions download -c tensorflow-speech-recognition-challenge
```

After Downloading the data, you need to unzip the files. 
```
!pip install pyunpack 
!pip install patool

!unzip /content/tensorflow-speech-recognition-challenge.zip -d /content/drive/MyDrive/Speech_Recognition/Unzip

from pyunpack import Archive

Archive("/content/drive/MyDrive/Speech_Recognition/Unzip/train.7z").extractall("/content/drive/MyDrive/Speech_Recognition/Unzip/data")
#Archive("/content/drive/MyDrive/Speech_Recognition/Unzip/test.7z").extractall("/content/drive/MyDrive/Speech_Recognition/Unzip/data")
```

***Data Pre-Processing:***

Before processing the data, we need to first cut the noise data into snippets of 1 seconds to make the input size of our data constant
```
import librosa
import numpy as np
import os
import csv
from scipy.io import wavfile
import matplotlib.pyplot as plt

train_dir = "/content/drive/MyDrive/Speech_Recognition/Unzip/data/train/audio/" 

classes = os.listdir("/content/drive/MyDrive/Speech_Recognition/Unzip/data/train/audio/")
classes.remove("_background_noise_")

def split_arr(arr):
    """
    split an array into chunks of length 16000
    Returns:
        list of arrays
    """
    return np.split(arr, np.arange(16000, len(arr), 16000))

import soundfile as sf

def create_silence():
    """
    reads wav files in background noises folder, 
    splits them and saves to silence folder in train_dir
    """
    for file in os.listdir("/content/drive/MyDrive/Speech_Recognition/Unzip/data/train/_background_noise_/"):
        if ".wav" in file:
            sig, sr = librosa.load("/content/drive/MyDrive/Speech_Recognition/Unzip/data/train/_background_noise_/"+file, sr = 16000) 
            sig_arr = split_arr(sig)
            if not os.path.exists(train_dir+"silence/"):
                os.makedirs(train_dir+"silence/")
            for ind, arr in enumerate(sig_arr):
                file_name = "frag%d" %ind + "_%s" %file # example: frag0_running_tap.wav
                sf.write(train_dir+"silence/"+file_name, arr, 16000)
                
create_silence()

folders = os.listdir(train_dir)
# put folders in same order as in the classes list, used when making sets
all_classes = [x for x in classes]
for ind, cl in enumerate(folders):
    if cl not in classes:
        all_classes.append(cl)
```

***Data Processing:***

In our project, we only want our model to classify the ten numbers, we will only work with audio files with labels=["eight", "five", "four", "nine", "one", "seven", "six", "tree", "two", "zero"]. This snippet of the code will read these audio files into raw waves.

```
labels=["eight", "five", "four",
        "nine", "one", "seven",
        "six", "tree", "two", "zero"]
duration_of_recordings=[]
x_list = []
y_list = []
for label in labels:
    waves = [f for f in os.listdir(train_dir + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_dir + '/' + label + '/' + wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))
        x_list.append(samples)
        y_list.append(label)
```

Due to the fact that some of the audio recording is noisy, we write this function for noise reduction

```
from scipy import signal
import random

def f_low(y,sr):
    b,a = signal.butter(35, 3000/(sr/2), btype='lowpass')
    yf = signal.lfilter(b,a,y)
    return yf
    
for i in range(len(x_list)):
  x_f = f_low(x_list[i]*1.0, 16000)
  x_list[i] = x_f
```

Also, some of the recordings are shorter than 1 second, we need codes to pad them into 1s

```
for i in range(len(duration_of_recordings)):
  if len(x_list[i]) != sample_rate:
    zeros = np.zeros(sample_rate - len(x_list[i]))
    new_x = np.concatenate((x_list[i], zeros), axis=None)
    x_list[i] = new_x
    duration_of_recordings[i] = float(len(x_list[i])/sample_rate)
```

Convert all the raw waves into a list of mel-spectrogram and MFCCs

```
mel_list = []
for i in range(len(x_list)):
  S = librosa.feature.melspectrogram(y = x_list[i]*1.0, sr=sample_rate, n_mels = 75, hop_length = 214)
  mel_list.append(S)
  
mel_list_db = []
for i in range(len(mel_list)):
  mel_list_db.append(librosa.power_to_db(mel_list[i])) #convert amplitude into db
  
from sklearn.preprocessing import normalize

for i in range(len(mel_list_db)):
  mel_list_db[i] = normalize(mel_list_db[i], axis=1, norm='l1') #normalize our mel-spectrogram matrix
  
mfcc_list = []
for i in range(len(mel_list)):
  mfcc = librosa.feature.mfcc(S=mel_list[i], n_mfcc = 75)
  mfcc_list.append(mfcc)
```

One-hot encoding for y values

```
import pandas as pd
from keras.utils import np_utils
y_df = pd.DataFrame(y_list)
y_codes, y_uniques = pd.factorize(y_df.iloc[:,0])
y_codes = np_utils.to_categorical(y_codes, num_classes=len(labels))
```

***Prediction with mel-spectrogram:***

train-val split:

```
from sklearn.model_selection import train_test_split

mel_train, mel_val, mel_y_train, mel_y_val = train_test_split(mel_list_db, np.array(y_codes), stratify = y_codes, test_size=0.1, random_state=12, shuffle = True)

mel_train = np.array(mel_train)
mel_train = mel_train.reshape(np.array(mel_train).shape[0], np.array(mel_train).shape[1], np.array(mel_train).shape[2], 1)
mel_val = np.array(mel_val)
mel_val = mel_val.reshape(np.array(mel_val).shape[0], np.array(mel_val).shape[1], np.array(mel_val).shape[2], 1)

mel_y_train = mel_y_train.astype(int)
mel_y_val = mel_y_val.astype(int)
```

Build and evaluate model 1(Traditional not-very-Deep CNN): 

```
import tensorflow as tf
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv2D, Input, MaxPooling2D
from keras.layers import BatchNormalization
model = Sequential()
model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=(75,75,1)))
model.add(BatchNormalization())

model.add(Conv2D(48, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(120, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        metrics=["accuracy"])
        
history = model.fit(mel_train, mel_y_train, batch_size=64, epochs=30, validation_data = (mel_val, mel_y_val))

plt.plot(history.history['loss'], label='train')  # losses learning curve of training set.
plt.plot(history.history['val_loss'], label='val') # losses learning curve of validation set.
plt.legend()
plt.title("losses learning curves")
plt.show()

plt.plot(history.history['accuracy'])      # Accuracy learning curve of training set.
plt.plot(history.history['val_accuracy'])  # Accuracy learning curve of validation set.
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.title("Accuracy learning curves")
plt.show()
```

Transfer learning, fine-tuning, evaluating Resnet 50 as our model2:
```
from keras.applications import ResNet50

model_resnet = ResNet50(weights = "imagenet", include_top = False, input_shape = (96,96,3))

new_model = Sequential()
new_model.add(model_resnet)
new_model.add(Flatten())
new_model.add(Dense(64, activation = "relu"))
new_model.add(Dense(10, activation = "softmax"))
new_model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
```

We need to pad our (75,75,1) data into (96,96,3) to fit the input size of ResNet50
```
mel_train = mel_train.reshape(mel_train.shape[0], mel_train.shape[1], mel_train.shape[2])
res_train = []
for i in range(len(mel_train)):
  res_train.append(np.dstack((mel_train[i], mel_train[i], mel_train[i])))
  
for i in range(len(res_train)):
  res_train[i] = np.pad(res_train[i], ((11,10), (11,10), (0, 0)), 'constant')

res_val = []
for i in range(len(mel_val)):
  res_val.append(np.dstack((mel_val[i], mel_val[i], mel_val[i])))

for i in range(len(res_val)):
  res_val[i] = np.pad(res_val[i], ((11,10), (11,10), (0, 0)), 'constant')

res_train = np.array(res_train)
res_val = np.array(res_val)
```
Fine Tuning ResNet50 visualize training process
```
tf.config.run_functions_eagerly(True)
history1 = new_model.fit(res_train, mel_y_train, batch_size=64, epochs=30, validation_data = (res_val, mel_y_val))

plt.plot(history1.history['loss'], label='train')  # losses learning curve of training set.
plt.plot(history1.history['val_loss'], label='val') # losses learning curve of validation set.
plt.legend()
plt.title("losses learning curves")
plt.show()

plt.plot(history1.history['accuracy'])      # Accuracy learning curve of training set.
plt.plot(history1.history['val_accuracy'])  # Accuracy learning curve of validation set.
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.title("Accuracy learning curves")
plt.show()
```

Transfer Learning and Fine Tuning InceptionV3
```
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3

model = InceptionV3(weights = "imagenet", include_top=False, input_shape = (75, 75, 3))

new_model = Sequential()
new_model.add(model)
new_model.add(Flatten())
new_model.add(Dense(64, activation = "relu"))
new_model.add(Dense(10, activation = "softmax"))
new_model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

inc_train = []
for i in range(len(mel_train)):
  inc_train.append(np.dstack((mel_train[i], mel_train[i], mel_train[i])))
inc_val = [] 
for i in range(len(mel_val)):
  inc_val.append(np.dstack((mel_val[i],mel_val[i],mel_val[i])))
  
inc_train = np.array(inc_train)
inc_val = np.array(inc_val)

tf.config.run_functions_eagerly(True)
history_inc = new_model.fit(inc_train, mel_y_train, epochs = 30, batch_size = 64, validation_data = (inc_val, mel_y_val))

plt.plot(history_inc.history['loss'], label='train')  # losses learning curve of training set.
plt.plot(history_inc.history['val_loss'], label='val') # losses learning curve of validation set.
plt.legend()
plt.title("losses learning curves")
plt.show()

plt.plot(history_inc.history['accuracy'])      # Accuracy learning curve of training set.
plt.plot(history_inc.history['val_accuracy'])  # Accuracy learning curve of validation set.
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.title("Accuracy learning curves")
plt.show()
```

***Training with MFCC***

We train a not-very-Deep CNN with the MFCC using this code:
```
mfcc_list = np.array(mfcc_list)

mfcc_train, mfcc_val, mfcc_y_train, mfcc_y_val = train_test_split(mfcc_list, np.array(y_codes), stratify = y_codes, test_size=0.1, random_state=12, shuffle = True)

model = Sequential()
model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=(75,75,1)))
model.add(BatchNormalization())

model.add(Conv2D(48, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(120, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        metrics=["accuracy"])
        
history_mfcc = model.fit(mfcc_train, mfcc_y_train, batch_size=64, epochs=100, validation_data = (mfcc_val,mfcc_y_val))    

plt.plot(history_mfcc.history['loss'], label='train')  # losses learning curve of training set.
plt.plot(history_mfcc.history['val_loss'], label='val') # losses learning curve of validation set.
plt.legend()
plt.title("losses learning curves")
plt.show()

plt.plot(history_mfcc.history['accuracy'])      # Accuracy learning curve of training set.
plt.plot(history_mfcc.history['val_accuracy'])  # Accuracy learning curve of validation set.
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.title("Accuracy learning curves")
plt.show()
```

***Conclusion:***

MFCC, as the compressed version of the spectrogram graphs, is destined to have a certain amount of information loss in the process of compression. Therefore, it ought to increase the loss and decrease the accuracy of the model.  But this conclusion is derived from general ideas rather than scientific research. 

In this study, we demonstrate how different image inputs could shift the performance of the models. Attest by the model performance trained with inputs differently, we can firmly announce that Mel_spectrogram fed speech recognition model has better performance than MFCC fed model on traditional  CNN architecture. However, we stay cautious on this conclusion since it may not apply to other  architectures . 

Beyond this, we also discover that models fed by MFCC tend to be more stable and “sustainable”. This is clearly demonstrated through the graph of model accuracy vs. time.  Training time is minimized when feeding MFCC to the models.




In addition to the findings regarding how different transformations of audio files could influence the model performance, we also evaluate how different architectures performed on classifying speech recognition. The  InceptionV3 model outperforms ResNet50 and traditional CNN(with only 3 conv layers), but on a really small scale. Again, we stay cautious about this finding because the model performance on the validation set, specifically accuracy, varies a lot. The slight difference in accuracy could result from noise or human error in this experiment. Overall, InceptionV3 has better performance in both training and testing dataset. And it’s performance is more stable than other two architectures. It also takes the most amount of time to train each epoch. 







Potential Future Studies:
Explore in-depth about the tradeoff curves among different inputs. [transformed images from audio files]
Explore in-depth about the input shape and model performance.
Explore in-depth about the tradeoff between model accuracy and training time.
Reference:
https://www.kaggle.com/code/mohamedahmedae/inceptiontime-ctc-models-val-f1socre-0-96
https://www.kaggle.com/code/davids1992/speech-representation-and-data-exploration?scriptVersionId=1924001
https://www.kaggle.com/code/faresabbasai2022/speech-recognition
https://keras.io/examples/audio/ctc_asr/
https://www.kaggle.com/code/mayarmohsen/numbersyscnn

Author: Tom Zhou     tz1307@nyu.edu
Jiabao(Sheldon) Wang jw6452@nyu.edu
