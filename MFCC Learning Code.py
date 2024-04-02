# 구글 코랩 환경에서 실행.


from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import itertools
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt


midi_file = "/content/drive/MyDrive/capstonedata/3mat_231114.wav"

#파일에는 3개의 재질의 소리가 200개씩 3초 간격으로 존재

materials = [0, 1, 2]
num_materials = 200
sec = 3

audio = []
inst = []
for inst_idx, num_mat in itertools.product(range(len(materials)), range(num_materials)):
  material = materials[inst_idx]
  offset = (material*num_materials*sec) + (num_mat*sec)
  print('material: {}, num_mat: {}, offset: {}'.format(material, num_mat, offset))
  y, sr = librosa.load(midi_file, sr=None, offset=offset, duration=3.0)
  audio.append(y)
  inst.append(inst_idx)

import numpy as np


max_length = max(len(y) for y in audio)

# 모든 오디오 신호를 최대 길이로 자르거나 0으로 채움
audio_padded = []
for y in audio:
    if len(y) < max_length:
        y_padded = np.pad(y, (0, max_length - len(y)), 'constant')
    else:
        y_padded = y[:max_length]
    audio_padded.append(y_padded)

audio_np = np.array(audio_padded, np.float32)
inst_np = np.array(inst, np.int16)

print(max_length)
print(audio_np.shape, inst_np.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(audio_np)



#-------------<MFCC를 이용한 머신러닝 오디오 분류>------------------------
audio_mfcc = []

for y in audio:
    ret = librosa.feature.mfcc(y=y,sr=sr, hop_length=1024)
    audio_mfcc.append(ret)

mfcc_np = np.array(audio_mfcc, np.float32)
inst_np = np.array(inst, np.int16)

print(mfcc_np.shape, inst_np.shape)

mfcc_np = mfcc_np.reshape((600, 20 * 141))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(mfcc_np)


#학습데이터와 실험 데이터를 분리
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(mfcc_np, inst_np, test_size=0.2)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)


#Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

LR = LogisticRegression()
LR.fit(train_x, train_y)
pred = LR.predict(test_x)
acc = accuracy_score(pred, test_y)
print(acc)



#Support Vector Machine

from sklearn import svm

SVM = svm.SVC(kernel='linear')
SVM.fit(train_x, train_y)
pred = SVM.predict(test_x)
acc = accuracy_score(pred, test_y)
print(acc)


#Decision Tree

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(train_x, train_y)
pred = DT.predict(test_x)
acc = accuracy_score(pred, test_y)
print(acc)


#----------------<DNN모델 구성>------------------

from keras.utils import to_categorical

mfcc_np = np.array(audio_mfcc, np.float32)
mfcc_np = mfcc_np.reshape((300, 20 * 141))
mfcc_array = np.expand_dims(mfcc_np, -1)
inst_cat = to_categorical(inst_np)

train_x, test_x, train_y, test_y = train_test_split(mfcc_array, inst_cat, test_size=0.2)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)



from keras.models import Sequential, Model
from keras.layers import Input, Dense

def model_build():
  model = Sequential()

  input = Input(shape=(2820, ), name='input')
  output = Dense(512, activation='relu', name='hidden1')(input)
  output = Dense(256, activation='relu', name='hidden2')(output)
  output = Dense(128, activation='relu', name='hidden3')(output)
  output = Dense(3, activation='softmax', name='output')(output)

  model = Model(inputs=[input], outputs=output)

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

  return model

model = model_build()
model.summary()

history = model.fit(train_x, train_y, epochs=50, batch_size=128, validation_split=0.2)

def plot_history(history_dict):
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

  epochs = range(1, len(loss) + 1)
  fig = plt.figure(figsize=(14, 5))

  ax1 = fig.add_subplot(1, 2, 1)
  ax1.plot(epochs, loss, 'b--', label='train_loss')
  ax1.plot(epochs, val_loss, 'r:', label='val_loss')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Loss')
  ax1.grid()
  ax1.legend()

  acc = history_dict['acc']
  val_acc = history_dict['val_acc']

  ax2 = fig.add_subplot(1, 2, 2)
  ax2.plot(epochs, acc, 'b--', label='train_accuracy')
  ax2.plot(epochs, val_acc, 'r:', label='val_accuracy')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Acc')
  ax2.grid()
  ax2.legend()

  plt.show()

plot_history(history.history)

model.evaluate(test_x, test_y)
model.save('/content/drive/MyDrive/capstonedata/mfcc_DNN_model')


#새로운 오디오 파일 불러오기
from google.colab import drive
drive.mount('/content/drive')


new_audio_file = "/content/drive/MyDrive/capstonedata/test_glass.wav"
test_audio = []
y, sr = librosa.load(new_audio_file, sr=None, duration=3.0)
test_audio.append(y)

#녹음 파일을 모델의 입력 형식에 맞게 변환

dec_mfcc = []

for y in test_audio:
  dec = librosa.feature.mfcc(y=y, sr=sr, hop_length=940)
  dec_mfcc.append(dec)

mfcc= np.array(dec_mfcc, np.float32)
mfcc = mfcc.reshape((1, 20 * 141))


print(mfcc.shape)

#모델에 적용하여 예측 수행, 결과 예측
predictions = model.predict(mfcc)
predicted_class = np.argmax(predictions, axis=1)

#예측된 클래스 출력
if predicted_class == 0:
    print("Audio_Predicted: plastic")
elif predicted_class == 1:
    print("Audio_Predicted: metal")
elif predicted_class == 2:
    print("Audio_Predicted: glass")
else:
    print("Audio_Predicted: Unknown")

#---------------------<CNN모델 구성>---------------------

from keras.utils import to_categorical

mfcc_np = np.array(audio_mfcc, np.float32)
mfcc_array = np.expand_dims(mfcc_np, -1)
inst_cat = to_categorical(inst_np)

train_x, test_x, train_y, test_y = train_test_split(mfcc_array, inst_cat, test_size=0.2)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

from keras.layers import Conv2D, MaxPool2D, Flatten, Input, Dense
from keras.models import Sequential, Model

def model_build():
  model = Sequential()

  input = Input(shape=(20, 141, 1))

  output = Conv2D(128, 3, strides=1, padding='same', activation='relu')(input)
  output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

  output = Conv2D(256, 3, strides=1, padding='same', activation='relu')(input)
  output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

  output = Conv2D(512, 3, strides=1, padding='same', activation='relu')(input)
  output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

  output = Flatten()(output)
  output = Dense(512, activation='relu')(output)
  output = Dense(256, activation='relu')(output)
  output = Dense(128, activation='relu')(output)

  output = Dense(3, activation='softmax')(output)

  model = Model(inputs=[input], outputs=output)

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

  return model

model = model_build()
model.summary()

history = model.fit(train_x, train_y, epochs=50, batch_size=128, validation_split=0.2)

def plot_history(history_dict):
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

  epochs = range(1, len(loss) + 1)
  fig = plt.figure(figsize=(14, 5))

  ax1 = fig.add_subplot(1, 2, 1)
  ax1.plot(epochs, loss, 'b--', label='train_loss')
  ax1.plot(epochs, val_loss, 'r:', label='val_loss')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Loss')
  ax1.grid()
  ax1.legend()

  acc = history_dict['acc']
  val_acc = history_dict['val_acc']

  ax2 = fig.add_subplot(1, 2, 2)
  ax2.plot(epochs, acc, 'b--', label='train_accuracy')
  ax2.plot(epochs, val_acc, 'r:', label='val_accuracy')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Acc')
  ax2.grid()
  ax2.legend()

  plt.show()

 plot_history(history.history)

model.evaluate(test_x, test_y)

model.save('/content/drive/MyDrive/mfcc_model2')

#녹음 파일을 모델의 입력 형식에 맞게 변환

dec_mfcc = []

for y in test_audio:
  dec = librosa.feature.mfcc(y=y, sr=sr, hop_length=940)
  dec_mfcc.append(dec)

mfcc= np.array(dec_mfcc, np.float32)
mfcc = np.expand_dims(mfcc, axis=-1)

print(mfcc.shape)

#모델에 적용하여 예측 수행, 결과 예측
predictions = model.predict(mfcc)
predicted_class = np.argmax(predictions, axis=1)

#예측된 클래스 출력
if predicted_class == 0:
    print("Audio_Predicted: plastic")
elif predicted_class == 1:
    print("Audio_Predicted: metal")
elif predicted_class == 2:
    print("Audio_Predicted: glass")
else:
    print("Audio_Predicted: Unknown")
