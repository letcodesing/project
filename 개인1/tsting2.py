import numpy as np
import pandas as pd
from yaml import ScalarEvent

data = np.load('c:/project/개인1/data2.npy')
# print(data)
print(data.shape)
print(len(data))
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,RobustScaler,StandardScaler,Normalizer
Minscaler = MinMaxScaler()
Maxscaler = MaxAbsScaler()
Stanscaler = StandardScaler()
Roscaler = RobustScaler()
Norscaler = Normalizer()
# print(data)
data[:,1:2] = Roscaler.fit_transform(data[:,1:2])
data[:,2:4] = Stanscaler.fit_transform(data[:,2:4])
data[:,4:5] = Maxscaler.fit_transform(data[:,4:5])
data[:,6:8] = Minscaler.fit_transform(data[:,6:8])
# print('스케일링 후/',data)
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset)-4):
        subset = dataset[i:i+size,1:7]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)
x = split_x(data,5)
# print(x)
# print(x.shape)
econo_x = x[:-5,:,:3]
# print(econo_x)
# print(econo_x.shape)
econo_x_pred = x[53:,:,:3]
# print(econo_x_test)
# print(econo_x_test.shape)
democ_x = x[:-5,:,3:6]
# print(democ_x)
# print(democ_x.shape)
democ_x_pred = x[53:,:,3:6]
# print(democ_x_test)
# print(democ_x_test.shape)
#data에서 그냥 잘라도 되지만 웨이트값 적용을 위해선 트레인과 와꾸가 맞아야하므로 자른다        맞나?

president_y  = data[4:57,7:8]
congressmember_y  = data[4:57,8:]

from sklearn.model_selection import train_test_split
econo_x_train, econo_x_test, president_y_train, president_y_test = train_test_split(econo_x, president_y, train_size=0.9, random_state=187)
democ_x_train, democ_x_test, congressmember_y_train, congressmember_y_test = train_test_split(econo_x, president_y, train_size=0.9, random_state=187)
# print(president_y)
# print(president_y.shape)
# print(congressmember_y)
# print(congressmember_y.shape)

# from sklearn.model_selection import train_test_split


    
print(econo_x_train.shape, democ_x_train.shape, econo_x_test.shape, democ_x_test.shape)
print(president_y_train.shape, president_y_test.shape, congressmember_y_train.shape, congressmember_y_test.shape)
print(econo_x_pred.shape, democ_x_pred.shape)
#프레지던트 y 시그모이드, 콩그래스멤버 리니어
#앙상블모델 구성
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv1D, Conv2D, Flatten, Reshape,ReLU, LSTM, GRU, concatenate,Dropout,MaxPooling1D,MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=200,mode='auto',restore_best_weights=True,verbose=1)
mc = ModelCheckpoint('bestmodel.h5',monitor='val_loss',mode='min',save_best_only=True)
#모델1
econo = Input(shape=(5,3))
econo = Dense(200,activation='relu')(econo)
econo = Dense(15)(econo)
econo = Reshape((75,1))(econo)
econo = ReLU(2800,5)(econo)
econo = LSTM(64,activation='relu')(econo)
# econo = Flatten()(econo)
econo = Dense(200)(econo)
econo = Dense(15)(econo)
econo = Reshape((5,3))(econo)
# econo  = Flatten()(econo)
# econo  = Dense(200)(econo)
# econo = Reshape((50,4))(econo)
# econo =  ReLU(8,10)(econo)

#모델2
democ = Input(shape=(5,3))
democ = Dense(200)(democ)
democ = Dense(200)(democ)
democ = Dense(200)(democ)
democ = Dense(15)(democ)
democ = Reshape((75,1))(democ)
democ = ReLU(28,10)(democ)
democ = LSTM(64,return_sequences=True)(democ)
democ = Flatten()(democ)
democ = Dense(2000, activation='relu')(democ)
democ = Dense(15)(democ)
democ = Reshape((5,3))(democ)

# democ  = Flatten()(democ)
# democ  = Dense(200)(democ)
# democ = Reshape((50,4))(democ)
# democ =  ReLU(8,10)(democ)


#merge
president = concatenate((econo,democ))
president = Conv1D(12,3)(president)
president = Flatten()(president)
president = Dense(200,activation='tanh')(president)
president = Dense(200,activation='tanh')(president)
president = Dense(200,activation='tanh')(president)
president = Dropout(0.2)(president)
president = Dense(200,activation='elu')(president)
president = Dense(20)(president)
president = Dense(1,activation='sigmoid')(president)

congress = concatenate((econo,democ))
congress = Conv1D(12,3)(congress)
congress = Flatten()(congress)
congress = Dense(800,activation='relu')(congress)
congress = Dense(800,activation='relu')(congress)
congress = Dense(800,activation='relu')(congress)
congress = Dropout(0.4)(congress)
congress = Dense(80)(congress)
congress = Dense(4)(congress)

model = Model(inputs=[econo,democ], outputs=[president,congress],)
model.summary()

model.load_weights('c:/project/개인1/weights.h5')

model.compile(loss=['binary_crossentropy','mse'], optimizer='AdaMax')
# hist = model.fit([econo_x,democ_x],[president_y,congressmember_y],epochs=15000,batch_size=10, validation_split=0.1, callbacks=[es,mc])

# model.save_weights('c:/project/개인1/weights.h5')


loss = model.evaluate([econo_x_test,democ_x_test],[president_y_test,congressmember_y_test])
pred = model.predict([econo_x_pred,democ_x_pred])
# from sklearn.metrics import r2_score
# r2 = r2_score(  )


print('loss:',loss)
# print(type(pred))
print(np.where(pred[0][-1]>=0.5,'27년 대선에서는 야당후보가 당선됩니다','27년 대선에서는 여당후보가 당선됩니다'))
# print(pred)
# print(pred[1][-4].round())
total_congress = pred[1][-4][0].round()+pred[1][-4][1].round()+pred[1][-4][2].round()+pred[1][-4][3].round()
print('24년 총선에서 민주당계는',pred[1][-4][0].round(),'명,','보수당계는',pred[1][-4][1].round(),'명,','진보당계',pred[1][-4][2].round(),'명,','무소속', pred[1][-4][3].round(),'명이 당선됩니다.')
print('합계', total_congress,'명')

# print(pred.shape)

print()


import matplotlib.pyplot  as plt
plt.figure(figsize=(10,10))
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.legend()
# plt.scatter()
# plt.show()

