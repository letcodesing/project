import numpy as np
import pandas as pd
import tensorflow as tf            
tf.random.set_seed(1987)    

data = np.load('c:/project/개인1/npy, weight 저장/2/data.npy')
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
econo_x_r2 = x[52:57,:,:3]
econo_x_predic = x[57:,:,:3]
# print(econo_x_test)
# print(econo_x_test.shape)
democ_x = x[:-5,:,3:6]
# print(democ_x)
# print(democ_x.shape)
democ_x_r2 = x[52:57,:,3:6]
democ_x_predic = x[57:,:,3:6]
# print(democ_x_test)
# print(democ_x_test.shape)
#data에서 그냥 잘라도 되지만 웨이트값 적용을 위해선 트레인과 와꾸가 맞아야하므로 자른다        맞나?

president_y  = data[4:57,7:8]
president_y_r2  = data[52:57,7:8]
president_y_predic  = data[57:,7:8]

congressmember_y  = data[4:57,8:]
congressmember_y_r2  = data[52:57,8:]
congressmember_y_predic  = data[57:,8:]

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
econo_x_train = econo_x_train.reshape(-1,15)
democ_x_train = democ_x_train.reshape(-1,15)

from scipy import interpolate

from imblearn.over_sampling import SMOTE
econo_x_train, president_y_train = SMOTE(k_neighbors=5).fit_resample(econo_x_train, president_y_train)
democ_x_train, congressmember_y_train = SMOTE(k_neighbors=5).fit_resample(democ_x_train, congressmember_y_train)
# econo_x_train = econo_x_train.reshape(-1,5,3)
# democ_x_train = democ_x_train.reshape(-1,5,3)
# from keras.layers import UpSampling1D
# up = UpSampling1D
# econo_x_train = up(size=2)(econo_x_train)
# econo_x_train = np.array(econo_x_train)
# econo_x_train = econo_x_train.reshape(-1,15)

# president_y_train = president_y_train.reshape(-1,1,1)
# president_y_train = up(size=2)(president_y_train)
# # president_y_train = president_y_train.ravel()

# # president_y_train = pd.DataFrame(president_y_train)
# # president_y_train = president_y_train.apply(pd.to_numeric, errors='coerce')
# # president_y_train = president_y_train.astype(int)
# president_y_train = np.array(president_y_train)
# econo_x_train, president_y_train = SMOTE(k_neighbors=5).fit_resample(econo_x_train, president_y_train)
# econo_x_train = econo_x_train.reshape(-1,5,3)

# econo_x_train = up(size=2)(econo_x_train)
# econo_x_train = np.array(econo_x_train)

# econo_x_train = econo_x_train.reshape(-1,15)

# econo_x_train, president_y_train = SMOTE(k_neighbors=5).fit_resample(econo_x_train, president_y_train)
# econo_x_train = econo_x_train.reshape(-1,5,3)

# econo_x_train = up(size=2)(econo_x_train)

    
print(econo_x_train.shape, democ_x_train.shape, econo_x_test.shape, democ_x_test.shape)
print(president_y_train.shape, president_y_test.shape, congressmember_y_train.shape, congressmember_y_test.shape)
print(econo_x_r2.shape, congressmember_y_r2.shape)
print(econo_x_predic.shape, congressmember_y_predic.shape)