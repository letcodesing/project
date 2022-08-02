import numpy as np
import pandas as pd
import tensorflow as tf            
tf.random.set_seed(1987)    

data = np.load('c:/project/개인1/npy, weight 저장/2/data.npy')
# print(data)
print(data.shape)
print(len(data))

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset)-4):
        subset = dataset[i:i+size,1:7]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)


x = split_x(data,5)
econo_x = x[:-5,:,:3]
democ_x = x[:-5,:,3:6]
econo_x_pred = x[-5:,:,:3]
democ_x_pred = x[-5:,:,3:6]
print(econo_x)
print('============================')
print(democ_x)

def split_y(dataset, size):
    aaa = []
    for i in range(len(dataset)-9):
        subset = dataset[i+5:i+5+size,7:]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

y = split_y(data,5)
print(y)
print(y.shape)
print(data.shape)
print(len(data))
president_y  = y[:,:,:1]
congressmember_y  = y[:,:,1:5]
print(president_y)
print(congressmember_y)
print(president_y.shape, congressmember_y.shape)

from sklearn.model_selection import train_test_split
econo_x_train, econo_x_test, democ_x_train, democ_x_test, president_y_train, president_y_test, congressmember_y_train, congressmember_y_test = train_test_split(econo_x, democ_x,president_y,congressmember_y, train_size=0.8, random_state=187)


from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,RobustScaler,StandardScaler,Normalizer
Minscaler = MinMaxScaler()
Maxscaler = MaxAbsScaler()
Stanscaler = StandardScaler()
Roscaler = RobustScaler()
Norscaler = Normalizer()

econo_x_train = econo_x_train.reshape(-1,15)
econo_x_test = econo_x_test.reshape(-1,15)
econo_x_pred = econo_x_pred.reshape(-1,15)

democ_x_train = democ_x_train.reshape(-1,15)
democ_x_test = democ_x_test.reshape(-1,15)
democ_x_pred = democ_x_pred.reshape(-1,15)

econo_x_train[:,0:1] = Roscaler.fit_transform(econo_x_train[:,0:1])
econo_x_test[:,0:1] = Roscaler.transform(econo_x_test[:,0:1])
econo_x_pred[:,0:1] = Roscaler.transform(econo_x_pred[:,0:1])

econo_x_train[:,1:3] = Stanscaler.fit_transform(econo_x_train[:,1:3])
econo_x_test[:,1:3] = Stanscaler.transform(econo_x_test[:,1:3])
econo_x_pred[:,1:3] = Stanscaler.transform(econo_x_pred[:,1:3])




democ_x_train[:,0:1] = Maxscaler.fit_transform(democ_x_train[:,0:1])
democ_x_test[:,0:1] = Maxscaler.transform(democ_x_test[:,0:1])
democ_x_pred[:,0:1] = Maxscaler.transform(democ_x_pred[:,0:1])

democ_x_train[:,1:3] = Minscaler.fit_transform(democ_x_train[:,1:3])
democ_x_test[:,1:3] = Minscaler.transform(democ_x_test[:,1:3])
democ_x_pred[:,1:3] = Minscaler.transform(democ_x_pred[:,1:3])
econo_x_train = econo_x_train.reshape(-1,5,3)
econo_x_test = econo_x_test.reshape(-1,5,3)
econo_x_pred = econo_x_pred.reshape(-1,5,3)

democ_x_train = democ_x_train.reshape(-1,5,3)
democ_x_test = democ_x_test.reshape(-1,5,3)
democ_x_pred = democ_x_pred.reshape(-1,5,3)

# print(president_y)
# print(president_y.shape)
# print(congressmember_y)
# print(congressmember_y.shape)

# from sklearn.model_selection import train_test_split

    
print(econo_x_train.shape, democ_x_train.shape, econo_x_test.shape, democ_x_test.shape)
print(president_y_train.shape, president_y_test.shape, congressmember_y_train.shape, congressmember_y_test.shape)
import torch
from torch import nn
import torch.nn.functional as F
import tensorflow as tf
print(type(econo_x_train))
econo_x_train = torch.tensor(econo_x_train)
econo_x_train = F.interpolate(econo_x_train, scale_factor=50, mode='linear')
from scipy.interpolate import griddata
grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
from scipy import interpolate
GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                          (xx, yy),
                             method='cubic')