import numpy as np
import pandas as pd


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
print('스케일링 후/',data)
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
print(econo_x)
# print(econo_x.shape)
econo_x_loss = x[52:56,:,:3]
econo_x_predic = x[57:,:,:3]
# print(econo_x_test)
# print(econo_x_test.shape)
democ_x = x[:-5,:,3:6]
# print(democ_x)
# print(democ_x.shape)
democ_x_loss = x[52:56,:,3:6]
democ_x_predic = x[57:,:,3:6]
# print(democ_x)
# print(f'그만')
# print(democ_x_loss)
# print(f'그만`')
# print(democ_x_predic)
# print(democ_x.shape, democ_x_loss.shape, democ_x_predic.shape)
# print(x)