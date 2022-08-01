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
econo_x = x[:,:,:3]
democ_x = x[:,:,3:6]
print(econo_x)
print('============================')
print(democ_x)

# def split_y(dataset, size):
#     aaa = []
#     for i in range(len(dataset)-9):
#         subset = dataset[i+5:i+5+size,7:]
#         aaa.append(subset)
#     print(type(aaa))
#     return np.array(aaa)

# y = split_y(data,5)
# print(y)
# print(y.shape)
# print(data.shape)
# print(len(data))
# president_y  = y[:,:,:1]
# congressmember_y  = y[:,:,1:5]
# print(president_y)
# print(congressmember_y)
# print(president_y.shape, congressmember_y.shape)