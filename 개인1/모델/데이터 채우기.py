import pandas as pd 
import numpy as np

data = pd.read_csv('c:/project/개인1/data.csv',thousands=',', index_col=0)
# print(data)

# data.fillna({'성장률':2, '물가 상승률':2}, inplace=True)
# data = pd.DataFrame(data)
#성장률 평균 채우기
print(data)
print(data.shape)
print(data.iloc[57:62,:3])
data.iloc[58:59,2:3] = np.mean(data.iloc[57:62,2:3])
data.iloc[59:60,2:3] = np.mean(data.iloc[57:62,2:3])
data.iloc[60:61,2:3] = np.mean(data.iloc[57:62,2:3])
print(data.iloc[57:62,:3])

print(data.iloc[57:62,:4])
data.iloc[58:59,3:4] = np.mean(data.iloc[57:62,3:4])
data.iloc[59:60,3:4] = np.mean(data.iloc[57:62,3:4])
data.iloc[60:61,3:4] = np.mean(data.iloc[57:62,3:4])
print(data.iloc[57:62,:4])
# print(data.columns)
#성장률과 국세 회귀모델
# z = data[['성장률','물가 상승률']]
# x = data.loc[[0,4,14,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54],['성장률','물가 상승률']]
# z2 = z.drop(index=[0,4,14,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54])
# y = data['국세/단위 억원']
# y = y.dropna()
# reg_lin.fit(x,y)
# kukse_na = reg_lin.predict(z2)
# print(kukse_na)
# [   91378.70681411  -438013.33489744  -782318.93760303  -353002.1365925
#    261284.1666887     98905.01593543 -1302253.86646059 -1168668.10459667
#   -888351.91805199  -249555.88367417  -518857.81214493  -596553.72649224
#   -709445.66811112   566905.79317013   276231.27234948   754684.47228855
#   1589815.07721682  1916130.52394329  1916130.52394329  1916130.52394329
#   1916130.52394329  1916130.52394329  1916130.52394329]
#국세 회귀모델 성장률로 회귀예측을 했을때 값이 너무 낮게 나오므로 보류
print(data.iloc[:5,0:2])
data.iloc[1:2,1:2] = np.mean(data.iloc[0:5,1:2])
data.iloc[2:3,1:2] = np.mean(data.iloc[0:5,1:2])
data.iloc[3:4,1:2] = np.mean(data.iloc[0:5,1:2])
print(data.iloc[:5,0:2])
print(data.iloc[4:15,0:2])
data.iloc[5:6,1:2] = np.mean(data.iloc[4:15,1:2])
data.iloc[6:7,1:2] = np.mean(data.iloc[4:15,1:2])
data.iloc[7:8,1:2] = np.mean(data.iloc[4:15,1:2])
data.iloc[8:9,1:2] = np.mean(data.iloc[4:15,1:2])
data.iloc[9:10,1:2] = np.mean(data.iloc[4:15,1:2])
data.iloc[10:11,1:2] = np.mean(data.iloc[4:15,1:2])
data.iloc[11:12,1:2] = np.mean(data.iloc[4:15,1:2])
data.iloc[12:13,1:2] = np.mean(data.iloc[4:15,1:2])
data.iloc[13:14,1:2] = np.mean(data.iloc[4:15,1:2])
print(data.iloc[4:15,0:2])
print(data.iloc[14:20,0:2])
data.iloc[15:16,1:2] = np.mean(data.iloc[14:20,1:2])
data.iloc[16:17,1:2] = np.mean(data.iloc[14:20,1:2])
data.iloc[17:18,1:2] = np.mean(data.iloc[14:20,1:2])
data.iloc[18:19,1:2] = np.mean(data.iloc[14:20,1:2])
print(data.iloc[14:20,0:2])

#21~25 국세 연평균 4.7 성장
print(data.iloc[55:,0:2])
data.iloc[55:56,1:2] = data.iloc[54:55,1:2]/100*104.7
data.iloc[56:57,1:2] = data.iloc[55:56,1:2]/100*104.7
data.iloc[57:58,1:2] = data.iloc[56:57,1:2]/100*104.7
data.iloc[58:59,1:2] = data.iloc[57:58,1:2]/100*104.7
data.iloc[59:60,1:2] = data.iloc[58:59,1:2]/100*104.7

data.iloc[60:61,1:2] = data.iloc[59:60,1:2]
data.iloc[61:62,1:2] = data.iloc[60:61,1:2]
print(data.iloc[55:,0:2])

#성장률기준 언론자유지수 
print(data)
from sklearn import linear_model
reg_lin = linear_model.LinearRegression()
x = data.iloc[27:51,2:4]
y = data['언론자유도']
y = y.dropna()
print(x.shape, y.shape)
reg_lin.fit(x,y)  
# pressfree = reg_lin.predict(data.iloc[:27,2:4])
# print(pressfree.shape)
print(data['언론자유도'])
data.iloc[0:1,4:5] = reg_lin.predict(data.iloc[0:1,2:4])
data.iloc[1:2,4:5] = reg_lin.predict(data.iloc[1:2,2:4])
data.iloc[2:3,4:5] = reg_lin.predict(data.iloc[2:3,2:4])
data.iloc[3:4,4:5] = reg_lin.predict(data.iloc[3:4,2:4])
data.iloc[4:5,4:5] = reg_lin.predict(data.iloc[4:5,2:4])
data.iloc[5:6,4:5] = reg_lin.predict(data.iloc[5:6,2:4])
data.iloc[6:7,4:5] = reg_lin.predict(data.iloc[6:7,2:4])
data.iloc[7:8,4:5] = reg_lin.predict(data.iloc[7:8,2:4])
data.iloc[8:9,4:5] = reg_lin.predict(data.iloc[8:9,2:4])
data.iloc[9:10,4:5] = reg_lin.predict(data.iloc[9:10,2:4])
data.iloc[10:11,4:5] = reg_lin.predict(data.iloc[10:11,2:4])
data.iloc[11:12,4:5] = reg_lin.predict(data.iloc[11:12,2:4])
data.iloc[12:13,4:5] = reg_lin.predict(data.iloc[12:13,2:4])
data.iloc[13:14,4:5] = reg_lin.predict(data.iloc[13:14,2:4])
data.iloc[14:15,4:5] = reg_lin.predict(data.iloc[14:15,2:4])
data.iloc[15:16,4:5] = reg_lin.predict(data.iloc[15:16,2:4])
data.iloc[16:17,4:5] = reg_lin.predict(data.iloc[16:17,2:4])
data.iloc[17:18,4:5] = reg_lin.predict(data.iloc[17:18,2:4])
data.iloc[18:19,4:5] = reg_lin.predict(data.iloc[18:19,2:4])
data.iloc[19:20,4:5] = reg_lin.predict(data.iloc[19:20,2:4])
data.iloc[20:21,4:5] = reg_lin.predict(data.iloc[20:21,2:4])
data.iloc[21:22,4:5] = reg_lin.predict(data.iloc[21:22,2:4])
data.iloc[22:23,4:5] = reg_lin.predict(data.iloc[22:23,2:4])
data.iloc[23:24,4:5] = reg_lin.predict(data.iloc[23:24,2:4])
data.iloc[24:25,4:5] = reg_lin.predict(data.iloc[24:25,2:4])
data.iloc[25:26,4:5] = reg_lin.predict(data.iloc[25:26,2:4])
data.iloc[26:27,4:5] = reg_lin.predict(data.iloc[26:27,2:4])

data.iloc[51:52,4:5] = reg_lin.predict(data.iloc[51:52,2:4])
data.iloc[52:53,4:5] = reg_lin.predict(data.iloc[52:53,2:4])
data.iloc[53:54,4:5] = reg_lin.predict(data.iloc[53:54,2:4])
data.iloc[54:55,4:5] = reg_lin.predict(data.iloc[54:55,2:4])
data.iloc[55:56,4:5] = reg_lin.predict(data.iloc[55:56,2:4])
data.iloc[56:57,4:5] = reg_lin.predict(data.iloc[56:57,2:4])
data.iloc[57:58,4:5] = reg_lin.predict(data.iloc[57:58,2:4])
data.iloc[58:59,4:5] = reg_lin.predict(data.iloc[58:59,2:4])
data.iloc[59:60,4:5] = reg_lin.predict(data.iloc[59:60,2:4])
data.iloc[60:61,4:5] = reg_lin.predict(data.iloc[60:61,2:4])
data.iloc[61:62,4:5] = reg_lin.predict(data.iloc[61:62,2:4])

print(data['언론자유도'])
#남은것 대통령,국회으원 투표율, y값 전부 평균채움 예정
print(data['대통령 투표율'])
data.iloc[0:1,5:6] = data.iloc[1:2,5:6]

data.iloc[2:3,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[3:4,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[4:5,5:6] = np.mean(data.iloc[1:6,5:6])

print(data.iloc[6:21,5:6])
data.iloc[6:7,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[7:8,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[8:9,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[9:10,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[10:11,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[11:12,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[12:13,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[13:14,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[14:15,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[15:16,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[16:17,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[17:18,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[18:19,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[19:20,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[20:21,5:6] = np.mean(data.iloc[1:6,5:6])
print(data.iloc[6:21,5:6])

print(data.iloc[21:27,5:6])
data.iloc[22:23,5:6] = np.mean(data.iloc[21:27,5:6])
data.iloc[23:24,5:6] = np.mean(data.iloc[21:27,5:6])
data.iloc[24:25,5:6] = np.mean(data.iloc[21:27,5:6])
data.iloc[25:26,5:6] = np.mean(data.iloc[21:27,5:6])
print(data.iloc[21:27,5:6])

print(data.iloc[26:27,5:6])
data.iloc[27:28,5:6] = np.mean(data.iloc[26:32,5:6])
data.iloc[28:29,5:6] = np.mean(data.iloc[26:32,5:6])
data.iloc[29:30,5:6] = np.mean(data.iloc[26:32,5:6])
data.iloc[30:31,5:6] = np.mean(data.iloc[26:32,5:6])
print(data.iloc[26:32,5:6])

print(data.iloc[31:37,5:6])
data.iloc[32:33,5:6] = np.mean(data.iloc[31:37,5:6])
data.iloc[33:34,5:6] = np.mean(data.iloc[31:37,5:6])
data.iloc[34:35,5:6] = np.mean(data.iloc[31:37,5:6])
data.iloc[35:36,5:6] = np.mean(data.iloc[31:37,5:6])
print(data.iloc[31:37,5:6])

print(data.iloc[36:42,5:6])
data.iloc[37:38,5:6] = np.mean(data.iloc[36:42,5:6])
data.iloc[38:39,5:6] = np.mean(data.iloc[36:42,5:6])
data.iloc[39:40,5:6] = np.mean(data.iloc[36:42,5:6])
data.iloc[40:41,5:6] = np.mean(data.iloc[36:42,5:6])
print(data.iloc[36:42,5:6])

print(data.iloc[41:47,5:6])
data.iloc[42:43,5:6] = np.mean(data.iloc[41:47,5:6])
data.iloc[43:44,5:6] = np.mean(data.iloc[41:47,5:6])
data.iloc[44:45,5:6] = np.mean(data.iloc[41:47,5:6])
data.iloc[45:46,5:6] = np.mean(data.iloc[41:47,5:6])
print(data.iloc[41:47,5:6])

print(data.iloc[46:52,5:6])
data.iloc[47:48,5:6] = np.mean(data.iloc[46:52,5:6])
data.iloc[48:49,5:6] = np.mean(data.iloc[46:52,5:6])
data.iloc[49:50,5:6] = np.mean(data.iloc[46:52,5:6])
data.iloc[50:51,5:6] = np.mean(data.iloc[46:52,5:6])
print(data.iloc[46:52,5:6])

print(data.iloc[51:57,5:6])
data.iloc[52:53,5:6] = np.mean(data.iloc[51:57,5:6])
data.iloc[53:54,5:6] = np.mean(data.iloc[51:57,5:6])
data.iloc[54:55,5:6] = np.mean(data.iloc[51:57,5:6])
data.iloc[55:56,5:6] = np.mean(data.iloc[51:57,5:6])
print(data.iloc[51:57,5:6])

print(data.iloc[56:62,5:6])
data.iloc[57:58,5:6] = np.mean(data.iloc[56:62,5:6])
data.iloc[58:59,5:6] = np.mean(data.iloc[56:62,5:6])
data.iloc[59:60,5:6] = np.mean(data.iloc[56:62,5:6])
data.iloc[60:61,5:6] = np.mean(data.iloc[56:62,5:6])
data.iloc[61:62,5:6] = np.mean(data.iloc[56:62,5:6])
print(data.iloc[56:62,5:6])
#대통령 투표율 완료 국회의원 투표율

data.iloc[0:1,6:7] = data.iloc[1:2,6:7]
print(data['국회의원 투표율'])

data.iloc[2:3,6:7] = np.mean(data.iloc[1:6,6:7])
data.iloc[3:4,6:7] = np.mean(data.iloc[1:6,6:7])
data.iloc[4:5,6:7] = np.mean(data.iloc[1:6,6:7])
print(data.iloc[1:6,6:7])

print(data.iloc[5:8,6:7])
data.iloc[6:7,6:7] = np.mean(data.iloc[5:8,6:7])
print(data.iloc[5:8,6:7])

print(data.iloc[7:13,6:7])
data.iloc[8:9,6:7] = np.mean(data.iloc[7:13,6:7])
data.iloc[9:10,6:7] = np.mean(data.iloc[7:13,6:7])
data.iloc[10:11,6:7] = np.mean(data.iloc[7:13,6:7])
data.iloc[11:12,6:7] = np.mean(data.iloc[7:13,6:7])
print(data.iloc[7:13,6:7])

print(data.iloc[12:16,6:7])
data.iloc[13:14,6:7] = np.mean(data.iloc[12:16,6:7])
data.iloc[14:15,6:7] = np.mean(data.iloc[12:16,6:7])
print(data.iloc[12:16,6:7])

print(data.iloc[15:20,6:7])
data.iloc[16:17,6:7] = np.mean(data.iloc[15:20,6:7])
data.iloc[17:18,6:7] = np.mean(data.iloc[15:20,6:7])
data.iloc[18:19,6:7] = np.mean(data.iloc[15:20,6:7])
print(data.iloc[15:20,6:7])

print(data.iloc[19:23,6:7])
data.iloc[20:21,6:7] = np.mean(data.iloc[19:23,6:7])
data.iloc[21:22,6:7] = np.mean(data.iloc[19:23,6:7])
print(data.iloc[19:23,6:7])

print(data.iloc[22:27,6:7])
data.iloc[23:24,6:7] = np.mean(data.iloc[22:27,6:7])
data.iloc[24:25,6:7] = np.mean(data.iloc[22:27,6:7])
data.iloc[25:26,6:7] = np.mean(data.iloc[22:27,6:7])
print(data.iloc[22:27,6:7])

print(data.iloc[26:31,6:7])
data.iloc[27:28,6:7] = np.mean(data.iloc[26:31,6:7])
data.iloc[28:29,6:7] = np.mean(data.iloc[26:31,6:7])
data.iloc[29:30,6:7] = np.mean(data.iloc[26:31,6:7])
print(data.iloc[26:31,6:7])

print(data.iloc[30:35,6:7])
data.iloc[31:32,6:7] = np.mean(data.iloc[30:35,6:7])
data.iloc[32:33,6:7] = np.mean(data.iloc[30:35,6:7])
data.iloc[33:34,6:7] = np.mean(data.iloc[30:35,6:7])
print(data.iloc[30:35,6:7])

print(data.iloc[34:39,6:7])
data.iloc[35:36,6:7] = np.mean(data.iloc[34:39,6:7])
data.iloc[36:37,6:7] = np.mean(data.iloc[34:39,6:7])
data.iloc[37:38,6:7] = np.mean(data.iloc[34:39,6:7])
print(data.iloc[34:39,6:7])

print(data.iloc[38:43,6:7])
data.iloc[39:40,6:7] = np.mean(data.iloc[38:43,6:7])
data.iloc[40:41,6:7] = np.mean(data.iloc[38:43,6:7])
data.iloc[41:42,6:7] = np.mean(data.iloc[38:43,6:7])
print(data.iloc[38:43,6:7])

print(data.iloc[42:47,6:7])
data.iloc[43:44,6:7] = np.mean(data.iloc[42:47,6:7])
data.iloc[44:45,6:7] = np.mean(data.iloc[42:47,6:7])
data.iloc[45:46,6:7] = np.mean(data.iloc[42:47,6:7])
print(data.iloc[42:47,6:7])

print(data.iloc[46:51,6:7])
data.iloc[47:48,6:7] = np.mean(data.iloc[46:51,6:7])
data.iloc[48:49,6:7] = np.mean(data.iloc[46:51,6:7])
data.iloc[49:50,6:7] = np.mean(data.iloc[46:51,6:7])
print(data.iloc[46:51,6:7])

print(data.iloc[50:55,6:7])
data.iloc[51:52,6:7] = np.mean(data.iloc[50:55,6:7])
data.iloc[52:53,6:7] = np.mean(data.iloc[50:55,6:7])
data.iloc[53:54,6:7] = np.mean(data.iloc[50:55,6:7])
print(data.iloc[50:55,6:7])

print(data.iloc[54:,6:7])
data.iloc[55:56,6:7] = data.iloc[54:55,6:7]
data.iloc[56:57,6:7] = data.iloc[54:55,6:7]
data.iloc[57:58,6:7] = data.iloc[54:55,6:7]
data.iloc[58:59,6:7] = data.iloc[54:55,6:7]
data.iloc[59:60,6:7] = data.iloc[54:55,6:7]
data.iloc[60:61,6:7] = data.iloc[54:55,6:7]
data.iloc[61:,6:7] = data.iloc[54:55,6:7]
print(data.iloc[54:,6:7])

# 투표율 완료 y값

#대통령 여/야
data.iloc[0:26,7:8] = 0
data.iloc[32:36,7:8] = 0
data.iloc[42:46,7:8] = 0

data.iloc[27:31,7:8] = 1
data.iloc[37:41,7:8] = 1
data.iloc[47:56,7:8] = 1

#국회의원수
data.iloc[:1,8:] = data.iloc[1:2,8:]
# print(data)
np.mean(data.iloc[1:8,8:])
print(data.iloc[1:8,8:])
data.iloc[2:3,8:] = np.mean(data.iloc[1:8,8:])
data.iloc[3:4,8:] = np.mean(data.iloc[1:8,8:])
data.iloc[4:5,8:] = np.mean(data.iloc[1:8,8:])
data.iloc[5:6,8:] = np.mean(data.iloc[1:8,8:])
data.iloc[6:7,8:] = np.mean(data.iloc[1:8,8:])
print(data.iloc[1:8,8:])

print(data.iloc[7:13,8:])
data.iloc[8:9,8:] = np.mean(data.iloc[7:13,8:])
data.iloc[9:10,8:] = np.mean(data.iloc[7:13,8:])
data.iloc[10:11,8:] = np.mean(data.iloc[7:13,8:])
data.iloc[11:12,8:] = np.mean(data.iloc[7:13,8:])
print(data.iloc[7:13,8:])

print(data.iloc[12:16,8:])
data.iloc[13:14,8:] = np.mean(data.iloc[12:16,8:])
data.iloc[14:15,8:] = np.mean(data.iloc[12:16,8:])
print(data.iloc[12:16,8:])

print(data.iloc[15:20,8:])
data.iloc[16:17,8:] = np.mean(data.iloc[15:20,8:])
data.iloc[17:18,8:] = np.mean(data.iloc[15:20,8:])
data.iloc[18:19,8:] = np.mean(data.iloc[15:20,8:])
print(data.iloc[15:20,8:])

print(data.iloc[19:23,8:])
data.iloc[20:21,8:] = np.mean(data.iloc[19:23,8:])
data.iloc[21:22,8:] = np.mean(data.iloc[19:23,8:])
print(data.iloc[19:23,8:])

print(data.iloc[22:27,8:])
data.iloc[23:24,8:] = np.mean(data.iloc[22:27,8:])
data.iloc[24:25,8:] = np.mean(data.iloc[22:27,8:])
data.iloc[25:26,8:] = np.mean(data.iloc[22:27,8:])
print(data.iloc[22:27,8:])

print(data.iloc[26:31,8:])
data.iloc[27:28,8:] = np.mean(data.iloc[26:31,8:])
data.iloc[28:29,8:] = np.mean(data.iloc[26:31,8:])
data.iloc[29:30,8:] = np.mean(data.iloc[26:31,8:])
print(data.iloc[26:31,8:])

print(data.iloc[30:35,8:])
data.iloc[31:32,8:] = np.mean(data.iloc[30:35,8:])
data.iloc[32:33,8:] = np.mean(data.iloc[30:35,8:])
data.iloc[33:34,8:] = np.mean(data.iloc[30:35,8:])
print(data.iloc[30:35,8:])

print(data.iloc[34:39,8:])
data.iloc[35:36,8:] = np.mean(data.iloc[34:39,8:])
data.iloc[36:37,8:] = np.mean(data.iloc[34:39,8:])
data.iloc[37:38,8:] = np.mean(data.iloc[34:39,8:])
print(data.iloc[34:39,8:])

print(data.iloc[38:43,8:])
data.iloc[39:40,8:] = np.mean(data.iloc[38:43,8:])
data.iloc[40:41,8:] = np.mean(data.iloc[38:43,8:])
data.iloc[41:42,8:] = np.mean(data.iloc[38:43,8:])
print(data.iloc[38:43,8:])

print(data.iloc[42:47,8:])
data.iloc[43:44,8:] = np.mean(data.iloc[42:47,8:])
data.iloc[44:45,8:] = np.mean(data.iloc[42:47,8:])
data.iloc[45:46,8:] = np.mean(data.iloc[42:47,8:])
print(data.iloc[42:47,8:])

print(data.iloc[46:51,8:])
data.iloc[47:48,8:] = np.mean(data.iloc[46:51,8:])
data.iloc[48:49,8:] = np.mean(data.iloc[46:51,8:])
data.iloc[49:50,8:] = np.mean(data.iloc[46:51,8:])
print(data.iloc[46:51,8:])

print(data.iloc[50:55,8:])
data.iloc[51:52,8:] = np.mean(data.iloc[50:55,8:])
data.iloc[52:53,8:] = np.mean(data.iloc[50:55,8:])
data.iloc[53:54,8:] = np.mean(data.iloc[50:55,8:])
print(data.iloc[50:55,8:])

print(data.iloc[54:,8:])
data.iloc[55:57,8:] = data.iloc[54:55,8:]
print(data.iloc[54:,8:])

print(data.isnull().sum())

#x,y 완성 23~ 서브미션 그 이하 트레인
# 경제, 민주화로 앙상블모델
#y는 시계열로 자른다음 나오게 된다
#시계열 함수
#시계열 데이터를 자르면 현재까지의 정보로 미래 5개년을 예측하면 되기 때문에 x에 해당하는 23~27년 데이터는 필요없다
#넘파이로 저장했다가 다시 불러와서 하자

from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,RobustScaler,StandardScaler,Normalizer
Minscaler = MinMaxScaler()
Maxscaler = MaxAbsScaler()
Stanscaler = StandardScaler()
Roscaler = RobustScaler()
Norscaler = Normalizer()
data['국세/단위 억원'] = Roscaler.fit_transform(data['국세/단위 억원'])
data['성장률','물가 상승률'] = Stanscaler.fit_transform(data['성장률','물가 상승률'])
data['언론자유도'] = Maxscaler.fit_transform(data['언론자유도'])
data['대통령 투표율','국회의원 투표율'] = Minscaler.fit_transform(data['대통령 투표율','국회의원 투표율'])

print(data)
print(type(data))
data = data.values
print(type(data))
print(data.shape)

np.save('c:/project/개인1/data.npy', arr=data)








# econo_x = data.iloc[:-5,1:4]
# demo_x  = data.iloc[:-5,4:7]
# print(econo_x.head())
# print(econo_x.tail())
# print(demo_x.head())
# print(demo_x.tail())
# print(econo_x.shape, demo_x.shape)

# president_y = data.iloc[:-5,7:8]
# congressmember_y = data.iloc[:-5,8:]

# print(president_y.head())
# print(president_y.tail())
# print(congressmember_y.head())
# print(congressmember_y.tail())
# print(president_y.shape, congressmember_y.shape)

# print()
# econo2023 = data.iloc[-5:,1:4]
# demo2023 = data.iloc[-5:,4:7]

# print(econo2023)
# print(demo2023)

