import pandas as pd 
import numpy as np

data = pd.read_csv('https://github.com/letcodesing/project/raw/main/%EA%B0%9C%EC%9D%B81/%EB%8D%B0%EC%9D%B4%ED%84%B0/data.csv',thousands=',', 
                #    index_col=0
                )

pd.set_option('display.max_rows',None)
print(data)
#국세수입
#1966~84
print(data.iloc[:5,0:2])
data.loc[1:3,['국세/단위 억원']] = data.loc[1:3,['국세/단위 억원']].fillna(data.loc[[0,4],['국세/단위 억원']].mean())
print(data.iloc[:5,0:2])
print(data.iloc[4:15,0:2])
data.loc[5:13,['국세/단위 억원']] = data.loc[5:13,['국세/단위 억원']].fillna(data.loc[[4,14],['국세/단위 억원']].mean())
print(data.iloc[4:15,0:2])
print(data.iloc[14:20,0:2])
data.loc[15:18,['국세/단위 억원']] = data.loc[15:18,['국세/단위 억원']].fillna(data.loc[[14,19],['국세/단위 억원']].mean())
print(data.iloc[14:20,0:2])
#2026~27
print(data.iloc[55:,0:2])
data.iloc[60:61,1:2] = data.iloc[59:60,1:2]
data.iloc[61:62,1:2] = data.iloc[60:61,1:2]
print(data.iloc[55:,0:2])

#성장률, 물가상승률 평균 채우기
print(data.loc[[57,58,59,60,61],['성장률','물가 상승률']])
data.loc[58:60,['성장률','물가 상승률']] = data.loc[58:60,['성장률','물가 상승률']].fillna(data.loc[[57,61],['성장률','물가 상승률']].mean())
print(data.loc[[57,58,59,60,61],['성장률','물가 상승률']])

#언론자유도
from sklearn import linear_model
reg_lin = linear_model.LinearRegression()
x = data.iloc[27:51,2:4]
y = data['언론자유도']
y = y.dropna()
reg_lin.fit(x,y)  
print(x.shape, y.shape)
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

# print(data['언론자유도'])
# print(data.columns)
print(data)

data.loc[data['언론자유도']<10,'언론자유도'] = 10
print(data)

#대통령 여/야
data.iloc[0:26,7:8] = 0
data.iloc[32:36,7:8] = 0
data.iloc[42:46,7:8] = 0

data.iloc[27:31,7:8] = 1
data.iloc[37:41,7:8] = 1
data.iloc[47:56,7:8] = 1
print(data)

# # data = data.interpolate(limit_direction='backward')
# # data = data.interpolate()
# print(data)

# # data = pd.DataFrame(data)
# # 예상대로 가장 가까운 두 수의 평균이므로 예상외의 값은 미리 채우고 적용



# # from sklearn.impute import KNNImputer
# # imputer=KNNImputer(n_neighbors=2)
# # data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
# # data_pre1 = data[['성장률', '물가 상승률']]
# # data_pre1 = pd.DataFrame(data_pre1)


# # from scipy import interpolate
# # pol = interpolate.interp1d()
# # print('knnimputer')
# # print(data)
# # print(data.isnull().sum())





#성장률기준 언론자유지수 
print(data)
#남은것 대통령,국회으원 투표율, y값 전부 평균채움 예정
print(data['대통령 투표율'])
data.iloc[0:1,5:6] = data.iloc[1:2,5:6]
data.iloc[2:3,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[3:4,5:6] = np.mean(data.iloc[1:6,5:6])
data.iloc[4:5,5:6] = np.mean(data.iloc[1:6,5:6])


print(data.iloc[5:22,5:6])


data.iloc[6:7,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[7:8,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[8:9,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[9:10,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[10:11,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[11:12,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[12:13,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[13:14,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[14:15,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[15:16,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[16:17,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[17:18,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[18:19,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[19:20,5:6] = np.mean(data.iloc[5:22,5:6])
data.iloc[20:21,5:6] = np.mean(data.iloc[5:22,5:6])
print(data.iloc[5:22,5:6])

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
data.iloc[57:58,5:6] = data.iloc[56:57,5:6]
data.iloc[58:59,5:6] = data.iloc[56:57,5:6]
data.iloc[59:60,5:6] = data.iloc[56:57,5:6]
data.iloc[60:61,5:6] = data.iloc[56:57,5:6]
data.iloc[61:62,5:6] = data.iloc[56:57,5:6]
print(data.iloc[56:62,5:6])
#대통령 투표율 완료 국회의원 투표율 및 분류별 국회의원 수
data.iloc[0:1,6:7] = data.iloc[1:2,6:7]
data.iloc[0:1,8:] = data.iloc[1:2,8:]
print(data.loc[:,['국회의원 투표율','민주당계','보수당계','진보당계','무소속']])

data.loc[[2,3,4],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = \
data.loc[[2,3,4],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(\
data.loc[[1,5],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[6],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[6],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[5,7],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[8,9,10,11],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[8,9,10,11],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[7,12],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[13,14],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[13,14],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[12,15],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[16,17,18],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[16,17,18],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[15,19],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[20,21],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[20,21],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[19,22],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[23,24,25],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[23,24,25],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[22,26],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[27,28,29],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[27,28,29],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[26,30],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[31,32,33],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[31,32,33],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[30,34],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[35,36,37],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[35,36,37],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[34,38],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[39,40,41],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[39,40,41],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[38,42],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[43,44,45],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[43,44,45],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[42,46],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[47,48,49],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[47,48,49],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[46,50],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[[51,52,53],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[51,52,53],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[50,54],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

data.loc[55:,['국회의원 투표율']] = data.loc[55:,['국회의원 투표율']].fillna(data.loc[[54],['국회의원 투표율']].mean())
data.loc[55:56,['민주당계','보수당계','진보당계','무소속']] = data.loc[55:56,['민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[54],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())


print(data.loc[:,['국회의원 투표율','민주당계','보수당계','진보당계','무소속']])
#x,y 완성 23~ 서브미션 그 이하 트레인
# 경제, 민주화로 앙상블모델
#y는 시계열로 자른다음 나오게 된다
#시계열 함수
#시계열 데이터를 자르면 현재까지의 정보로 미래 5개년을 예측하면 되기 때문에 x에 해당하는 23~27년 데이터는 필요없다
#넘파이로 저장했다가 다시 불러와서 하자

# from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,RobustScaler,StandardScaler,Normalizer
# Minscaler = MinMaxScaler()
# Maxscaler = MaxAbsScaler()
# Stanscaler = StandardScaler()
# Roscaler = RobustScaler()
# Norscaler = Normalizer()
# data['국세/단위 억원'] = Roscaler.fit_transform(data['국세/단위 억원'])
# # data['성장률','물가 상승률'] = Stanscaler.fit_transform(data['성장률','물가 상승률'])
# # data['언론자유도'] = Maxscaler.fit_transform(data['언론자유도'])
# # data['대통령 투표율','국회의원 투표율'] = Minscaler.fit_transform(data['대통령 투표율','국회의원 투표율'])
# #nan값 때문에 오류나는 듯 아닌듯?
# data[:,1:2] = Roscaler.fit_transform(data[:,1:2])
# data[:,2:4] = Stanscaler.fit_transform(data[:,2:4])
# data[:,4:5] = Maxscaler.fit_transform(data[:,4:5])
# data[:,6:8] = Minscaler.fit_transform(data[:,6:8])

print(data)
print(type(data))
data = data.values
print(type(data))
print(data.shape)


np.save('c:/project/개인1/npy, weight 저장/2/data.npy', arr=data)
"""







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
"""
