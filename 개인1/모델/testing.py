import pandas as pd 
import numpy as np

data = pd.read_csv('c:/project/개인1/데이터/data.csv',thousands=',', 
                #    index_col=0
                )

pd.set_option('display.max_rows',None)
print(data)
#1966~84
print(data.iloc[:5,0:2])
# data.iloc[1:2,1:2] = np.mean(data.iloc[0:5,1:2])
# data.iloc[2:3,1:2] = np.mean(data.iloc[0:5,1:2])
# data.iloc[3:4,1:2] = np.mean(data.iloc[0:5,1:2])
data.loc[[1,2,3],['국세/단위 억원']] = data.loc[[1,2,3],['국세/단위 억원']].fillna(data.loc[[0,4],['국세/단위 억원']].mean())
print(data.iloc[:5,0:2])
print(data.iloc[4:15,0:2])
data.loc[[5,6,7,8,9,10,11,12,13],['국세/단위 억원']] = data.loc[[5,6,7,8,9,10,11,12,13],['국세/단위 억원']].fillna(data.loc[[4,14],['국세/단위 억원']].mean())
print(data.iloc[4:15,0:2])
print(data.iloc[14:20,0:2])
data.loc[[15,16,17,18],['국세/단위 억원']] = data.loc[[15,16,17,18],['국세/단위 억원']].fillna(data.loc[[14,19],['국세/단위 억원']].mean())
print(data.iloc[14:20,0:2])

print(data.loc[[57,58,59,60,61],['성장률','물가 상승률']])
data.loc[[58,59,60],['성장률','물가 상승률']] = data.loc[[58,59,60],['성장률','물가 상승률']].fillna(data.loc[[57,61],['성장률','물가 상승률']].mean())
print(data.loc[[57,58,59,60,61],['성장률','물가 상승률']])

print(data['대통령 투표율'])
data.iloc[0:1,5:6] = data.iloc[1:2,5:6]



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
#대통령 투표율 완료 국회의원 투표율
print(data.loc[:,['국회의원 투표율','민주당계','보수당계','진보당계','무소속']])
data.loc[[2,3,4],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[[2,3,4],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[1,5],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())

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

data.loc[55:,['국회의원 투표율','민주당계','보수당계','진보당계','무소속']] = data.loc[55:,['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].fillna(data.loc[[54],['국회의원 투표율','민주당계','보수당계','진보당계','무소속']].mean())


print(data.loc[:,['국회의원 투표율','민주당계','보수당계','진보당계','무소속']])