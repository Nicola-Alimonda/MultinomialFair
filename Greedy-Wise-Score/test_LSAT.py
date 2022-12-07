import Greedy_Wise_Score
import numpy
import seaborn as sns
import json
import pandas as pd
import matplotlib.pyplot as plt

with open('LSAT.json') as file:
    data = json.load(file)
## Read the data
data = sorted(data, reverse=True,key = lambda user: float(user['LSAT']))


for diz, i in zip(data, range(len(data))):
    diz['k'] = i + 1

cb_rank = pd.DataFrame(data)
cb_rank['DCG_PreR'] = cb_rank['LSAT']/(numpy.log(2 + cb_rank['k']))

attribute = {'race_sex':16}

#initialization
alpha = float(0.00001)
p = [0.1,0.05, 0.03, 0.02, 0.01, 0.09, 0.07, 0.03, 0.2, 0.04, 0.06, 0.02,0.08, 0.1, 0.05, 0.05]
k_th = 301
attributeQuality = 'LSAT'



ranking = Greedy_Wise_Score.GWS(data, p, alpha, k_th, attributeQuality, attribute)


data_ = pd.DataFrame(ranking)
data_['new_k'] = list(range(1,k_th))
data_['PreR_DCG'] = data_['LSAT']/ (numpy.log(2 + data_['k']))
data_['PostR_DCG'] = data_['LSAT']/ (numpy.log(2 + data_['new_k']))
data_['Utility_Loss_individual'] = data_['PostR_DCG'] - data_['PreR_DCG']
data_['Utility_Loss_position'] = data_['PostR_DCG'] - cb_rank['DCG_PreR']
data_['ULP*ULI'] = data_['Utility_Loss_position'].abs()*data_['Utility_Loss_individual'].abs() + data_['Utility_Loss_individual'].abs()/data_['Utility_Loss_position'].abs()


data_.to_csv('re_rank_LSAT.csv', sep = ",", index=False)
sns.catplot(x="race_sex", y="LSAT", kind="box", data=data_)
plt.savefig('.\Plots\LSAT_PostRank_distribution.png')


sns.catplot(x="race_sex", y="LSAT", kind="box", data=cb_rank.head(k_th-1))
plt.savefig('.\Plots\LSAT_PreRank_distribution.png')

## Per questa metrica è meglio normalizzare lo Score tra 0 e 1

print(data_['ULP*ULI'].sum())
