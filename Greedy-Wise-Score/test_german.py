import Greedy_Wise_Score
import numpy
import seaborn as sns
import json
import pandas as pd
import ast
import matplotlib.pyplot as plt

json_data = open('.\german_credit.json').read()
data = numpy.array(json_data.split("\n"))
data_set = []
## Clean the data
for i in data:
    data_set.append(ast.literal_eval(i))## Read the data
data = sorted(data_set, reverse=True,key = lambda user: float(user['Score']))



for diz, i in zip(data, range(len(data))):
    diz['k'] = i + 1
cb_rank = pd.DataFrame(data)
cb_rank['DCG_PreR'] = cb_rank['Score']/(numpy.log(2 + cb_rank['k']))

attribute = {'Group':4}

#initialization
alpha = float(0.001)


p = [0.2,0.1,0.4,0.3]
k_th = 51

attributeQuality = 'Score'

ranking = Greedy_Wise_Score.GWS(data, p, alpha, k_th, attributeQuality, attribute)

data_ = pd.DataFrame(ranking)
data_['new_k'] = list(range(1,k_th))
data_['PreR_DCG'] = data_['Score']/ (numpy.log(2 + data_['k']))
data_['PostR_DCG'] = data_['Score']/ (numpy.log(2 + data_['new_k']))
data_['Utility_Loss_individual'] = data_['PostR_DCG'] - data_['PreR_DCG']
data_['Utility_Loss_position'] = data_['PostR_DCG'] - cb_rank['DCG_PreR']
data_['ULP*ULI'] = data_['Utility_Loss_position'].abs()*data_['Utility_Loss_individual'].abs() + data_['Utility_Loss_individual'].abs()/data_['Utility_Loss_position'].abs()

data_.to_csv('re_rank_German.csv', sep = ",", index=False)

sns.catplot(x="Group", y="Score", kind="box", data=data_)
plt.savefig('.\Plots\German_PostRank_distribution.png')

sns.catplot(x="Group", y="Score", kind="box", data=cb_rank.head(k_th-1))
plt.savefig('.\Plots\German_PreRank_distribution.png')

print(data_['ULP*ULI'].sum())
