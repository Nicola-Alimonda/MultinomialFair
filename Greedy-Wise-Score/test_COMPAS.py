import Greedy_Wise_Score
import numpy
import seaborn as sns
import json
import pandas as pd
import matplotlib.pyplot as plt
import ast

json_data = open('.\compas.json').read()
data = numpy.array(json_data.split("\n"))
data_set = []


## Clean the data
for i in data:
    data_set.append(ast.literal_eval(i)) ## Read the data
## Read the data
data = sorted(data_set, reverse=True,key = lambda user: float(user['Recidivism']))


for diz, i in zip(data, range(len(data))):
    diz['k'] = i + 1

cb_rank = pd.DataFrame(data)
cb_rank['DCG_PreR'] = cb_rank['Recidivism']/(numpy.log(2 + cb_rank['k']))

attribute = {'Group':4}

#initialization
alpha = float(0.05)
p = [0.66,0.11,0.11,0.12]
k_th = 201
attributeQuality = 'Recidivism'



ranking = Greedy_Wise_Score.GWS(data, p, alpha, k_th, attributeQuality, attribute)


data_ = pd.DataFrame(ranking)
data_['new_k'] = list(range(1,k_th))
data_['PreR_DCG'] = data_['Recidivism']/ (numpy.log(2 + data_['k']))
data_['PostR_DCG'] = data_['Recidivism']/ (numpy.log(2 + data_['new_k']))
data_['Utility_Loss_individual'] = data_['PostR_DCG'] - data_['PreR_DCG']
data_['Utility_Loss_position'] = data_['PostR_DCG'] - cb_rank['DCG_PreR']
data_['ULP*ULI'] = data_['Utility_Loss_position'].abs()*data_['Utility_Loss_individual'].abs() + data_['Utility_Loss_individual'].abs()/data_['Utility_Loss_position'].abs()


data_.to_csv('re_rank_COMPAS.csv', sep = ",", index=False)
sns.catplot(x="Group", y="Recidivism", kind="box", data=data_)
plt.savefig('.\Plots\COMPAS_PostRank_distribution.png')


sns.catplot(x="Group", y="Recidivism", kind="box", data=cb_rank.head(k_th-1))
plt.savefig('.\Plots\COMPAS_PreRank_distribution.png')

## Per questa metrica è meglio normalizzare lo Score tra 0 e 1

print(data_['ULP*ULI'].sum())
print(data_.groupby('Group')['Group'].count())
