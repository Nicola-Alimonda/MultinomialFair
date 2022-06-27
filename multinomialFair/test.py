import numpy as np

json_data = open('.\german_credit.json').read()
data = np.array(json_data.split("\n"))
data_set = []

import matplotlib

import random
import scipy.stats as stats
import pandas as pd

import copy
import json
import ast
from anytree import PreOrderIter
import math
import itertools
import scipy.stats as stats
import matplotlib.pyplot as pl
from operator import itemgetter, attrgetter, methodcaller
from scipy.stats import poisson
from docx import Document
from docx.text.run import Font, Run
from docx.dml.color import ColorFormat
from docx.shared import RGBColor
from time import time
import blockCount


from multinomFair import multinomFair
from multinomFair import plot_scatter
import multinomial_icdf
import ranking_algorithm

from multinomFair import plot_scatter
from multinomFair import plot_scatter_pre_opt
## Clean the data
for i in data:
    data_set.append(ast.literal_eval(i))
attribute = {'Group':4}


#initialization
k = 50
p = [0.4,0.3,0.2,0.1]

#Sort Values
data_set = sorted(data_set, reverse=True,key = lambda user: (user['Score']))

## Generate color blind ranking
cb_rank = pd.DataFrame(data_set).sort_values('Score', ascending = False)
klist = np.array(range(1,len(cb_rank)+1))
# cb_rank['Score_Norm'] = (cb_rank.Score - min(cb_rank.Score)) / (max(cb_rank.Score) - min(cb_rank.Score))
# cb_rank = cb_rank.head(k)
cb_rank['k'] = klist
cb_rank['Utility'] = cb_rank['Score']*(1/np.log(2+cb_rank['k']))



# Generate fair rank
plot_scatter_pre_opt(data_set[0:k],{'Group':4}  , 'Score')

test_ranking = multinomFair(data_set, {'Group':4}, 'Score',k, p, float(0.1), False)

data_set = pd.DataFrame(data_set)
data_set.dropna(inplace=True)
data_set = data_set[data_set.k.notna()].sort_values('k')
data_set['Score_preNorm'] = cb_rank['Score']
data_set['Utility_UnNorm'] = data_set['Score_preNorm']*(1/np.log(2+data_set['k']))
data_set['old_k'] = cb_rank['k']

## Loss/Gain di utilità (derivante dalla posizione) tra il medesimo individuo
data_set['Utility_Loss_individual'] =  data_set['Utility_UnNorm'] - cb_rank['Utility']
data_set['Utility_Loss_individual_perc'] =  (data_set['Utility_UnNorm'] - cb_rank['Utility'])/(cb_rank['Utility'])*100

## Loss/Gain di utilità (derivante dall'individuo) tra la medesima posizione
shish = np.array(data_set.head(k).reset_index(drop=True)['Utility_UnNorm'] - cb_rank.head(k).reset_index(drop=True)['Utility'])
shish_perc = np.array((data_set.head(k).reset_index(drop=True)['Utility_UnNorm'] - cb_rank.head(k).reset_index(drop=True)['Utility'])/(cb_rank.head(k).reset_index(drop=True)['Utility'])*100)
data_set['Utility_Loss_position'] = shish
data_set['Utility_Loss_position_perc'] = shish_perc

data_set.to_html("Ranking/post_opt_dataset.html")
data_set.groupby('Group')[['Utility_Loss_individual_perc','Utility_Loss_position_perc']].describe().to_html('Ranking/post_opt_dataset_aggregated_by_class.html')
