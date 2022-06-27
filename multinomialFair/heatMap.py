
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing the data from csv
path='compas-scores-raw.csv'
data = pd.read_csv(path, delimiter=',')

#Generating the correlation heatmap
f, ax = plt.subplots(figsize=(15, 10)) 
corr = data.corr()
print corr
hm = sns.heatmap(corr, annot=True, ax=ax, cmap= "coolwarm", fmt='.2f',linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Generated Data Heatmap', fontsize=8)
f.savefig('Generated_Heatmap',pad_inches=1, bbox_inches='tight')

