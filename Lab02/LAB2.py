import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Smartphone,Brand,Model,RAM,Storage,Color,Free,Final Price
data_path = "Dataset.txt"
data = pd.read_csv(data_path)
data.head(10)
data.info()
plt.figure(1)
feats = ['Final Price','RAM','Storage']
sns.heatmap(data[feats].corr(), cmap=plt.cm.PuBuGn);
# c = data['Free'].map({"Yes": 'lightblue', "No": 'orange'})
# edge_c = data['Free'].map({"Yes": 'blue', "No": 'red'})
# plt.scatter(data['Final Price'],
#             data['Storage'],
#             color=c, edgecolors=edge_c)
# plt.xlabel('Финальная цена')
# plt.ylabel('Хранилище')
# plt.title('Распределение по 2 признакам');
#data[feats].hist(figsize=(5,5));
#sns.pairplot(data[feats + ['Free']], hue='Free');
#sns.countplot(data[data['Brand'].isin(data['Brand'].value_counts().tail(40).index)]['Brand']);
#sns.countplot(data['Free']);
# top_data = data[['Brand','Final Price']]D
# top_data = top_data.groupby('Brand').sum()
# top_data = top_data.sort_values('Final Price',ascending=False)
# top_data = top_data[:20].index.values
# sns.boxplot(y='Brand',
#            x='Final Price',
#            data=data[data.Brand.isin(top_data)], palette='Set2');
# sns.boxplot(data['Final Price']);
# hist = data['Final Price'].value_counts()
# plt.bar(hist.index, hist);
# plt.bar(data.index, data['Final Price'])
# data['Final Price'].hist();
# plt.figure(2)
# data['RAM'].hist();
# plt.figure(3)
# data['Storage'].hist();

plt.show()
