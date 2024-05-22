from __future__ import division, print_function
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
ds_anime = pd.read_csv('dataanime.csv')
test_anime = pd.read_csv('dataanime.csv', nrows=10)

feats = ['Title', 'Licensors', 'Studios', 'Sources']

# Вычисление размера обучающей подвыборки (70% от размера исходного набора данных)
train_size = int(0.7 * ds_anime.shape[0])

# Вывод информации о размере исходного набора данных и размере обучающей подвыборки
print('Размер исходного набора: ', len(ds_anime), \
      '\nРазмер обучающей подвыборки: ', train_size)

# Выделение признаков и целевой переменной для обучающей и тестовой выборок
X, y = ds_anime.loc[:, feats], ds_anime['Score']
X_test = test_anime.loc[:, feats]
X_train, X_valid = X.iloc[:train_size, :], X.iloc[train_size:, :]
y_train, y_valid = y.iloc[:train_size], y.iloc[train_size:]

from sklearn.feature_extraction.text import TfidfVectorizer
# Инициализация TfidfVectorizer с заданными параметрами
vectorizer_title = TfidfVectorizer(min_df=3, max_df=0.3, ngram_range=(1,3))
# Обучение TfidfVectorizer на обучающих данных и вывод размера словаря
vX_train_title = vectorizer_title.fit(X_train['Title'])
print('vX_train_title.vocabulary_: ', len(vX_train_title.vocabulary_))

# Применение обученного TfidfVectorizer к валидационным данным и вывод размера словаря
vX_valid_title = vectorizer_title.fit(X_valid['Title'])
print('vX_train_title.vocabulary_: ', len(vX_valid_title.vocabulary_))

# Применение обученного TfidfVectorizer к тестовым данным и вывод размера словаря
vX_test_title = vectorizer_title.fit(X_test['Title'])
print('vX_train_title.vocabulary_: ', len(vX_test_title.vocabulary_))

# Преобразование обучающих данных с помощью обученного TfidfVectorizer и вывод размера полученной матрицы
X_train_title = vectorizer_title.fit_transform(X_train['Title'])
print('X_train_title.shape: ', X_train_title.shape)

# Преобразование валидационных данных с помощью обученного TfidfVectorizer и вывод размера полученной матрицы
X_valid_title = vectorizer_title.transform(X_valid['Title'])
print('X_valid_title.shape: ', X_valid_title.shape)

# Преобразование тестовых данных с помощью обученного TfidfVectorizer и вывод размера полученной матрицы
X_test_title = vectorizer_title.transform(X_test['Title'])
print('X_test_title.shape: ', X_test_title.shape)

vectorizer_title_ch = TfidfVectorizer(analyzer='char')

vX_train_title_ch = vectorizer_title_ch.fit(X_train['Title'])
print('vX_train_title_ch.vocabulary: ', len(vX_train_title_ch.vocabulary_))
vX_valid_title_ch = vectorizer_title_ch.fit(X_valid['Title'])
print('vX_valid_title_ch.vocabulary_: ', len(vX_valid_title_ch.vocabulary_))
vX_test_title_ch = vectorizer_title_ch.fit(X_test['Title'])
print('vX_test_title_ch.vocabulary_: ', len(vX_test_title_ch.vocabulary_))

X_train_title_ch = vectorizer_title_ch.transform(X_train['Title'])
print('X_train_title_ch.shape: ', X_train_title_ch.shape)
X_valid_title_ch = vectorizer_title_ch.transform(X_valid['Title'])
print('X_valid_title_ch.shape: ', X_valid_title_ch.shape)
X_test_title_ch = vectorizer_title_ch.transform(X_test['Title'])
print('X_test_title_ch.shape: ', X_test_title_ch.shape)

from sklearn.feature_extraction import DictVectorizer
vectorizer_feats = DictVectorizer()

tmp_dict_train = X_train[feats].fillna('-').T.to_dict().values()
tmp_dict_valid = X_valid[feats].fillna('-').T.to_dict().values()
tmp_dict_test = X_test[feats].fillna('-').T.to_dict().values()

X_train_feats = vectorizer_feats.fit_transform(tmp_dict_train)
X_valid_feats = vectorizer_feats.transform(tmp_dict_valid)
X_test_feats = vectorizer_feats.transform(tmp_dict_test)
print(X_train_feats.shape)

print(X_valid_feats.shape)

print(X_test_feats.shape)
X_train_new = scipy.sparse.hstack([X_train_title, X_train_feats, X_train_title_ch])

X_valid_new = scipy.sparse.hstack([X_valid_title, X_valid_feats, X_valid_title_ch])

X_test_new = scipy.sparse.hstack([X_test_title, X_test_feats, X_test_title_ch])

print(X_train_new.shape)

print(X_valid_new.shape)

print(X_test_new.shape)

model_1 = Ridge(alpha=.1, random_state=1)
model_1.fit(X_train_new, y_train)

train_preds1 = model_1.predict(X_train_new)
valid_preds1 = model_1.predict(X_valid_new)

print('Ощибка на трейне:', mean_squared_error(y_train, train_preds1))
print('Ошибка на тесте:', mean_squared_error(y_valid, valid_preds1))

model_2 = Ridge(alpha=1.8, random_state=1)
model_2.fit(X_train_new, y_train)

train_preds2 = model_2.predict(X_train_new)
valid_preds2 = model_2.predict(X_valid_new)

print('Ощибка на трейне:', mean_squared_error(y_train, train_preds2))
print('Ошибка на тесте:', mean_squared_error(y_valid, valid_preds2))