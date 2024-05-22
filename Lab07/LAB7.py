import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('Dataset.txt')
dataset.head()
X = dataset.iloc[:,-1:].values
y = dataset.iloc[:, 6].values
print ("Матрица признаков"); print(X[:5])
print ("Зависимая переменная"); print(y[:5])


labelencoder_y = LabelEncoder()
print("Зависимая переменная до обработки")
print(y)
y = labelencoder_y.fit_transform(y)
print("Зависимая переменная после обработки")
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(y_pred)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Наличие симкарты VS Финальная стоимость')
plt.xlabel('Финальная стоимость')
plt.ylabel('Наличие симкарты')
plt.show()