import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

data_path = "Dataset.data"

dt = np.dtype("f8, f8, f8, f8, f8, f8, U30")
data2 = np.genfromtxt(data_path, delimiter=",", dtype=dt)
print('Shape of the dataset:', data2.shape)
print('Dataset type:', type(data2))
print('A single row of the dataset is type of:', type(data2[0]))
print('Types of elements:', type(data2[0][1]), type(data2[0][6]))
print('Dataset slice:')
print(data2[:10])

# Данные из отдельных столбцов
SC5 = [] # SC5
SP6 = [] # SP6
SHBd = [] # SHBd
minHaaCH = [] # minHaaCH
maxwHBa = [] # maxwHBa
FMF = [] # FMF

# Выполняем обход всей коллекции data2
for dot in data2:
    SC5.append(dot[0])
    SP6.append(dot[1])
    SHBd.append(dot[2])
    minHaaCH.append(dot[3])
    maxwHBa.append(dot[4])
    FMF.append(dot[5])

# Строим графики по проекциям данных
# Учитываем, что каждые 50 типов  идут последовательно
plt.figure(1)
High_BFE, = plt.plot(SC5[:50], FMF[:50], 'ro', label='High_BFE')
Low_BFE, = plt.plot(SC5[50:100], FMF[50:100], 'g^', label='Low_BFE')
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
plt.xlabel('SC5')
plt.ylabel('FMF')

plt.figure(2)
High_BFE, = plt.plot(minHaaCH[:50], FMF[:50], 'ro', label='High_BFE')
Low_BFE, = plt.plot(minHaaCH[50:100], FMF[50:100], 'g^', label='Low_BFE')
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
plt.xlabel('minHaaCH')
plt.ylabel('FMF')

plt.figure(3)
High_BFE, = plt.plot(maxwHBa[:50], FMF[:50], 'ro', label='High_BFE')
Low_BFE, = plt.plot(maxwHBa[50:100], FMF[50:100], 'g^', label='Low_BFE')
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
plt.xlabel('maxwHBa')
plt.ylabel('FMF')


plt.show()