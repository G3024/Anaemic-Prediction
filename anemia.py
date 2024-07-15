import pandas as pd

data = pd.read_csv("anemia.csv")

sex = data.Sex
sex = pd.DataFrame(sex)
data.drop(columns=['Sex'], inplace=True)
from sklearn.preprocessing import LabelEncoder
la_ = LabelEncoder()
for i in sex:
    sex[i] = la_.fit_transform(sex[i])

Anaemic = data.Anaemic
Anaemic = pd.DataFrame(Anaemic)
data.drop(columns=['Anaemic'], inplace=True)
from sklearn.preprocessing import LabelEncoder
la_ = LabelEncoder()
for i in Anaemic:
    Anaemic = la_.fit_transform(Anaemic[i])
Anaemic = pd.DataFrame({'Anaemic' : Anaemic})

data_fix = pd.concat([data, sex], axis='columns')


# split data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(data_fix, Anaemic, train_size=0.3, shuffle=True)


# modeling
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest)*100)

# testing
print(xtest) 
'''
Red Pixel  Green pixel  Blue pixel    Hb  Sex
80    43.8783      29.4752     26.6465  14.1    0
18    45.7104      27.5693     26.7204  13.6    0
15    45.5842      28.7311     25.6848  14.0    1
54    46.0477      28.6476     25.3047  14.4    1
95    52.1540      26.0853     21.7607  16.0    1
..        ...          ...         ...   ...  ...
19    40.9365      31.9687     27.0948   9.9    0
50    45.9659      28.4015     25.6326  13.0    1
10    45.3506      29.1248     25.5246  12.6    0
65    45.4201      29.4684     25.1115  12.0    0
42    46.7628      28.0180     25.2192  15.2    1
'''
print(model.predict(xtest))

'''
[0 0 0 0 0 1 1 1 0 1 0 0 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 1 1 1 0 1 1 1 0 0
 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0]
'''