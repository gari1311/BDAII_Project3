# Name : Garima Gautam
# Roll No. : 180001
# Project3 SectionB

#Importing the libraries required 
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import warnings
warnings.filterwarnings('ignore')

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Loading the dataset

train_df = pd.read_csv(r"Titanic.csv")
test_df = pd.read_csv(r"test.csv")
combine = [train_df, test_df]
combine

print(train_df.columns.values)
print (test_df.columns.values)

# preview the data
train_df.head()


train_df.tail()
train_df.info()
print('_'*40)
test_df.info()

train_df.describe()

train_df.describe(include=['O'])

train_df[['pclass','survived']].groupby(['pclass'],as_index=False).mean()

train_df[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().sort_values(by='survived', ascending=False)

train_df[["sex", "survived"]].groupby(['sex'], as_index=False).mean().sort_values(by='survived', ascending=False)

train_df[["sibsp", "survived"]].groupby(['sibsp'], as_index=False).mean().sort_values(by='survived', ascending=False)

train_df[["parch", "survived"]].groupby(['parch'], as_index=False).mean().sort_values(by='parch',ascending=True)

train_df[['fare']].describe()

g = sns.FacetGrid(train_df, col='pclass')
g.map(plt.hist,'fare', bins=10)
g = sns.FacetGrid(train_df, col='survived')
g.map(plt.hist, 'age', bins=20)


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='survived', row='pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'pclass', 'survived','sex', palette='deep')
grid.add_legend()


grid = sns.FacetGrid(train_df, row='sex', col='survived', size=2.2, aspect=1.6)
grid.map(plt.hist,'age',bins=20)
grid.add_legend();


grid = sns.FacetGrid(train_df, row='embarked', col='survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'sex', 'fare', alpha=.5, ci=None)
grid.add_legend()


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['ticket', 'cabin'], axis=1)
test_df = test_df.drop(['ticket', 'cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

for dataset in combine:
    dataset['Title'] = dataset.name.str.extract(' ([A-Za-z]+)\.', expand=False)


pd.crosstab(train_df['Title'], train_df['sex'])


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

train_df['Title'].value_counts()


print(combine[0].head())

train_df = train_df.drop(['name', 'passengerId'], axis=1)
test_df = test_df.drop(['name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


train_df.shape
train_df.head()

for dataset in combine:
    dataset['sex'] = dataset['sex'].map( {'female': 1, 'male': 0}).astype(int)

print(train_df.head())

print (test_df.head())

grid = sns.FacetGrid(train_df, row='pclass', col='sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend()


guess_ages = np.zeros((2,3))
guess_ages


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['sex'] == i) & (dataset['pclass'] == j+1)]['age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.age.isnull()) & (dataset.sex == i) & (dataset.pclass == j+1),'age'] = guess_ages[i,j]

    dataset['age'] = dataset['age'].astype(int)

train_df.head()


train_df['AgeBand'] = pd.cut(train_df['age'], 5)
train_df[['AgeBand', 'survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['age'] <= 16, 'age'] = 0
    dataset.loc[(dataset['age'] > 16) & (dataset['age'] <= 32), 'age'] = 1
    dataset.loc[(dataset['age'] > 32) & (dataset['age'] <= 48), 'age'] = 2
    dataset.loc[(dataset['age'] > 48) & (dataset['age'] <= 64), 'age'] = 3
    dataset.loc[ dataset['age'] > 64, 'age']
train_df.head()


train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
test_df.head()

train_df.head()

test_df.head()


for dataset in combine:
    dataset['FamilySize'] = dataset['sibsp'] + dataset['parch'] + 1


train_df.head()
train_df[['FamilySize', 'survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='survived', ascending=False)

train_df.info()

train_df['FamilySize'].value_counts()


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df.head()
train_df[['IsAlone', 'survived']].groupby(['IsAlone'], as_index=False).mean()

dropped_one = train_df['parch']
dropped_two = train_df['sibsp']
dropped_three = train_df['FamilySize']
dropped_one

test_df.head()

combine = [train_df, test_df]

train_df.head()

for dataset in combine:
    dataset['age*Class'] = dataset.age * dataset.pclass

train_df.loc[:, ['age*Class', 'age', 'pclass']].head(10)


train_df['age*Class'].value_counts()

freq_port = train_df['embarked'].dropna().mode()[0]
freq_port

for dataset in combine:
    dataset['embarked'] = dataset['embarked'].fillna(freq_port)
    
train_df[['embarked', 'survived']].groupby(['embarked'], as_index=False).mean().sort_values(by='survived', ascending=False)


for dataset in combine:
    dataset['embarked'] = dataset['embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()

test_df['fare'].fillna(test_df['fare'].dropna().median(), inplace=True)
test_df.head()


train_df['FareBand'] = pd.qcut(train_df['fare'], 4)
train_df[['FareBand', 'survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


for dataset in combine:
    dataset.loc[ dataset['fare'] <= 7.91, 'fare'] = 0
    dataset.loc[(dataset['fare'] > 7.91) & (dataset['fare'] <= 14.454), 'fare'] = 1
    dataset.loc[(dataset['fare'] > 14.454) & (dataset['fare'] <= 31), 'fare']   = 2
    dataset.loc[ dataset['fare'] > 31, 'fare'] = 3
    dataset['fare'] = dataset['fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


test_df.head(10)


copy_df=train_df.copy()
copyTest_df=test_df.copy()


from sklearn.preprocessing import OneHotEncoder



train_Embarked = copy_df["embarked"].values.reshape(-1,1)
test_Embarked = copyTest_df["embarked"].values.reshape(-1,1)


onehot_encoder = OneHotEncoder(sparse=False)
train_OneHotEncoded = onehot_encoder.fit_transform(train_Embarked)
test_OneHotEncoded = onehot_encoder.fit_transform(test_Embarked)


copy_df["EmbarkedS"] = train_OneHotEncoded[:,0]
copy_df["EmbarkedC"] = train_OneHotEncoded[:,1]
copy_df["EmbarkedQ"] = train_OneHotEncoded[:,2]
copyTest_df["EmbarkedS"] = test_OneHotEncoded[:,0]
copyTest_df["EmbarkedC"] = test_OneHotEncoded[:,1]
copyTest_df["EmbarkedQ"] = test_OneHotEncoded[:,2]

copy_df.head()
copyTest_df.head()
train_df.head()
test_df.head()




X_trainTest = copy_df.drop(copy_df.columns[[0,5]],axis=1)
Y_trainTest = copy_df["survived"]
X_testTest = copyTest_df.drop(copyTest_df.columns[[0,5]],axis=1)
X_trainTest.head()


X_testTest.head()


logReg = LogisticRegression()
logReg.fit(X_trainTest,Y_trainTest)
acc = logReg.score(X_trainTest,Y_trainTest)
acc

X_train = train_df.drop("survived", axis=1)
Y_train = train_df["survived"]
X_test  = test_df.drop("passengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
X_train.head()



X_test.head()


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
acc_log = round(logReg.score(X_train, Y_train) * 100, 2)
acc_log


svcTest = SVC()
svcTest.fit(X_trainTest, Y_trainTest)
acc_svcTest = round(svcTest.score(X_trainTest, Y_trainTest)*100,2)
acc_svcTest


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


random_forestTest = RandomForestClassifier(n_estimators=100)
random_forestTest.fit(X_trainTest, Y_trainTest)
acc_random_forestTest = round(random_forestTest.score(X_trainTest, Y_trainTest) * 100, 2)
acc_random_forestTest



random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest



models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent',  
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian,
              acc_sgd,  acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
