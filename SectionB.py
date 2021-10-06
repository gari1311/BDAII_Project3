# In[1]:

#Importing the libraries required 
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

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


# In[2]:
# Loading the dataset

train_df = pd.read_csv(r"Titanic.csv")
test_df = pd.read_csv(r"test.csv")
combine = [train_df, test_df]
combine


# In[3]:


print(train_df.columns.values)
print (test_df.columns.values)


# In[4]:


# preview the data
train_df.head()


# In[5]:


train_df.tail()


# In[6]:


train_df.info()
print('_'*40)
test_df.info()


# In[7]:


train_df.describe()


# In[8]:


train_df.describe(include=['O'])


# In[9]:


train_df[['pclass','survived']].groupby(['pclass'],as_index=False).mean()


# In[10]:


train_df[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[11]:


train_df[["sex", "survived"]].groupby(['sex'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[12]:


train_df[["sibsp", "survived"]].groupby(['sibsp'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[13]:


train_df[["parch", "survived"]].groupby(['parch'], as_index=False).mean().sort_values(by='parch',ascending=True)


# In[14]:


train_df[['fare']].describe()


# In[15]:


g = sns.FacetGrid(train_df, col='pclass')
g.map(plt.hist,'fare', bins=10)


# In[16]:


g = sns.FacetGrid(train_df, col='survived')
g.map(plt.hist, 'age', bins=20)


# In[17]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='survived', row='pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();


# In[20]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'pclass', 'survived','sex', palette='deep')
grid.add_legend()


# In[22]:


grid = sns.FacetGrid(train_df, row='sex', col='survived', size=2.2, aspect=1.6)
grid.map(plt.hist,'age',bins=20)
grid.add_legend();


# In[23]:


grid = sns.FacetGrid(train_df, row='embarked', col='survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'sex', 'fare', alpha=.5, ci=None)
grid.add_legend()


# In[24]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['ticket', 'cabin'], axis=1)
test_df = test_df.drop(['ticket', 'cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape


# In[25]:


for dataset in combine:
    dataset['Title'] = dataset.name.str.extract(' ([A-Za-z]+)\.', expand=False)


pd.crosstab(train_df['Title'], train_df['sex'])


# In[26]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'survived']].groupby(['Title'], as_index=False).mean()


# In[27]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()


# In[28]:


train_df['Title'].value_counts()


# In[29]:


print(combine[0].head())


# In[30]:


train_df = train_df.drop(['name', 'passengerId'], axis=1)
test_df = test_df.drop(['name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# In[31]:


train_df.shape
train_df.head()


# In[32]:


for dataset in combine:
    dataset['sex'] = dataset['sex'].map( {'female': 1, 'male': 0}).astype(int)

print(train_df.head())


# In[33]:


print (test_df.head())


# In[34]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='pclass', col='sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend()


# In[35]:


guess_ages = np.zeros((2,3))
guess_ages


# In[36]:


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


# In[37]:


train_df['AgeBand'] = pd.cut(train_df['age'], 5)
train_df[['AgeBand', 'survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[38]:


for dataset in combine:    
    dataset.loc[ dataset['age'] <= 16, 'age'] = 0
    dataset.loc[(dataset['age'] > 16) & (dataset['age'] <= 32), 'age'] = 1
    dataset.loc[(dataset['age'] > 32) & (dataset['age'] <= 48), 'age'] = 2
    dataset.loc[(dataset['age'] > 48) & (dataset['age'] <= 64), 'age'] = 3
    dataset.loc[ dataset['age'] > 64, 'age']
train_df.head()


# In[39]:


train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
test_df.head()


# In[40]:


train_df.head()


# In[41]:


test_df.head()


# In[42]:


for dataset in combine:
    dataset['FamilySize'] = dataset['sibsp'] + dataset['parch'] + 1


# In[43]:


train_df.head()
train_df[['FamilySize', 'survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[44]:


train_df.info()


# In[45]:


train_df['FamilySize'].value_counts()


# In[46]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


# In[47]:


train_df.head()
train_df[['IsAlone', 'survived']].groupby(['IsAlone'], as_index=False).mean()


# In[48]:


dropped_one = train_df['parch']
dropped_two = train_df['sibsp']
dropped_three = train_df['FamilySize']
dropped_one


# In[49]:


test_df.head()


# In[50]:


combine = [train_df, test_df]

train_df.head()


# In[51]:


for dataset in combine:
    dataset['age*Class'] = dataset.age * dataset.pclass

train_df.loc[:, ['age*Class', 'age', 'pclass']].head(10)


# In[53]:


train_df['age*Class'].value_counts()


# In[54]:


freq_port = train_df['embarked'].dropna().mode()[0]
freq_port


# In[55]:


for dataset in combine:
    dataset['embarked'] = dataset['embarked'].fillna(freq_port)
    
train_df[['embarked', 'survived']].groupby(['embarked'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[56]:


for dataset in combine:
    dataset['embarked'] = dataset['embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# In[57]:


test_df['fare'].fillna(test_df['fare'].dropna().median(), inplace=True)
test_df.head()


# In[58]:


train_df['FareBand'] = pd.qcut(train_df['fare'], 4)
train_df[['FareBand', 'survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[59]:


for dataset in combine:
    dataset.loc[ dataset['fare'] <= 7.91, 'fare'] = 0
    dataset.loc[(dataset['fare'] > 7.91) & (dataset['fare'] <= 14.454), 'fare'] = 1
    dataset.loc[(dataset['fare'] > 14.454) & (dataset['fare'] <= 31), 'fare']   = 2
    dataset.loc[ dataset['fare'] > 31, 'fare'] = 3
    dataset['fare'] = dataset['fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


# In[60]:


test_df.head(10)


# In[61]:


copy_df=train_df.copy()
copyTest_df=test_df.copy()


# In[62]:


from sklearn.preprocessing import OneHotEncoder


# In[64]:


train_Embarked = copy_df["embarked"].values.reshape(-1,1)
test_Embarked = copyTest_df["embarked"].values.reshape(-1,1)


# In[65]:


onehot_encoder = OneHotEncoder(sparse=False)
train_OneHotEncoded = onehot_encoder.fit_transform(train_Embarked)
test_OneHotEncoded = onehot_encoder.fit_transform(test_Embarked)


# In[66]:


copy_df["EmbarkedS"] = train_OneHotEncoded[:,0]
copy_df["EmbarkedC"] = train_OneHotEncoded[:,1]
copy_df["EmbarkedQ"] = train_OneHotEncoded[:,2]
copyTest_df["EmbarkedS"] = test_OneHotEncoded[:,0]
copyTest_df["EmbarkedC"] = test_OneHotEncoded[:,1]
copyTest_df["EmbarkedQ"] = test_OneHotEncoded[:,2]


# In[67]:


copy_df.head()


# In[68]:


copyTest_df.head()


# In[69]:


train_df.head()


# In[70]:


test_df.head()


# In[71]:


X_trainTest = copy_df.drop(copy_df.columns[[0,5]],axis=1)
Y_trainTest = copy_df["survived"]
X_testTest = copyTest_df.drop(copyTest_df.columns[[0,5]],axis=1)
X_trainTest.head()


# In[72]:


X_testTest.head()


# In[73]:


logReg = LogisticRegression()
logReg.fit(X_trainTest,Y_trainTest)
acc = logReg.score(X_trainTest,Y_trainTest)
acc

# In[75]:


X_train = train_df.drop("survived", axis=1)
Y_train = train_df["survived"]
X_test  = test_df.drop("passengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
X_train.head()


# In[76]:


X_test.head()


# In[77]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[78]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[79]:


svcTest = SVC()
svcTest.fit(X_trainTest, Y_trainTest)
acc_svcTest = round(svcTest.score(X_trainTest, Y_trainTest)*100,2)
acc_svcTest


# In[80]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[81]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[82]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[83]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[84]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[85]:


random_forestTest = RandomForestClassifier(n_estimators=100)
random_forestTest.fit(X_trainTest, Y_trainTest)
acc_random_forestTest = round(random_forestTest.score(X_trainTest, Y_trainTest) * 100, 2)
acc_random_forestTest


# In[86]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[87]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent',  
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian,
              acc_sgd,  acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
