import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

import seaborn as sb

df = pd.read_csv('titanic dataset.csv')

# Section A
# Ques1 : Find out the overall chance of survival for a Titanic passenger
print('Chance of survival for a Titanic passenger: ')
print(df.survived.mean() * 100, '%')

print()

# Ques2 : Find out the chance of survival for a Titanic passenger based on their sex and plot it
print('Chance of survival based on sex: ')
result = df[['sex', 'survived']].groupby('sex').mean() * 100
print(result)

daf = DataFrame(result, columns=['sex', 'survived'])
daf.plot.pie(y='survived', figsize=(8, 5), autopct='%1.1f%%', startangle=90)
plt.title('Chance of survival for a Titanic passenger based on their sex')
plt.show()

print()

# Ques3 : Find out the chance of survival for a Titanic passenger by traveling class wise and plot it.
print('Chance of survival based on class: ')
result = df[['pclass', 'survived']].groupby('pclass').mean() * 100
print(result)

print()

daf3 = DataFrame(result, columns=['pclass', 'survived'])
daf3.plot.pie(y='survived', figsize=(8, 5), autopct='%1.1f%%', startangle=90)
plt.title('Chance of survival for a Titanic passenger by traveling class wise')
plt.show()

# Ques4: Find out the average age for a Titanic passenger who survived by passenger class and sex.
print('Average age for a Titanic passenger who survived by passenger class and sex: ')
print(df.groupby(['pclass', 'sex'])['age'].mean())

print()

# Ques5: Find out the chance of survival for a Titanic passenger based on number of siblings the passenger had on the ship
# and plot it.
print('Chance of survival based on the no. of siblings the passenger had on the ship: ')
result = df[['sibsp', 'survived']].groupby('sibsp').mean() * 100
print(result)

daf4 = DataFrame(result, columns=['sibsp', 'survived'])
daf4.plot.pie(y='survived', figsize=(8, 5), autopct='%1.1f%%', startangle=90)
plt.title('Chance of survival for a Titanic passenger based on number of \n siblings the passenger had on the ship')
plt.show()

print()

# Ques6: Find out the chance of survival for a Titanic passenger based on number of parents/children the passenger had on
# the ship and plot it.
print('Chance of survival based on the no. of parents/children the passenger had on the ship: ')
result = df[['parch', 'survived']].groupby('parch').mean() * 100
print(result)

daf5 = DataFrame(result, columns=['parch', 'survived'])
daf5.plot.pie(y='survived', figsize=(8, 5), autopct='%1.1f%%', startangle=90)
plt.title('Chance of survival for a Titanic passenger based on the \nno. of parents/children the passenger had on the ship')

plt.show()

print()

# Ques7: Plot out the variation of survival and death amongst passengers of different age.
g = sb.FacetGrid(df, col="survived")
g.map(plt.hist, "age")
plt.show()

print()

# Ques8: Plot out the variation of survival and death with age amongst passengers of different passenger classes.
g = sb.FacetGrid(df, row="survived", col="pclass")
g.map(plt.hist, "age")
plt.show()

print()

# Ques9: Find out the survival probability for a Titanic passenger based on title from the name of passenger.
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

print()

# Ques10: What conclusions are you derived from the analysis?
print("Conclusions: \n" + "1. Females have a much higher chance of survival than males."+
    "\n2. People with higher socioeconomic class had a higher rate of survival"+
    "\n3. People with no siblings or spouses were less to likely to survive than those with one or two"+
    "\n4. People traveling alone are less likely to survive than those with 1-3 parents or children."+
    "\n5. Babies are more likely to survive than any other age group."+
    "\n6. People with a recorded Cabin number are, in fact, more likely to survive.")
