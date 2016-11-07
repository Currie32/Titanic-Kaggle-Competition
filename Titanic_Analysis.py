
# coding: utf-8

# This is my analysis of the Titanic dataset as supplied by Kaggle.com for their 'Titanic: Machine Learning from Disaster' competition. My goal was to score above 0.8, and I am pleased to say that I have accomplished this, as I have reached the score of 0.80861, which puts me in the top 9% of entries.

# The sections for this analysis are: data exploration, feature engineering, machine learning, and evaluation.

# In[1]:

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
import operator
import re
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from nameparser import HumanName 


# # Section 1: Data Exploration

# Step one, load the data.

# In[2]:

train = pd.read_csv("/Users/Dave/Desktop/Programming/Personal Projects/titanic_kaggle/train.csv")
test = pd.read_csv("/Users/Dave/Desktop/Programming/Personal Projects/titanic_kaggle/test.csv")


# Let's see what kind of data we are working with

# In[3]:

train.describe()


# In[4]:

train.head()


# We can already see that we have some missing data in the Cabin variable. Let's have a look at what else we are missing.

# In[5]:

train.isnull().sum()


# Pretty good for the most part. Age is missing about 20% of its data, hopefully we will be able to use other data to provide a fair guess as to what those ages should be. Cabin is missing most of its data, but we might be able to learn something from the information that we have. Embarked is missing just two values, that won't be a problem.

# Next, let's look at the data via some graphs.

# In[9]:

train.Survived.plot(kind='hist', bins = 2, edgecolor = 'red')
plt.xticks((1, 0))
plt.xlabel(('Died','Survived'))
plt.show()
train.Survived.value_counts()


# 61.6% of people died...

# In[10]:

n_groups = 3
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.7

PclassSurvived = train[train.Survived==1].Pclass.value_counts().sort_index()
PclassDied = train[train.Survived==0].Pclass.value_counts().sort_index()

plt.bar(index, PclassSurvived, bar_width,
        alpha=opacity,
        color='b',
        label='Survived')

plt.bar(index + bar_width, PclassDied, bar_width,
        alpha=opacity,
        color='g',
        label='Died')

plt.xticks(index + bar_width, (1,2,3))
plt.xlabel("Pclass")
plt.ylabel("Number of Passengers")
plt.legend(loc = 2)
plt.show()

print pd.crosstab(train.Pclass, train.Survived, margins=True)


# Looks like you don't want to be in third class. I wonder what happens when we factor in gender as well.

# In[26]:

n_groups = 3
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.7

trainMale = train[train.Sex == 'male']
trainFemale = train[train.Sex == 'female']

PclassMaleSurvived = trainMale[trainMale.Survived==1].Pclass.value_counts().sort_index()
PclassMaleDied = trainMale[trainMale.Survived==0].Pclass.value_counts().sort_index()

PclassFemaleSurvived = trainFemale[trainFemale.Survived==1].Pclass.value_counts().sort_index()
PclassFemaleDied = trainFemale[trainFemale.Survived==0].Pclass.value_counts().sort_index()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))


PclassM1 = axes[0].bar(index, PclassMaleSurvived, bar_width,
                 alpha=opacity,
                 color='darkblue',
                 label='Male Survived')

PclassM0 = axes[0].bar(index + bar_width, PclassMaleDied, bar_width,
                 alpha=opacity,
                 color='lightblue',
                 label='Male Died')

PclassF1 = axes[1].bar(index, PclassFemaleSurvived, bar_width,
                 alpha=opacity,
                 color='darkgreen',
                 label='Female Survived')

PclassF0 = axes[1].bar(index + bar_width, PclassFemaleDied, bar_width,
                 alpha=opacity,
                 color='lightgreen',
                 label='Female Died')

for ax in axes:
    ax.legend(loc = 9)
    ax.set_xlabel('Pclass')
    ax.set_ylabel('Number of People')
    
plt.setp(axes, 
         xticks = index + bar_width,
         xticklabels=['1', '2', '3'],
        )
plt.show()

print "Male Values:"
print pd.crosstab(trainMale.Pclass, trainMale.Survived, margins = True)
print ""
print "Female Values:"
print pd.crosstab(trainFemale.Pclass, trainFemale.Survived, margins = True)


# Upper and middle class women mostly survived, not so much for the lower class.
# None of the men did particularly well, especially those in the lower class.

# To get a better general sense, let's compare the survival of men and women.

# In[27]:

n_groups = 2
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.7

SexSurvived = train[train.Survived==1].Sex.value_counts().sort_index(ascending = False)
SexDied = train[train.Survived==0].Sex.value_counts().sort_index(ascending = False)

Sex1 = plt.bar(index, SexSurvived, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Survived')
Sex0 = plt.bar(index + bar_width, SexDied, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Died')
plt.xticks(index + bar_width, ('Male', 'Female'))
plt.xlabel("Sex")
plt.legend()
plt.show()

print "Total number of people in each sex:" 
print pd.crosstab(train.Sex, train.Survived, margins = True)


# As expected, women typically survived and men did not.

# Let's move on to Ages, and see how things worked out there.

# In[28]:

fig = plt.figure(figsize=(15, 6))
alpha = 0.7

AgeSurvived = train[train.Survived==1].Age.value_counts().sort_index()
AgeDied = train[train.Survived==0].Age.value_counts().sort_index()

Age1 = plt.plot(AgeSurvived,
                 alpha=opacity,
                 color='b',
                 label='Survived')
Age0 = plt.plot(AgeDied,
                 alpha=opacity,
                 color='g',
                 label='Died')
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()


# It appears that passengers near the age of 4/5 are more likely to survive, otherwise it seems that death is more common.

# In[29]:

print "Now let's take care of those", train.Age.isnull().sum(), "null values"


# In[30]:

print "One idea would be to take the median age:", train.Age.median(), "or mean:", train.Age.mean(), "but I think we can get a clue from people's titles (ex Mr., Mrs.)"


# First let's see what titles we have.

# In[31]:

titles = []
for name in train.Name:
    titles.append(HumanName(name).title)
print set(titles)


# The titles look good, expect there's an empty string, perhaps that's for the less common titles, but I feel pretty good about this range since it has covered the basicis.
# Now let's make a new feature for these titles.

# In[32]:

train.Title = train.Name.map(lambda x: HumanName(x).title)


# In[33]:

print train[train.Title == ''].Name
print train[train.Title == ''].Survived


# These are the people with the 'empty' titles. Since there are only seven of them, and many of their titles are unique, I don't mind grouping them together into a 'uncommon title' group. Plus, they seem to follow the typical pattern of women survived and men died, so I do not expect any issues to arise in the machine learning section.

# In[34]:

titleAges = {}
for title in train.Title:
    if title not in titleAges:
        titleAges[title] = train[train.Title == title].Age.median()
print titleAges


# Now we can add the median age of the respective title to the passengers who do not have an age.

# In[35]:

for title in train.Title:
    train.Age = train.Age.fillna(titleAges[title])


# Now let's do the same for the test dataframe, to keep things equal.

# In[36]:

test.Title = test.Name.map(lambda x: HumanName(x).title)
testTitleAges = {}
for title in test.Title:
    if title not in testTitleAges:
        testTitleAges[title] = test[test.Title == title].Age.median()
for title in test.Title:
    test.Age = test.Age.fillna(testTitleAges[title])


# Next, let's move on to SibSp (siblings and spouses)

# In[37]:

n_groups1 = 5
index1 = np.arange(n_groups1)
n_groups2 = 7
index2 = np.arange(n_groups2)
bar_width = 0.35
opacity = 0.7

SSSurvived = train[train.Survived==1].SibSp.value_counts().sort_index()
SSDied = train[train.Survived==0].SibSp.value_counts().sort_index()

SS1 = plt.bar(index1, SSSurvived, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Survived')
plt.xticks(index + bar_width, (0,1,2,3,4,5,8))

SS0 = plt.bar(index2 + bar_width, SSDied, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Died')

plt.xlabel("SibSp")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()
print "Number of people in each SibSp group:"
pd.crosstab(train.SibSp, train.Survived, margins = True)


# Things only look decent for pairs, otherwise not too brilliant.

# Next, to Parch (parents and children).

# In[38]:

n_groups1 = 5
index1 = np.arange(n_groups1)
n_groups2 = 7
index2 = np.arange(n_groups2)
bar_width = 0.35
opacity = 0.7

PCSurvived = train[train.Survived==1].Parch.value_counts().sort_index()
PCDied = train[train.Survived==0].Parch.value_counts().sort_index()

PC1 = plt.bar(index1, PCSurvived, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Survived')

PC0 = plt.bar(index2 + bar_width, PCDied, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Died')

plt.xlabel("Parch")
plt.legend()
plt.xticks(index + bar_width, (0,1,2,3,4,5,6))
plt.ylabel("Number of Passengers")
plt.show()
print "Number of people in each Parch group:"
print pd.crosstab(train.Parch, train.Survived, margins = True)


# Hmm, this looks very similar to the SibSp plot/values. I'm going to combine the two, and call the new variable 'FamilyMembers.' Perhaps something will standout here. I'm particularly interested in passengers with 2 family members, as they might be the most likely to survived based on the SibSp and Parch data.

# In[39]:

train['FamilyMembers'] = (train.SibSp + train.Parch)
test['FamilyMembers'] = (test.SibSp + test.Parch)


# In[40]:

n_groups1 = 7
index1 = np.arange(n_groups1)
n_groups2 = 9
index2 = np.arange(n_groups2)
bar_width = 0.35
opacity = 0.7

FamilySurvived = train[train.Survived==1].FamilyMembers.value_counts().sort_index()
FamilyDied = train[train.Survived==0].FamilyMembers.value_counts().sort_index()

F1 = plt.bar(index1, FamilySurvived, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Survived')

F0 = plt.bar(index2 + bar_width, FamilyDied, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Died')

plt.xlabel("Number of Family Members")
plt.ylabel("Number of Passengers")
plt.legend()
plt.xticks(index + bar_width, (0,1,2,3,4,5,6,7,10))
plt.show()
print "Number of people in each Family Member size:"
print pd.crosstab(train.FamilyMembers, train.Survived, margins = True)


# Things don't look too good for solo travellers or larger families, but those with 1-3 family members do rather well. 

# I'm not going to plot the ticket values, but maybe we'll learn something when looking at the values.

# In[41]:

train.Ticket.value_counts()


# It's interesting to see that some people have the same ticket number, perhaps they are a part of the same family?

# In[42]:

print train[train.Ticket == 'CA. 2343'].Name, train[train.Ticket == 'CA. 2343'].Survived
print
print train[train.Ticket == '347082'].Name, train[train.Ticket == '347082'].Survived
print 
print train[train.Ticket == '1601'].Name, train[train.Ticket == '1601'].Survived 


# After looking at the 3 most common ticket numbers, we can see that sometimes, but not always, a family shares the same ticket number. In the feature engineering section, I think it will be worthwhile to add a feature for 'shared tickets.' I wonder if there are going to be some similarities in the fare prices.

# In[43]:

print train.Fare.describe()
print
print train.Fare.value_counts().sort_index(ascending = False)


# Some of those fare values are pretty high, mainly 512, which is nearly double the next highest value. There are 15 people who didn't pay anything. Let's look into those two things.

# In[44]:

print train[train.Fare == 512.3292].Name
print
print train[train.Fare == 0].Name


# I haven't learned anything conclusive yet, let's see if the last names match up with anyone.
# As for the people who paid 512 for a ticket, I'm just going to leave them alone, I suppose they just had the best rooms on the boat.

# In[45]:

FareNames = []
for name in train[train.Fare == 512.3292].Name:
    FareNames.append(name)
for name in train[train.Fare == 0].Name:
    FareNames.append(name)

for FareName in FareNames:
    for name in train.Name:
        if HumanName(FareName).last == HumanName(name).last:
            print FareName, " - ", name


# Let's see what the fares were for the Johnson and Andrews families, maybe they weren't shared accordingly, or Alfred and William's fares were just left out.

# In[46]:

for name in train.Name:
    if 'Johnson' in name or 'Andrews' in name:
        print name, train[train.Name == name].Fare


# There are some values here that we can match for the Johnsons and Andrews, but I want to see what their family sizes are, just to make sure that they are a member of these families.

# In[47]:

print train[train.Name == 'Johnson, Mr. William Cahoone Jr'].FamilyMembers
print
print train[train.Name == 'Johnson, Mr. Alfred'].FamilyMembers
print 
print train[train.Name == 'Andrews, Mr. Thomas Jr'].FamilyMembers


# That's somewhat surprising. It looks as though none of these men are members of those families. Since there aren't many people with a fare value of 0, I'm going to assign their new values to the median.

# In[48]:

train.loc[train['Fare'] == 0, 'Fare'] = train.Fare.median()


# Next on the list is Cabins. Unfortunately, we are missing most of the information about cabins, but let's take another look at what we have.

# In[49]:

train.Cabin.value_counts()


# I'm going to hold on manipulating this data until the feature engineering stage, but I think something can be gained by sorting cabins based on their letter (which should represent the floor of the titanic they are on; higher floor = more wealthy = closer to life boats = more likely to survive).

# Lastly, Embarked. Let's see what we have here.

# In[50]:

print pd.crosstab(train.Embarked, train.Survived, margins=True)


# It seems that the French were more likely to survive (C = Cherbourg, France). I wonder how the demographics of the French compared to the English speakers.

# In[51]:

print pd.crosstab(train.Embarked, train.Sex, margins=True)
print
print "Males and females from Cherbourg who survived:"
print train[(train.Embarked == 'C') & (train.Survived == 1)].Sex.value_counts()
print 
print pd.crosstab(train.Embarked, train.Pclass, margins=True)
print
print pd.crosstab(train.Embarked, train.Family, margins=True)


# The only real outlier that I see from Cherbourg's data is that a higher percentage of them are from the upper class; more than half of them. I suppose this helps to explain why such a high percentage of them survived the trip.

# Let's take care of those two missing values.

# In[52]:

train[train.Embarked.isnull()]


# Since they are two first class women, I'm sure that any algorithm would classify them as survivors, so I'm just going to assign their Embarked values to the most common location, Southhampton.

# In[53]:

train['Embarked'] = train['Embarked'].fillna('Empty')
train.loc[train['Embarked'] == 'Empty', 'Embarked'] = 'S'


# # Section 2: Feature Engineering

# To simplify things, I'm going to use the variable 'df' to represent my dataframes, then apply this function to my train and test dataset.

# In[54]:

def set_features(df):

    #need to give males and females numeric values 
    df.loc[df["Sex"] == "male", "Sex"] = 0
    df.loc[df["Sex"] == "female", "Sex"] = 1

    #need to give Embarked values, numeric values.
    #Set NAs to S as it is the most common port of departure.
    df['Embarked'] = df['Embarked'].fillna('S')
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2

    #Set fare values of 0 and NaN to the median value of fare.
    df.loc[df['Fare'] == 0, 'Fare'] = df['Fare'].median()
    for value in df['Fare']:
        if pd.isnull(value):
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    #If someone has a sibling or spouse onboard, SibSp needs to be 1 or greater.
    df['Has_SibSp'] = df['SibSp'].map(lambda x: 1 if x >= 1 else 0)

    #To be a parent, you need a spouse, have at least 1 child, and be older than 18.
    #SibSp is >= 1 because you could be an adult with a sibling on board.
    df['Parent'] = (df['SibSp'] >= 1) & (df['Parch'] > 0) & (df['Age'] >= 18)

    #Single Parent can't have Spouse (duh) or siblings, minimum 1 kid, and 18 or older.
    df['Single_Parent'] = (df['SibSp'] == 0) & (df['Parch'] > 0) & (df['Age'] >= 18)

    #To be a mother, you need to be a parent and female. Fathers = Parent & Male
    df['Mother'] = (df['Parent'] == 1) & (df['Sex'] == 1)
    df['Father'] = (df['Parent'] == 1) & (df['Sex'] == 0)
    df['Single_Mother'] = (df['Single_Parent'] == 1) & (df['Sex'] == 1)
    df['Single_Father'] = (df['Single_Parent'] == 1) & (df['Sex'] == 0)

    #Child has at least 1 parent and is 17 or younger
    df['Child'] = (df['Parch'] >= 1) & (df['Age'] <= 17)

    #To be a daughter, you need to be a girl, and a child.
    #To be a son, likewise, but a boy.
    df['Daughter'] = (df['Sex'] == 1) & (df['Child'] == 1)
    df['Son'] = (df['Sex'] == 0) & (df['Child'] == 1)

    #Only child has no siblings, and is a child
    df['Only_Child'] = (df['Child'] == 1) & (df['SibSp'] == 0)

    #Orphan if you have no parents and are 17 or younger
    df['Orphan'] = (df['Age'] <= 17) & (df['Parch'] == 0)

    #To have 'One Parent', you need 1 parent and be 17 or younger
    df['One_Parent'] = (df['Parch'] == 1) & (df['Age'] <= 17)

    #Various life stages
    df['Infant'] = df['Age'] <= 2
    df['Kid'] = (df['Age'] > 3) & (df['Age'] <= 12)
    df['Teenager'] = (df['Age'] > 13) & (df['Age'] <= 17)
    df['Adult'] = df['Age'] >= 18

    #To be single, must be 18 or older, no parents/children, no siblings/spouses.
    df['Single_Man'] = (df['Sex'] == 0) & (df['Age'] >= 18) & (df['SibSp'] == 0) & (df['Parch'] == 0)
    df['Single_Woman'] = (df['Sex'] == 1) & (df['Age'] >= 18) & (df['SibSp'] == 0) & (df['Parch'] == 0)

    #Married without kids or parents, need to be at least 18, and have at least 1 sibling or spouse.
    df['Married_no_Kids_no_Parents'] = (df['SibSp'] >= 1) & (df['Parch'] == 0) & (df['Age'] >= 18)

    #NameLength is number of characters in someone's name
    df['NameLength'] = df['Name'].map(lambda x: len(x))
    #print df['NameLength'].describe()
    #Long names > 37, medium names 36-18, short names 0-17. Used describe to learn mean and standard deviation
    df['NameLengthGroup'] = df['NameLength'].map(lambda x: 3 if x > 37 else (2 if x > 18 else 1)) 
    df['FemaleLongName'] = (df['Sex'] == 1) & (df['NameLengthGroup'] == 3)
    df['FemaleMediumName'] = (df['Sex'] == 1) & (df['NameLengthGroup'] == 2)
    df['FemaleShortName'] = (df['Sex'] == 1) & (df['NameLengthGroup'] == 1)
    df['MaleLongName'] = (df['Sex'] == 0) & (df['NameLengthGroup'] == 3)
    df['MaleMediumName'] = (df['Sex'] == 0) & (df['NameLengthGroup'] == 2)
    df['MaleShortName'] = (df['Sex'] == 0) & (df['NameLengthGroup'] == 1)
    
    #Sort fares into three even categories.
    #print df['Fare'].describe()
    df['HighFare'] = df['Fare'] >= 31
    df['MediumFare'] = (df['Fare'] < 31) & (df['Fare'] > 8)
    df['LowFare'] = df['Fare'] <= 8

    #Combined class and gender to better organize people.
    df['RichWoman'] = (df['Pclass'] == 1) & (df['Sex'] == 1) & (df['Age'] >= 18)
    df['MiddleClassWoman'] = (df['Pclass'] == 2) & (df['Sex'] == 1) & (df['Age'] >= 18)
    df['PoorWoman'] = (df['Pclass'] == 3) & (df['Sex'] == 1) & (df['Age'] >= 18)
    df['RichMan'] = (df['Pclass'] == 1) & (df['Sex'] == 0) & (df['Age'] >= 18)
    df['MiddleClassMan'] = (df['Pclass'] == 2) & (df['Sex'] == 0) & (df['Age'] >= 18)
    df['PoorMan'] = (df['Pclass'] == 3) & (df['Sex'] == 0) & (df['Age'] >= 18)
    df['RichChild'] = (df['Pclass'] == 1) & (df['Age'] <= 17)
    df['MiddleClassChild'] = (df['Pclass'] == 2) & (df['Age'] <= 17)
    df['PoorChild'] = (df['Pclass'] == 3) & (df['Age'] <= 17)
    df['RichGirl'] = (df['RichChild'] == 1) & (df['Sex'] == 1)
    df['MiddleClassGirl'] = (df['MiddleClassChild'] == 1) & (df['Sex'] == 1)
    df['PoorGirl'] = (df['PoorChild'] == 1) & (df['Sex'] == 1)
    df['RichBoy'] = (df['RichChild'] == 1) & (df['Sex'] == 0)
    df['MiddleClassBoy'] = (df['MiddleClassChild'] == 1) & (df['Sex'] == 0)
    df['PoorBoy'] = (df['PoorChild'] == 1) & (df['Sex'] == 0)

    #Family size is the sum of siblings, spouses, parents, and children.
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['SmallFamily'] = df['FamilySize'] <= 3
    df['MediumFamily'] = (df['FamilySize'] >= 4) & (df['FamilySize'] <= 6)
    df['LargeFamily'] = df['FamilySize'] >= 7

    #Based on correlation and plot analysis, these features were combined; hoping for useful learning.
    df['Pclass*Age'] = df['Pclass'] * df['Age']
    df['Fare/Pclass'] = df['Fare'] / df['Pclass']
    df['FamilySize*Pclass'] = df['FamilySize'] * df['Pclass']

    #a helper function to provide ids to the variations of a variable.
    #the ids for each variation of the variable are stored in the variable_id_mapping (a dictionary)
    def id_mapping(variable_id, variable_id_mapping):
        if variable_id not in variable_id_mapping:
            if len(variable_id_mapping) == 0:
                current_id = 1
            else:
                current_id = (max(variable_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
            variable_id_mapping[variable_id] = current_id
        return variable_id_mapping[variable_id]

    #create a mapping of the title_ids
    title_id_mapping = {}
    def get_title_id(row):
        title = row["Name"].split(' ')[1]
        title_id = "{0}".format(title)
        return id_mapping(title_id, title_id_mapping)
    df["TitleID"] = df.apply(get_title_id, axis = 1)

    #cabin_level_ids relate to the first letter in the cabin column.
    cabin_level_id_mapping = {}
    def get_cabin_level(row):
        cabin_level = str(row['Cabin'])[0]
        cabin_level_id = "{0}".format(cabin_level)
        return id_mapping(cabin_level_id, cabin_level_id_mapping)
    df["CabinLevelID"] = df.apply(get_cabin_level, axis = 1)
    
    #If the cabin number is konwn, Known_cabin = 1, else 0
    df['Known_Cabin'] = df['Cabin'].map(lambda x: 1 if pd.notnull(x) else 0)
    
    #Find the length of the ticket, perhaps longer tickets are for include more amenities, 
    #therefore associated with wealth.
    df['TicketLength'] = df['Ticket'].map(lambda x: len(df['Ticket']))
    
    #ticket_id realtes to the first character in the ticket.
    ticket_id_mapping = {}
    def get_ticket_id(row):
        ticket = str(row['Ticket'])[0]
        ticket_id = "{0}".format(ticket)
        return id_mapping(ticket_id, ticket_id_mapping)
    df['TicketID'] = df.apply(get_ticket_id, axis = 1)
    
    #If there is no value for cabin, set it to 0.
    for value in df['Cabin']:
        if pd.isnull(value):
            value = 0

    #dict to give each family a unique id
    family_id_mapping = {}
    def get_family_id(row):
        last_name = row['Name'].split(',')[0]
        family_id = "{0}{1}".format(last_name, row["FamilySize"])
        return id_mapping(family_id, family_id_mapping)
    df["FamilyID"]  = df.apply(get_family_id, axis = 1)


# # Section 3: Machine Learning

# Let's set everything equal for out train and test datasets by applying the set_features function to them.

# In[55]:

set_features(train)
set_features(test)


# In[68]:

#A list of all of our numeric features, which will be used for machine learning.
Predictors = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked',
              'Has_SibSp', 'Parent','Mother','Father','Daughter','Son',
              'Married_no_Kids_no_Parents','TitleID','FamilySize','FamilyID',
              'Known_Cabin','Single_Parent','Single_Mother','Single_Father',
              'Child','Only_Child','One_Parent','Single_Man','Single_Woman',
              'RichWoman','MiddleClassWoman','PoorWoman','RichMan','MiddleClassMan',
              'PoorMan','RichChild','MiddleClassChild','PoorChild','Orphan',
              'Infant','Teenager','Adult','Pclass*Age','SmallFamily',
              'MediumFamily','LargeFamily','CabinLevelID','Fare/Pclass',
              'RichGirl','MiddleClassGirl','PoorGirl','RichBoy','MiddleClassBoy',
              'PoorBoy','NameLength','NameLengthGroup','FemaleLongName','FemaleMediumName',
              'FemaleShortName','MaleLongName','MaleMediumName','MaleShortName',
              'FamilySize*Pclass','LowFare','MediumFare','HighFare','TicketLength','TicketID']


# Let's spilt the data so that we have a training and testing set.

# In[69]:

trainData, testData = train_test_split(train, test_size = 0.3)


# In[70]:

cv = 10


# To do our machine learning, we are going to use six different algorithms. These algorithms will learn via their own pipelines, then the voting classifier will combine their results with different weights, based on their accuracies.
# The six algorithms are: RandomForestClassifier, GaussianNB, GradientBoostingClassifier, AdaBoostClassifier, LogisticRegression, and KNeighborsClassifier.

# In[71]:

randomForest_pipe = Pipeline(steps = [('feature_union', FeatureUnion([('pca', PCA()),
                                                         ('select_KBest', SelectKBest())
                                                         ])),
                                      ('randomForest', RandomForestClassifier())
                                      ])

randomForest_parameters = dict(feature_union__pca__n_components = [12],
                               feature_union__pca__whiten = [False],
                               feature_union__select_KBest__k = [35],
                               randomForest__n_estimators = [300],
                               #randomForest__min_samples_split = [6],
                               #randomForest__min_samples_leaf = [1],
                               randomForest__max_leaf_nodes = [10],
                               #randomForest__warm_start = [False],
                               #randomForest__oob_score = [True],
                               #randomForest__bootstrap = [True],
                               )

randomForest_grid_search = GridSearchCV(randomForest_pipe,
                                        randomForest_parameters,
                                        cv = cv, 
                                        scoring = 'accuracy')
randomForest_grid_search.fit(trainData[Predictors], trainData['Survived'])


# In[72]:

gau_pipe = Pipeline(steps = [('feature_union', FeatureUnion([('pca', PCA()),
                                                             ('select_KBest', SelectKBest())
                                                            ])),
                             ('gau', GaussianNB())
                            ])

gau_parameters = dict(feature_union__pca__n_components = [12],
                      feature_union__pca__whiten = [True],
                      feature_union__select_KBest__k = [45],
                      )

gau_grid_search = GridSearchCV(gau_pipe,
                               gau_parameters,
                               cv = cv, 
                               scoring = 'accuracy')
gau_grid_search.fit(trainData[Predictors], trainData['Survived'])


# In[73]:

gBoost_pipe = Pipeline(steps = [('feature_union', FeatureUnion([('pca', PCA()),
                                                                ('select_KBest', SelectKBest())
                                                               ])),
                                ('gBoost', GradientBoostingClassifier())
                                ])

gBoost_parameters = dict(feature_union__pca__n_components = [24],
                         feature_union__pca__whiten = [False],
                         feature_union__select_KBest__k = [32],
                         gBoost__min_samples_split = [6],
                         gBoost__min_samples_leaf = [1],
                         gBoost__n_estimators = [50],
                         gBoost__max_leaf_nodes = [5],
                         gBoost__learning_rate = [0.1],
                         gBoost__subsample = [0.65],
                         gBoost__presort = [True]
                         )

gBoost_grid_search = GridSearchCV(gBoost_pipe,
                                  gBoost_parameters,
                                  cv = cv, 
                                  scoring = 'accuracy')
gBoost_grid_search.fit(trainData[Predictors], trainData['Survived'])


# In[74]:

ada_pipe = Pipeline(steps = [('feature_union', FeatureUnion([('pca', PCA()),
                                                             ('select_KBest', SelectKBest())
                                                            ])),
                            ('ada', AdaBoostClassifier())
                            ])

ada_parameters = dict(feature_union__pca__n_components = [28],
                      feature_union__pca__whiten = [True],
                      feature_union__select_KBest__k = [36],
                      ada__n_estimators = [100],
                      ada__learning_rate = [0.2],
                      ada__base_estimator = [DecisionTreeClassifier(max_depth = 1,
                                                                    splitter = 'random')]
                      )

ada_grid_search = GridSearchCV(ada_pipe,
                               ada_parameters,
                               cv = cv, 
                               scoring = 'accuracy')
ada_grid_search.fit(trainData[Predictors], trainData['Survived'])


# In[75]:

lr_pipe = Pipeline(steps = [('feature_union', FeatureUnion([('pca', PCA()),
                                                            ('select_KBest', SelectKBest())
                                                           ])),
                            ('lr', LogisticRegression())
                            ])

lr_parameters = dict(feature_union__pca__n_components = [25],
                      feature_union__pca__whiten = [False],
                      feature_union__select_KBest__k = [35],
                      lr__fit_intercept = ['True'],
                      lr__C = [1]
                     )

lr_grid_search = GridSearchCV(lr_pipe,
                              lr_parameters,
                              cv = cv, 
                              scoring = 'accuracy')
lr_grid_search.fit(trainData[Predictors], trainData['Survived'])


# In[76]:

kn_pipe = Pipeline(steps = [('feature_union', FeatureUnion([('pca', PCA()),
                                                            ('select_KBest', SelectKBest())
                                                           ])),
                            ('kn', neighbors.KNeighborsClassifier())
                            ])

kn_parameters = dict(feature_union__pca__n_components = [30],
                      feature_union__pca__whiten = [True],
                      feature_union__select_KBest__k = [45],
                      kn__n_neighbors = [4],
                      kn__algorithm = ['auto'],
                      kn__leaf_size = [10],
                      kn__weights = ['uniform'],
                      kn__p = [1]
                     )

kn_grid_search = GridSearchCV(kn_pipe,
                              kn_parameters,
                              cv = cv, 
                              scoring = 'accuracy')
kn_grid_search.fit(trainData[Predictors], trainData['Survived'])


# The randomForest, Gradient Boosting, Adaboost, and LogisticRegression algorithms typical perform about equal and the best, so that's why they are weighted the highest.

# In[77]:

voting = VotingClassifier(estimators=[('randomForest', randomForest_grid_search.best_estimator_),
                                      ('gau', gau_grid_search.best_estimator_),
                                      ('gBoost', gBoost_grid_search.best_estimator_),
                                      ('ada', ada_grid_search.best_estimator_),
                                      ('lr', lr_grid_search.best_estimator_),
                                      ('kn', kn_grid_search.best_estimator_)],
                          voting='soft',
                          weights=[0.23,0.04,0.23,0.23,0.23,0.04])
voting.fit(trainData[Predictors], trainData['Survived'])


# # Section 4: Evaluation

# In[78]:

algorithms = {randomForest_grid_search: "randomForest", 
              gau_grid_search: "gaussian",
              gBoost_grid_search: "Gradient Boosting",
              ada_grid_search: "Ada Boost",
              lr_grid_search: "Linear Regression",
              kn_grid_search: "KNeighbor",
              voting: "voting"}


# In[79]:

for k,v in algorithms.iteritems():
    print v, "trainData score:", k.score(trainData[Predictors], trainData['Survived'])
    print v, "testData score:", k.score(testData[Predictors], testData['Survived'])


# In[80]:

predictions = voting.predict(test[Predictors]).astype(int)
submission = pd.DataFrame({'PassengerID': test['PassengerId'],
                           'Survived': predictions
                           })
submission.to_csv('/Users/Dave/Desktop/Programming/Personal Projects/titanic_kaggle/Titanic_Submission3.csv', index=False)


# # Conclusion

# I hope that you enjoyed, and possibly even learned something from, my analysis. The results could be improved by using more creativity in the feature engineering section, or tuning the parameters of the algorithms better, but I am happy with what I accomplished, and feel that it would be best to move onto another competition, rather than continue this one. Thanks for reading!

# In[ ]:



