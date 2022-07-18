#!/usr/bin/env python
# coding: utf-8

# In[177]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree


# In[178]:


data = pd.read_csv("D:/Skillslash/Stats and ML/Projects/Titanic/train.csv")
data.head()


# In[179]:


data.info()


# In[180]:


#data.select_dtypes(include='object')


# In[181]:


data.isnull().sum()


# In[182]:


figure = plt.figure(figsize = (20,5))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']],stacked = True, bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('No. of Passengers')
plt.legend()


# In[183]:


data.isnull().sum()


# In[184]:


data['Age'] = data['Age'].fillna(data['Age'].mode()[0])
data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode()[0])
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])


# In[185]:


data.isnull().sum()


# In[186]:


data.dtypes


# In[187]:


data.drop(['Name','Cabin', 'Ticket'], axis = 1, inplace = True)


# In[188]:


embark = pd.get_dummies(data['Embarked'], drop_first = True, prefix = 'Embark')
sex = pd.get_dummies(data['Sex'], drop_first = True, prefix = 'Sex')


# In[189]:


data = pd.concat([data,sex,embark],axis = 1)
data.head()


# In[190]:


data.drop(['Sex', 'Embarked'], axis = 1, inplace = True)


# In[191]:


data.head()


# In[192]:


data.drop_duplicates(inplace = True)
data.shape


# In[193]:


X = data.drop(['Survived'], axis = 1)


# In[194]:


y = data['Survived']


# In[195]:


from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[196]:


from sklearn.linear_model import LogisticRegression


# In[197]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[198]:


prediction = logmodel.predict(x_test)


# In[199]:


from sklearn.metrics import confusion_matrix
accuracy = confusion_matrix(y_test, prediction)


# In[200]:


accuracy


# In[201]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, prediction)
score*100


# In[202]:


prediction


# ### Test Data Set 

# In[247]:


test_data = pd.read_csv("D:/Skillslash/Stats and ML/Projects/Titanic/test.csv")
test_data.head()


# In[ ]:





# In[248]:


test_data.info()


# In[249]:


test_data.isnull().sum()


# In[250]:


test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mode()[0])
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mode()[0])


# In[251]:


test_data.isnull().sum()


# In[252]:


test_data.drop(['Name','Cabin', 'Ticket'], axis = 1, inplace = True)


# In[253]:


sex = pd.get_dummies(test_data['Sex'], drop_first = True, prefix = 'Sex')
embarked = pd.get_dummies(test_data['Embarked'], drop_first = True, prefix = 'Embarked')


# In[254]:


test_data = pd.concat([test_data,sex,embarked],axis = 1)
test_data.head()


# In[255]:


test_data.drop(['Sex','Embarked'], axis = 1, inplace = True)
test_data.head()


# In[256]:


data.drop_duplicates(inplace = True)
data.shape


# In[257]:


test_data.columns


# In[258]:


X_train.columns


# In[259]:


predictions = logmodel.predict(test_data)
predictions


# In[266]:


submission = pd.DataFrame()
submission['PassengerID'] = test_data['PassengerId']
submission['Survived'] = predictions
submission.head()


# In[267]:


#test_data.drop_index(inplace = True)


# In[268]:


submission.to_csv('D:/Skillslash/Stats and ML/Projects/Titanic/test_Sub3.csv')


# In[ ]:




