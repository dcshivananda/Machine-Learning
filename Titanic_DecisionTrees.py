#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import OneHotEncoder
#from sklearn import tree


# In[3]:


data = pd.read_csv("D:/Skillslash/Stats and ML/Projects/Titanic/train.csv")
data.head()


# In[4]:


data.info()


# In[5]:


#data.select_dtypes(include='object')


# In[6]:


data.isnull().sum()


# In[7]:


figure = plt.figure(figsize = (20,5))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']],stacked = True, bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('No. of Passengers')
plt.legend()


# In[8]:


data.isnull().sum()


# In[9]:


data['Age'] = data['Age'].fillna(data['Age'].mode()[0])
data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode()[0])
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])


# In[10]:


data.isnull().sum()


# In[11]:


data.dtypes


# In[12]:


data.drop(['Name','Cabin', 'Ticket'], axis = 1, inplace = True)


# In[13]:


embark = pd.get_dummies(data['Embarked'], drop_first = True, prefix = 'Embark')
sex = pd.get_dummies(data['Sex'], drop_first = True, prefix = 'Sex')


# In[14]:


data = pd.concat([data,sex,embark],axis = 1)
data.head()


# In[15]:


data.drop(['Sex', 'Embarked'], axis = 1, inplace = True)


# In[16]:


data.head()


# In[17]:


data.drop_duplicates(inplace = True)
data.shape


# In[18]:


X = data.drop(['Survived'], axis = 1)


# In[19]:


y = data['Survived']


# In[43]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
print(X_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[28]:


dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
dt.fit(X_train, y_train)


# In[36]:


y_predict_train = dt.predict(X_train)


# In[39]:


from sklearn.metrics import accuracy_score
print("Training Accuracy is : ", accuracy_score(y_train, y_predict_train))


# In[33]:


y_predict_test = dt.predict(x_test)


# In[38]:


from sklearn.metrics import accuracy_score
print("Testing Accuracy is : ", accuracy_score(y_test, y_predict_test))


# ### Test Data Set 

# In[48]:


test_data = pd.read_csv("D:/Skillslash/Stats and ML/Projects/Titanic/test.csv")
test_data.head()


# In[ ]:





# In[49]:


test_data.info()


# In[50]:


test_data.isnull().sum()


# In[51]:


test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mode()[0])
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mode()[0])


# In[52]:


test_data.isnull().sum()


# In[53]:


test_data.drop(['Name','Cabin', 'Ticket'], axis = 1, inplace = True)


# In[54]:


sex = pd.get_dummies(test_data['Sex'], drop_first = True, prefix = 'Sex')
embarked = pd.get_dummies(test_data['Embarked'], drop_first = True, prefix = 'Embarked')


# In[55]:


test_data = pd.concat([test_data,sex,embarked],axis = 1)
test_data.head()


# In[56]:


test_data.drop(['Sex','Embarked'], axis = 1, inplace = True)
test_data.head()


# In[57]:


data.drop_duplicates(inplace = True)
data.shape


# In[58]:


test_data.columns


# In[59]:


X_train.columns


# In[60]:


predictions = dt.predict(test_data)


# In[61]:


submission = pd.DataFrame()
submission['PassengerID'] = test_data['PassengerId']
submission['Survived'] = predictions
submission.head()


# In[62]:


#test_data.drop_index(inplace = True)


# In[63]:


y_predict_test = dt.predict(x_test)
from sklearn.metrics import accuracy_score
print("Testing Accuracy is : ", accuracy_score(y_test, y_predict_test))


# In[65]:


submission.to_csv('D:/Skillslash/Stats and ML/Projects/Titanic/test_Sub_DT.csv')


# In[ ]:




