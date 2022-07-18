#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import OneHotEncoder
#from sklearn import tree


# In[9]:


data = pd.read_csv("D:/Skillslash/Stats and ML/Projects/Titanic/train.csv")
data.head()


# In[10]:


data.info()


# In[11]:


#data.select_dtypes(include='object')


# In[12]:


data.isnull().sum()


# In[13]:


figure = plt.figure(figsize = (20,5))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']],stacked = True, bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('No. of Passengers')
plt.legend()


# In[14]:


data.isnull().sum()


# In[15]:


data['Age'] = data['Age'].fillna(data['Age'].mode()[0])
data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode()[0])
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])


# In[16]:


data.isnull().sum()


# In[17]:


data.dtypes


# In[18]:


data.drop(['Name','Cabin', 'Ticket'], axis = 1, inplace = True)


# In[19]:


embark = pd.get_dummies(data['Embarked'], drop_first = True, prefix = 'Embark')
sex = pd.get_dummies(data['Sex'], drop_first = True, prefix = 'Sex')


# In[20]:


data = pd.concat([data,sex,embark],axis = 1)
data.head()


# In[21]:


data.drop(['Sex', 'Embarked'], axis = 1, inplace = True)


# In[22]:


data.head()


# In[23]:


data.drop_duplicates(inplace = True)
data.shape


# In[24]:


X = data.drop(['Survived'], axis = 1)


# In[25]:


y = data['Survived']


# In[27]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# In[94]:


knn = KNeighborsClassifier(n_neighbors = 18)
knn.fit(X_train,y_train)
print(knn.score(X_train, y_train))
#knn.predict(X_train)


# In[43]:


neighbors = np.arange(1,20)
train_accuracy = (np.empty(len(neighbors)))
test_accuracy = (np.empty(len(neighbors)))

for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(x_test, y_test)
    
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.show()


# In[76]:


knn.predict(X_train)


# ### Test Data Set 

# In[77]:


test_data = pd.read_csv("D:/Skillslash/Stats and ML/Projects/Titanic/test.csv")
test_data.head()


# In[ ]:





# In[78]:


test_data.info()


# In[79]:


test_data.isnull().sum()


# In[80]:


test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mode()[0])
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mode()[0])


# In[81]:


test_data.isnull().sum()


# In[82]:


test_data.drop(['Name','Cabin', 'Ticket'], axis = 1, inplace = True)


# In[83]:


sex = pd.get_dummies(test_data['Sex'], drop_first = True, prefix = 'Sex')
embarked = pd.get_dummies(test_data['Embarked'], drop_first = True, prefix = 'Embarked')


# In[84]:


test_data = pd.concat([test_data,sex,embarked],axis = 1)
test_data.head()


# In[85]:


test_data.drop(['Sex','Embarked'], axis = 1, inplace = True)
test_data.head()


# In[86]:


data.drop_duplicates(inplace = True)
data.shape


# In[87]:


test_data.columns


# In[88]:


X_train.columns


# In[91]:


predictions = knn.predict(test_data)


# In[92]:


submission = pd.DataFrame()
submission['PassengerID'] = test_data['PassengerId']
submission['Survived'] = predictions
submission.head()


# In[267]:


#test_data.drop_index(inplace = True)


# In[93]:


submission.to_csv('D:/Skillslash/Stats and ML/Projects/Titanic/test_Sub_KNN.csv')


# In[ ]:




