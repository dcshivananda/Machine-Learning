#!/usr/bin/env python
# coding: utf-8

# ### Importing CarPrice CSV File 

# In[114]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[115]:


car_price = pd.read_csv("D:\Skillslash\Stats and ML\Project Related\CarPrice_Assignment.csv")
df = pd.DataFrame(car_price)
car_price.shape
#car_price.info()
car_price.head()


# ### Mean and Median of all Features 

# In[116]:


mean_all = df[["wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]].mean().round(2)
median_all = df[["wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]].median()
central_tendancy = {'Mean':mean_all,'Median':median_all}
df1 = pd.DataFrame(central_tendancy)
print(df1)


# ### Finding bottom 5% and top 5% Values using Percentile

# In[117]:


min_threshold = df[["wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]].quantile(0.05)
#print(min_threshold)

max_threshold = df[["wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]].quantile(0.95)
#print(max_threshold)
#df_threshold = df[[min_threshold],[max_threshold]]

threshold = {'Minimum Threshold(5%)':min_threshold, 'Maxmimum Threshold(95%)':max_threshold}
df_threshold = pd.DataFrame(threshold)
print(df_threshold)


# ### Finding Outliers using 'IQR' Method 

# In[118]:


q1 = df[["wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]].quantile(0.25)
q3 = df[["wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]].quantile(0.75)

iqr = q3 - q1

lower_limit = q1 - 1.5*iqr
upper_limit = q3 + 1.5*iqr

iqr1 = {'q1':q1,'q3':q3,'iqr':iqr, 'lower limit':lower_limit,'uppper limit':upper_limit}
iqr_df = pd.DataFrame(iqr1)
print(iqr_df)

#outlier = df[~((df["wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"] < (Q1 - 1.5 * IQR)) |(df["wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"] > (Q3 + 1.5 * IQR))).any(axis=1)]
#outlier = df[(df["wheelbase"] < lower_limit) or (df["wheelbase"] > upper_limit)].any(axis=0)
#print(outlier)


# ### Outlier treatment in smart way

# In[119]:


columns = car_price.columns.values.tolist()

for names in columns:
    if car_price[f'{names}'].dtype != 'object':
        cols_sort = np.sort(car_price[f'{names}'])
        
        q1 = np.percentile(cols_sort, 25)
        q3 = np.percentile(cols_sort,75)
        
        iqr = q3 - q1
        
        lower_limit = q1 - 1.5*iqr
        upper_limit = q3 + 1.5*iqr
        
        outlier = []
        
        for x in cols_sort:
            if((x>upper_limit) or (x<lower_limit)):
                outlier.append(x)
                
        if len(outlier)>0:
            car_price['columns'] = np.where((car_price[f'{names}']>upper_limit) & (car_price[f'{names}']<lower_limit), car_price[f'{names}'].median(),car_price[f'{names}'])
            print('\n outliers for {} \n'.format(names), outlier)
            print('are replaced by median',car_price[f'{names}'].median())

car_price_outlier_treated = car_price.copy()


# In[120]:


car_price_outlier_treated.shape


# In[121]:


car_price_outlier_treated.isnull().sum()


# ### Replacing incorrect Car Names  

# In[122]:


car_price_outlier_treated['Brand_Name'] = car_price_outlier_treated['CarName'].apply(lambda x:x.split(" ")[0])
car_price_outlier_treated['Brand_Name'].unique()


# In[123]:


car_price_outlier_treated.loc[(car_price_outlier_treated['Brand_Name'] == 'vw') | (car_price_outlier_treated['Brand_Name'] == 'vokswagen'),'Brand_Name'] = 'volkswagen'

car_price_outlier_treated.loc[car_price_outlier_treated['Brand_Name'] == 'porcshce','Brand_Name'] = 'porsche'

car_price_outlier_treated.loc[car_price_outlier_treated['Brand_Name'] == 'toyouta','Brand_Name'] = 'toyota'

car_price_outlier_treated.loc[car_price_outlier_treated['Brand_Name'] == 'Nissan','Brand_Name'] = 'nissan'

car_price_outlier_treated.loc[car_price_outlier_treated['Brand_Name'] == 'maxda','Brand_Name'] = 'mazda'


# In[124]:


car_price_outlier_treated['Brand_Name'].unique()


# In[125]:


car_price_outlier_treated.head()


# In[126]:


car_price_cat = car_price_outlier_treated.drop(['CarName'], axis = 1)


# In[127]:


car_price_outlier_treated.head()


# ### Finding Std. deviation of each feature

# In[128]:


car_price.std()


# ### Finding Variance of each feature 

# In[129]:


car_price.var()


# ### Finding Skewness of each feature

# In[130]:


df.skew(axis = 0,skipna = True)


# ### Correlation of features with Carprice

# In[131]:


car_price_corr = car_price.corr()['price']
#df.corr()
corr_df = {'car_price_corr':car_price_corr}
print(pd.DataFrame(corr_df))


# In[ ]:





# In[132]:


car_price_num = car_price_outlier_treated.select_dtypes(include = [np.number])


# In[133]:


car_price_num


# In[134]:


car_price_num.drop(['columns'],axis = 1,inplace = True)


# In[135]:


car_price_num


# ### Pair Plot 

# In[136]:


sns.pairplot(car_price_num)


# ### Heat Map 

# In[137]:


sns.heatmap(car_price_num.corr(), annot = True)


# ### One Hot Encoding 

# In[138]:


num_columns = car_price_cat.select_dtypes(exclude = ['object'])
num_columns


# In[139]:


cat_columns = car_price_cat.select_dtypes(include=['object'])
cat_columns


# In[140]:


ohe = pd.get_dummies(cat_columns, drop_first = True)
ohe.head()


# In[144]:


num_columns1 = num_columns.drop(['columns'],axis = 1, inplace = True)


# In[145]:


model_data = pd.concat([num_columns,ohe], axis = 1)
model_data.head()


# In[148]:


X = model_data.drop(['price'],axis = 'columns')
y = model_data['price']


# In[149]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[151]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[152]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[160]:


print("Coefficients are: \n", lr.coef_)

print("\n Intercept is : \n", lr.intercept_)


# In[156]:


coeff = pd.DataFrame(zip(X_train.columns,lr.coef_))
coeff.transpose()


# In[169]:


y_train_pred = lr.predict(X_train)
#y_train_pred


# In[165]:


from sklearn import metrics

train_accuracy = lr.score(X_train, y_train)
print("Training Accuracy is : ", train_accuracy)


# In[170]:


y_test_pred = lr.predict(X_test)
#y_test_pred


# In[171]:


test_accuracy = lr.score(X_test,y_test)
print("Testing Accuracy is : ", test_accuracy)


# In[173]:


y_test_pred_df = pd.DataFrame(y_test_pred, columns = ['Predicted Car Price'])
X_test.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)
y_test_pred_df.reset_index(drop = True, inplace = True)

df_with_pred_price = pd.concat([X_test, y_test, y_test_pred_df], axis = 1)
df_with_pred_price


# In[ ]:




