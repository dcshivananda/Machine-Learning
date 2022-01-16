import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_excel("D:\PESITM\Subjects\BTD\Sep 2020\IA\Final Average IA for ML.xlsx")
print(data.shape)
data.dropna()
print(data.head(n=5))
x=data['3rd IA'].values.reshape(-1,1)
y=data['Final Average'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)
lm=linear_model.LinearRegression()
print(x_train)
model=lm.fit(x_train, y_train)
r2=lm.score(x_train, y_train)
print("r square value is ", r2)
print("Intercept = ", model.intercept_)
print("Slope = ", model.coef_)

y_pred = model.predict(x_test)
print(y_pred)

plt.scatter(x_train, y_train, color='black')
plt.plot(x_train, lm.predict(x_train), color= 'blue', linewidth=1, label = 'Regression Line')
plt.xlabel("3rd IA")
plt.ylabel("Final Average")
plt.show()