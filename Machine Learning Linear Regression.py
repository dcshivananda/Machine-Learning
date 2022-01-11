import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_excel("D:\PESITM\Subjects\BTD\Sep 2020\IA\Final Average IA for ML.xlsx")
print(data.shape)
data.dropna()
data.head()
X=data['3rd IA'].values
Y=data['Final Average'].values
mean_x=np.nanmean(X)
mean_y=np.nanmean(Y)
print(mean_x, mean_y)
n=len(X)
print(n)

up=0
down=0
for i in range(n):
    up += (X[i]-mean_x)*(Y[i]-mean_y)
    down +=(X[i]-mean_x)**2
m=up/down
c=mean_y-(m*mean_x)
print(m,c)
max_x=np.max(X)
min_x=np.min(X)
x=np.linspace(min_x, max_x)
y=m*x+c
plt.plot(x,y, Color='blue', label='Regression Line')
plt.scatter(X,Y, color='orange', label='Scatter Plot')
plt.xlabel('1st IA')
plt.ylabel('Average IA')
plt.legend()
plt.show()
sst=0
ssr=0
for i in range(n):
    y=c+m*X[i]
    sst += (Y[i] - mean_y)**2
    ssr += (Y[i]-y)**2
r2 = 1-(ssr/sst)
print(r2)



    

    
    


