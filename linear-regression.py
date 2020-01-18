#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
x = np.random.rand(100,1)
y = 2 + 3 * x + np.random.rand(100,1)
plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[6]:


import numpy as np
from sklearn.linear_model import LinearRegression
x = np.array([5,15,25,35,45,55]).reshape((-1,1))
y = np.array([5,20,14,32,22,38])
print(x)
print(y)
model = LinearRegression()
model.fit(x,y)
r_sq = model.score(x,y)
print('coefficient of determination: ',r_sq)
print('intercept:', model.coef_)
new_model = LinearRegression().fit(x,y.reshape((-1,1)))
print('intercept:', new_model.intercept_)
print('Slope:', new_model.coef_)
y_pred = model.predict(x)
print('Predicted Response: ', y_pred, sep = '\n')
y_pred = model.intercept_ + model.coef_ * x
print('Predicted Response: ', y_pred, sep = '\n')
x_new = np.arange(5).reshape(-1,1)
print(x_new)
y_new = model.predict(x_new)
print(y_new)


# In[12]:


import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(0)
x = np.random.rand(100,1)
y = 2 + 3 * np.random.rand(100,1)
regression_model=LinearRegression()
regression_model.fit(x,y)
y_predicted = regression_model.predict(x)
rmse = mean_squared_error(y,y_predicted)
r2 = r2_score(y,y_predicted)
print('Slope:', regression_model.coef_)
print('Intercept: ', regression_model.intercept_)
print('Root Mean Squared Error: ', rmse)
print('R2 Score: ', r2)
plt.scatter(x,y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y_predicted, color ='r')
plt.show()

y_actual = y
y_pred = y_predicted
m = 10
mse = np.sum((y_pred - y_actual)**2)
rmse = np.sqrt(mse/m)
ssr = np.sum((y_pred - y_actual)**2)
sst = np.sum((y_actual - np.mean(y_actual))**2)
r2_score = 1 - (ssr/sst)


# In[16]:


import numpy as np
from sklearn.linear_model import LinearRegression
x = [[0,1],[5,1],[15,2],[25,5],[35,11],[45,15],[55,34],[60,35]]
y = [4,5,20,14,32,22,38,43]
x,y = np.array(x), np.array(y)
print(x)
print(y)
model = LinearRegression().fit(x,y)
r_sq = model.score(x,y)
print('Coeffiecient of determination: ',r_sq)
print('Intercept:' , model.intercept_)
print('Slope: ', model.coef_)
y_pred = model.predict(x)
print('Predicted response: ', y_pred, sep = '\n')
y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
print('predicted respose: ', y_pred, sep = '\n')
x_new = np.arange(10).reshape((-1,2))
print(x_new)


# In[17]:


from sklearn.metrics import confusion_matrix
y_actu = [2,0,2,2,0,1,1,2,2,0,1,2]
y_pred = [0,0,2,1,0,2,1,0,2,0,2,2]
confusion_matrix(y_actu, y_pred)


# In[20]:


import pandas as pd
y_actu = pd.Series([2,0,2,2,0,1,1,2,2,0,1,2], name = 'Actual')
y_pred = pd.Series([0,0,2,1,0,2,1,0,2,0,2,2], name = 'Predicted')
df_confusion = pd.crosstab(y_actu,y_pred)
print(df_confusion)


# In[23]:


import pandas as pd
y_actu = pd.Series([2,0,2,2,0,1,1,2,2,0,1,2])
y_pred = pd.Series([0,0,2,1,0,2,1,0,2,0,2,2])
df_confusion = pd.crosstab(y_actu,y_pred,rownames=['Actual'], colnames=['Predicted'], margins = True)
print(df_confusion)
print('')
df_conf_norm = df_confusion/df_confusion.sum(axis=1)
print(df_conf_norm)


# In[25]:


import matplotlib.pyplot as plt
import matplotlib as mpl
def plot_confusion_matrix(df_confusion, title = 'Confusion Matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns,rotation=45)
    plt.xticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
plot_confusion_matrix(df_confusion)


# In[ ]:




