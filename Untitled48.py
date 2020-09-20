#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import seaborn as sns


# In[4]:


import sklearn


# In[5]:


import warnings


# In[6]:


warnings.filterwarnings("ignore")


# In[9]:


from sklearn.tree import DecisionTreeClassifier


# In[10]:


from sklearn import tree


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


from sklearn.metrics import confusion_matrix


# In[13]:


from sklearn.metrics import accuracy_score


# In[14]:


from sklearn.model_selection import GridSearchCV


# In[15]:


heartdf = pd.read_csv("Downloads/Heart Disease Dataset.csv")


# In[16]:


heartdf.head()


# In[18]:


heartdf.shape


# In[20]:


heartdf.info()


# In[21]:


heartdf.isnull().sum()


# In[22]:


heartdf.describe()


# In[26]:


heartdf.columns


# In[27]:


num_cols = list(heartdf.columns[0:len(heartdf.columns)-1])
num_cols.remove('sex')


# In[31]:


plt.figure(figsize=(30,15))
for i in enumerate(num_cols):
    plt.subplot(3,4,i[0]+1)
    ax = sns.boxplot(heartdf[i[1]])
    ax.set_xlabel(i[1], fontsize=20)
    
plt.tight_layout()
plt.show()


# In[32]:


heartdf.nunique(axis=0)


# In[36]:


fig = plt.figure(figsize=(25,8))

for i in heartdf["target"].unique():
    # extract the data
    x = heartdf[heartdf["target"] == i]["chol"]
    # plot the data using seaborn
    plt.subplot(1,4,1)
    sns.kdeplot(x, shade=True, label = "{} target".format(i))
plt.title("Density Plot of chol by target")
    
for i in heartdf["target"].unique():
    x = heartdf[heartdf["target"] == i]["trestbps"]
    plt.subplot(1,4,2)
    sns.kdeplot(x, shade=True, label="{} target".format(i))
plt.title("Density plot of tresbps by target")

for i in heartdf["target"].unique():
    x = heartdf[heartdf["target"] == i]["thalach"]
    plt.subplot(1,4,3)
    sns.kdeplot(x, shade=True, label = "{} target".format(i))
plt.title("Density plot of thalach by target")

for i in heartdf["target"].unique():
    x = heartdf[heartdf["target"] == i]["oldpeak"]
    plt.subplot(1,4,4)
    sns.kdeplot(x, shade=True, label = "{} target".format(i))
plt.title("Density plot of oldpeak by target")
plt.tight_layout()
plt.show()


# In[39]:


plt.figure(figsize=(25,8), dpi=80)
plt.subplot(1,2,1)
ax = sns.violinplot(x='sex', y='chol', hue='target', split=True, data=heartdf)
ax.set_title('Distributon of chol for different target by sex', fontsize=15)

plt.subplot(1,2,2)
ay = sns.violinplot(x='sex', y='trestbps', hue='target', split=True, data=heartdf)
ay.set_title('Distributon of trestbps for different target by sex', fontsize=15)

plt.tight_layout()
plt.show()


# In[40]:


plt.figure(figsize=(25,8),dpi=80)
plt.subplot(1,2,1)
ax = sns.violinplot(x = "sex", y = "thalach", hue = "target", split = True, data = heartdf)
ax.set_title('Distribution of thalach for different target by sex', fontsize = 15)

plt.subplot(1,2,2)
ay = sns.violinplot(x = "sex", y = "oldpeak", hue = "target", split = True, data = heartdf)
ay.set_title('Distribution of oldpeak for different target by sex', fontsize = 15)

plt.tight_layout()
plt.show()


# In[41]:


cat_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']


# In[42]:


heart_des = heartdf[heartdf['target']==1]
heart_notdes = heartdf[heartdf['target']==0]


# In[43]:


plt.figure(figsize=(30,15))
for i in enumerate(cat_cols):
    plt.subplot(2,4,i[0]+1)
    ax = heart_des[i[1]].value_counts(normalize=True).plot.barh()
    ax.set_title("Deceased showing by "+i[1],fontsize=15)
plt.show()


# In[44]:


plt.figure(figsize=(30,15))
for i in enumerate(cat_cols):
    plt.subplot(2,4,i[0]+1)
    ax = heart_notdes[i[1]].value_counts(normalize=True).plot.barh()
    ax.set_title("Not deceased showing by "+i[1],fontsize=15)
plt.show()


# In[45]:


df_train,df_test = train_test_split(heartdf,train_size=0.7,random_state=50)


# In[46]:


y_train = df_train.pop('target')
X_train = df_train


# In[47]:


y_test = df_test.pop('target')
X_test = df_test


# In[48]:


def check_model(dt):
    print("train confusion matrix : ",confusion_matrix(y_train,dt.predict(X_train)))
    print("train accuracy score : ",accuracy_score(y_train,dt.predict(X_train)))
    print("__"*50)
    print("test confusion matrix : ",confusion_matrix(y_test,dt.predict(X_test)))
    print("test accuracy score : ",accuracy_score(y_test,dt.predict(X_test)))  


# In[49]:


dt_default = DecisionTreeClassifier(random_state=0)
dt_res = dt_default.fit(X_train,y_train)


# In[50]:


check_model(dt_res)


# In[51]:


check_model(dt_res)


# In[52]:


def tree_graph(dt):

    fig = plt.figure(figsize=(25,20))

    dt_plot = tree.plot_tree(dt,feature_names=X_train.columns,class_names=['Not Deceased','Deceased'],filled=True)


# In[53]:


tree_graph(dt_res)


# In[54]:


params = {'max_depth':[2,3,4,5,6,7,8,9,10],
          'min_samples_split':[5,10,25,50,75,100,150]}


# In[55]:


grid_search = GridSearchCV(estimator=dt_default,param_grid=params,scoring='accuracy',n_jobs=-1,verbose=1)
grid_search.fit(X_train,y_train)


# In[56]:


grid_search.best_estimator_
best_dt = grid_search.best_estimator_


# In[57]:


check_model(best_dt)


# In[58]:


tree_graph(best_dt)


# In[ ]:




