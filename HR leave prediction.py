#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('C:/Users/JATIN SINGH KANYAL/Dropbox/My PC (LAPTOP-70ACV0E9)/Desktop/new/py/ML/7_logistic_reg/Exercise/HR_comma_sep.csv')


# In[3]:


df


# In[8]:


df.columns


# In[9]:


df


# In[10]:


df.isnull().sum()


# In[11]:


df['Department'].unique()


# In[12]:


df=df.drop(['Department'],axis='columns')


# In[13]:


df.salary.unique()


# In[14]:


dum=pd.get_dummies(df['salary'])


# In[15]:


dum


# In[16]:


df


# In[17]:


df=pd.concat([df,dum],axis='columns')


# In[18]:


df


# In[19]:


df=df.drop(['salary','medium'],axis='columns')


# In[20]:


df


# In[21]:


from sklearn.model_selection import train_test_split 


# In[22]:


X=df.drop(['left'],axis='columns')


# In[42]:


X_train,X_test,y_train,y_test=train_test_split(X,df.left,test_size=0.1)


# In[43]:


from sklearn.linear_model import LogisticRegression


# In[44]:


model=LogisticRegression()


# In[45]:


model.fit(X_train,y_train)


# In[47]:


model.predict(X_test)


# In[49]:


model.score(X_test,y_test)


# In[ ]:




