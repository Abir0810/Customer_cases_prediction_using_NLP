#!/usr/bin/env python
# coding: utf-8

# In[163]:


import pandas as pd
import numpy as np


# In[164]:


df= pd.read_csv(r"E:\AI\data.csv",encoding='unicode_escape')


# In[165]:


df= pd.read_csv(r"E:\AI\data.csv",encoding='latin1',usecols=['Customer Message','Cases'])


# In[166]:


df.head(5)


# In[167]:


df['Cases'].value_counts()


# In[168]:


import re


# In[169]:


def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = re.sub(r'https?://[A-Za-z0-9./]+', '', x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x


# In[170]:


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(x):
    return TAG_RE.sub('', x)


# In[171]:


df['Customer Message']=df['Customer Message'].apply(lambda x: get_clean(x))


# In[172]:


df.head()


# In[173]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[174]:


from sklearn.model_selection import train_test_split


# In[175]:


from sklearn.svm import LinearSVC


# In[176]:


from sklearn.metrics import classification_report


# In[177]:


tfidf=TfidfVectorizer(max_features=2000,ngram_range=(1,3),analyzer='char')


# In[178]:


X=tfidf.fit_transform(df['Customer Message'])
y=df['Cases']


# In[179]:


X.shape


# In[180]:


y.shape


# In[181]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[182]:


X_train.shape


# In[183]:


Clf=LinearSVC(C=20,class_weight='balanced')


# In[184]:


Clf.fit(X_train,y_train)


# In[185]:


y_pred=Clf.predict(X_test)


# In[186]:


print(classification_report(y_test,y_pred))


# In[187]:


x='Refund please'
x=get_clean(x)
vec=tfidf.transform([x])
Clf.predict(vec)


# In[188]:


Clf.score(X_train,y_train)


# In[189]:


from sklearn.metrics import accuracy_score


# In[190]:


accuracy_score(y_test,y_pred)


# In[194]:


from sklearn import tree


# In[195]:


logmodel = tree.DecisionTreeClassifier()


# In[196]:


logmodel.fit(X_train, y_train)


# In[197]:


predictions = logmodel.predict(X_test)


# In[198]:


from sklearn.metrics import accuracy_score


# In[199]:


accuracy_score(y_test,predictions)


# In[200]:


from sklearn.metrics import classification_report


# In[201]:


classification_report(y_test,predictions)


# In[215]:


x='Refund please'
x=get_clean(x)
vec=tfidf.transform([x])
logmodel.predict(vec)


# In[ ]:





# In[ ]:




