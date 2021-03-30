#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import nltk
from nltk.stem import PorterStemmer , WordNetLemmatizer
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
import re
import matplotlib.pyplot as py
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report


# In[8]:


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# In[5]:


df = pd.read_csv(r'C:\Users\rrajn\Python\PROJECTS\Sentiment_analysis\IMDB dataset.csv')


# In[6]:


df


# In[11]:


corpus=[]
for i in range(len(df)):
    word = re.sub('[^a-zA-Z]',' ',df['review'][i])
    word = word.lower()
    word = word.split()
    word = [stemmer.stem(j) for j in word if j not in stopwords.words('english')]
    word = ' '.join(word)
    corpus.append(word)


# In[12]:


cv = CountVectorizer(max_features=5000 , ngram_range=(1,3))


# In[13]:


X =  cv.fit_transform(corpus).toarray()


# In[14]:


y = pd.get_dummies(df['sentiment'])
y = y.iloc[:,1].values


# In[15]:


X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=123 , test_size=0.2)


# In[16]:


spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)


# In[27]:


print(confusion_matrix(y_test , y_pred))


# In[17]:


print(accuracy_score(y_test , y_pred))


# In[30]:


print(classification_report(y_test , y_pred))


# In[19]:


randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(X_train , y_train)
y_pred_rf = randomclassifier.predict(X_test)


# In[28]:


print(confusion_matrix(y_test , y_pred_rf))


# In[31]:


print(classification_report(y_test , y_pred_rf))


# In[20]:


print(accuracy_score(y_test,y_pred_rf))

