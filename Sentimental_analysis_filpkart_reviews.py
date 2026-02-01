#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing all libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import textblob, string, re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob,Word
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import Word,TextBlob
from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1)


# In[3]:


df = pd.read_csv('data.csv')
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


print("percentage of missing values\n",round(df.isna().sum()/len(df)*100,2))


# In[7]:


# df['Reviewer Name'].fillna('Anonymous',inplace = True)
# df['Review Title'].fillna('No Title',inplace = True)
# df['Place of Review'].fillna(df['Place of Review'].mode()[0],inplace = True)
# df['Up Votes'].fillna(0,inplace = True)
# df['Down Votes'].fillna(0,inplace = True)
# df['Month'].fillna(df['Month'].mode()[0],inplace = True)
df['Review text'].fillna('No Review',inplace = True)


# In[8]:


df.isna().sum()


# In[9]:


df['Review text'].head()


# In[10]:


backup = df['Review text']


# In[11]:


df['Review text'] = df['Review text'].str.lower() # making into lower case
df['Review text'].head()


# In[12]:


df['Review text'] = df['Review text'].str.replace(r"[^\w\s]","",regex=True)
df['Review text'].head() # removing all special punctuation sysmbols


# In[13]:


df['Review text'] = df['Review text'].str.replace(r"\d","",regex=True)
df['Review text'].head()


# In[14]:


sw = stopwords.words('english')


# In[15]:


df['Review text'] = df['Review text'].astype(str)


# In[16]:


df['Review text'] = df['Review text'].apply(lambda x: " ".join(word for word in x.split() 
                                                                           if word not in sw))


# In[17]:


df['Review text'].head()


# In[18]:


lemmatizer = WordNetLemmatizer()


# In[19]:


df['Review text'] = df['Review text'].apply(
    lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]) # lemmatizing the words of each index
)


# In[20]:


df['Review text'].head()


# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix


# In[22]:


X = df['Review text']
df['Sentiment'] = df['Ratings'].apply(lambda x: 'Positive' if x >= 4 else 'Negative')
y = df.Sentiment


# In[23]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)


# In[27]:


# appliying tf-idf
tfidf = TfidfVectorizer(max_features=4800, stop_words='english',ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.fit_transform(X_test)



# In[28]:


models = {
    "logistic regression":LogisticRegression(max_iter=1000,class_weight='balanced'),
    'svc':SVC(),
    'RandomForest':RandomForestClassifier(),
    'multinb':MultinomialNB()
}


# In[33]:


for name,model in models.items():
    model.fit(X_train_tfidf,y_train)
    y_pred = model.predict(X_test_tfidf)
    print(name)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test,y_pred))


# In[ ]:





# At first glance, the models (Logistic Regression, SVM, Random Forest, Naive Bayes) all show around 81% accuracy,
# but that number is misleading because the models are only predicting the “Positive” class.

# In[30]:


df.Sentiment.value_counts()


# In[ ]:





# In[34]:


from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('tfidf',TfidfVectorizer(max_features=4800,ngram_range=(1,2),min_df = 5,stop_words='english')),
    ('clf',LogisticRegression(class_weight='balanced',max_iter = 1000))
])
pipeline.fit(X_train,y_train)


# In[35]:


# export with joblib:
import joblib
joblib.dump(pipeline,'sentiment_pipeline.joblib')


# In[38]:


import streamlit as st, joblib
model = joblib.load('sentiment_pipeline.joblib')
st.title("Review Sentiment")
txt = st.text_area('Paste review')
if st.button('predict'):
    pred = model.predict([txt])[0]
    st.write('Sentiment:',pred)


# In[41]:




# In[ ]:




