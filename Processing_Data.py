#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[2]:


df = pd.read_csv("Simple_Data_with_NID.csv")
df.tail(10)


# In[3]:


d = pd.factorize(df['NameId'])[0]


# In[4]:


df['nid'] = d


# In[5]:


df.isnull().sum()


# In[6]:


df.shape


# In[7]:


df = df.fillna('')
df


# In[8]:


#nltk.download()


# In[9]:


df.dropna(axis='index', how='all')


# In[10]:


#Convert to lower case all
import string
df['Text'] = [doc.lower()for doc in df['Text']]
print(df['Text'][:10])


# In[11]:


#Tokenization
from nltk.tokenize import word_tokenize
tokenized_doc = [word_tokenize(doc) for doc in df['Text']]
# print(tokenized_doc)


# In[12]:


custom_abbr = {
    "u" : "you",
    "k" : "ok",
    "hlw" : "hello",
    "idk" :  "i do not mind",
    "ur": "Your",
    "im": "i am",
    "sx": "sex",
    "yaa": "yet another acronym",
    "sxy": "sexy",
    "yarly": "ya, really? ",
    "yas": "meaning praise",
    "ybic": "your brother in Christ",
    "ybs": "you'll be sorry",
    "ygg": "you go girl",
    "ygtbkm": "you have got to be kidding me",
    "yl":"young lady",
    "ymmv":"your mileage may vary",
    "gd":"good night",
    "m9":"morning",
    "gdn8":"good night",

    
}
counter = 0
for index, text_list in enumerate(tokenized_doc):
    text_list = [custom_abbr[abb] if abb in custom_abbr.keys() else abb for abb in text_list]
    tokenized_doc[index] = text_list


# In[13]:


from nltk.tokenize import sent_tokenize
sent_token = [sent_tokenize(doc) for doc in df['Text']]
# print(sent_token)


# In[14]:


#Punctuation Remove frome sentance
regex = re.compile('[%s]' % re.escape(string.punctuation))
tokenize_doc_punc =[]
for review in tokenized_doc:
    new_review =[]
    for token in review:
        new_token = regex.sub(u'' , token)
        if not new_token == u'':
            new_review.append(new_token)
    tokenize_doc_punc.append(new_review)
#print(tokenize_doc_punc)     #tokenize_doc_punc this is total stirng vatiable


# In[15]:


#removing stopwords
from nltk.corpus import stopwords
tokenize_doc_no_stopwords = []
for doc in tokenize_doc_punc:
    new_term_vector = []
    for word in doc:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)
    tokenize_doc_no_stopwords.append(new_term_vector)
#print(tokenize_doc_no_stopwords) #String Varoable -tokenize_doc_no_stopwords


# In[16]:


#Streming Lemnatization
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
wordnet = WordNetLemmatizer()

preprocessed_text = []
Data_ne =[]
for doc in tokenize_doc_no_stopwords:
    final_text = []
    for word in doc:
        #final_text.append(porter.stem(word))
        final_text.append(wordnet.lemmatize(word))
    preprocessed_text.append(' '.join(final_text))
    Data_ne.append(final_text)
   
print(preprocessed_text) #preprocessed_text -Final text


# In[17]:


df['Level_Text'] = Data_ne
df['Text'] = preprocessed_text
#df['Text'][:10]
df['Level_Text'][:5]


# In[18]:



sentimentIntensityAnalyzer = SentimentIntensityAnalyzer()

negetive = []
neutral = []
positive = []
bully = []
for index, row in df.iterrows():
    scores = sentimentIntensityAnalyzer.polarity_scores(str(row['Text']))
    
    negetive.append(scores['neg'])
    neutral.append(scores['neu'])
    positive.append(scores['pos'])
    
    if scores['neg'] and (scores['neg'] >= scores['neu'] or scores['neu'] >= scores['pos']):
        bully.append(1)  #Bullying_True
    else:
        bully.append(0)  #Bullying_False

df['Negetive'] = negetive
df['Neutral'] = neutral
df['Positive'] = positive
df['bully'] = bully
df.to_csv('TrainData.csv', index=False)

df.head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




