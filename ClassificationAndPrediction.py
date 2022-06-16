#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv("TrainData.csv")
df.head(300)


# In[3]:


df['bully'].value_counts()


# In[4]:


sns.countplot(df['bully'])


# In[5]:


df["bully"].value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True, startangle=140)


# In[6]:


df["bully"].hist(bins=3)


# In[7]:


df.describe()


# In[8]:


df.corr()


# In[9]:


sns.heatmap(df.corr(), cmap ='RdYlGn', linewidths = 0.30, annot = True)


# In[10]:


X = df['Level_Text'].values
Y = df['bully'].values


# In[11]:


print(X)


# In[12]:


print(Y)


# In[13]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

print(X)


# In[ ]:





# In[14]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y , test_size = 0.2 , stratify = Y,random_state = 2  )


# In[15]:


#from sklearn.metrics import confusion_matrix
#confusion_matrix(Y_train,Y_test)


# In[16]:


#LogisticRegression 
model = LogisticRegression()


# In[17]:


model.fit(X_train,Y_train)


# In[18]:


#Score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction ,Y_train)


# In[19]:


print('Accuracy of training data : ',training_data_accuracy)


# In[20]:


#Score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction ,Y_test)


# In[21]:


print('Accuracy of test data : ',test_data_accuracy)


# In[22]:


#Score of LogisticRegression 
model.score(X_test,Y_test)


# In[23]:


#PERFORMANCE


# In[24]:


pred = model.predict(X_test)


# In[25]:


pred


# In[26]:


ac = accuracy_score(Y_test,pred)


# In[27]:


from sklearn.metrics import confusion_matrix
cf = confusion_matrix(Y_test,pred)
print('Accuracy Score is :', ac)
print('Confusion Matrix')
print('\tPredictions')
print('\t{:>5}\t{:>5}'.format(0,1))
for row_id, real_row in enumerate(cf):
    print('{}\t{:>5}\t{:>5}'.format(row_id, real_row[0], real_row[1]))


# In[28]:


#Confusion Matrix Histogram
sns.set(font_scale=1.5)
def plot_conf_mat(Y_test,pred):
    fig,ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(cf, annot = True, cbar = False)
    plt.xlabel("True Label")
    plt.ylabel("Pred Label")
plot_conf_mat(Y_test,pred)


# In[29]:


#RandomForestClassifier.............................

from sklearn.ensemble import RandomForestClassifier


# In[30]:


Rclf=RandomForestClassifier()


# In[31]:


Rclf.fit(X_train,Y_train)


# In[32]:


X_train_predi = Rclf.predict(X_train)
training_data_accu = accuracy_score(X_train_predi ,Y_train)


# In[33]:


print('Accuracy of training data : ',training_data_accu)


# In[34]:


X_test_predi = Rclf.predict(X_test)
test_data_accu = accuracy_score(X_test_predi ,Y_test)


# In[35]:


print('Accuracy of test data : ',test_data_accu)


# In[36]:


#Score RandomForestClassifier 
Rclf.score(X_test,Y_test)


# In[ ]:





# In[37]:


#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier


# In[38]:


Clf=DecisionTreeClassifier()


# In[39]:


Clf.fit(X_train,Y_train)


# In[40]:


Clf.score(X_test,Y_test)


# In[ ]:





# In[41]:


from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(X_train,Y_train)


# In[42]:


classifier.score(X_test,Y_test)


# In[43]:


from sklearn.neighbors import KNeighborsClassifier  
knn= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
knn.fit(X_train,Y_train)


# In[44]:


knn.score(X_test,Y_test)


# In[45]:


#Multi-layer Perceptron Classifier
from sklearn.neural_network import MLPClassifier
ml = MLPClassifier(solver='lbfgs', alpha=1e-5,
               hidden_layer_sizes=(5, 2), random_state=1)
ml.fit(X_train,Y_train)


# In[46]:


ml.score(X_test,Y_test)


# In[47]:


ndf = pd.DataFrame({
    'name': ['LogisticRegression', 'RandomForest','DecisionTree','SVC', 'KNeighbors','MLPClassifier'],
    'score': [model.score(X_test,Y_test),Rclf.score(X_test,Y_test), Clf.score(X_test,Y_test),classifier.score(X_test,Y_test),knn.score(X_test,Y_test),ml.score(X_test,Y_test)]
})


# In[48]:


ndf.to_csv('score.csv')
ndf = pd.read_csv('score.csv')


# In[ ]:





# In[49]:


# function to add value labels
def addlabels(x,y):
    UPPER_OFFSET = 0.005
    for i in range(len(x)):
        plt.text(i,float('%.2f' % y[i]) + UPPER_OFFSET,float('%.2f' % y[i]))
        

# Text below each barplot with a rotation at 90°
scores = [model.score(X_test,Y_test),Rclf.score(X_test,Y_test), Clf.score(X_test,Y_test),classifier.score(X_test,Y_test),knn.score(X_test,Y_test),ml.score(X_test,Y_test)]
names = ['Logistic Regression', 'Random Forest','DecisionTree','SVC', 'KNeighbors','MLP Classifier']

# calling the function to add value labels
fig = plt.figure(figsize =(10,5))

plt.bar(names, scores)
      
# calling the function to add value labels
addlabels(names, scores)

# giving title to the plot
plt.title("Accuracy Score Test")

# giving X and Y labels
# plt.xlabel("Courses")
# plt.ylabel("Number of Admissions")

# visualizing the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




