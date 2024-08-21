#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


# import the data
ExasensData = "Exasens.csv"
ExasensDF = pd.read_csv("Exasens.csv")
ExasensDF.head(30) # show first 30 rows


# In[3]:


# drop the first two rows
# ExasensDF = ExasensDF.drop([0,1], axis = 0)
ExasensDF.head(5)


# In[4]:


# casting the real and imaginary (permittivity reading) part columns into float64 data
# use pandas'.to_numeric() function to convert string to int
ExasensDF['ImagMin'] = pd.to_numeric(ExasensDF['ImagMin'])
ExasensDF['RealMin'] = pd.to_numeric(ExasensDF['RealMin'])

ExasensDF['ImagAvg'] = pd.to_numeric(ExasensDF['ImagAvg'])
ExasensDF['RealAvg'] = pd.to_numeric(ExasensDF['RealAvg'])

ExasensDF.head(5)


# In[5]:


# check data types
ExasensDF.dtypes


# In[7]:


# clean the data by removing any participants with NaN values from the dataframe
# drop any participants (rows from the Dataframe) with 'Real Part', 'Imaginary Part', or 'Diagnosis' column values of NaN.

ExasensDF = ExasensDF[ExasensDF['ImagAvg'].notna()]
ExasensDF = ExasensDF[ExasensDF['RealAvg'].notna()]
ExasensDF = ExasensDF[ExasensDF['ImagMin'].notna()]
ExasensDF = ExasensDF[ExasensDF['RealMin'].notna()]
ExasensDF = ExasensDF[ExasensDF['Diagnosis'].notna()]
ExasensDF.head(5)


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

# convert into their absolute values for ease of scatterplot interpretation
ExasensDF['ImagMin']=ExasensDF['ImagMin'].abs()
ExasensDF['ImagAvg']=ExasensDF['ImagAvg'].abs()

ExasensDF['RealMin']=ExasensDF['RealMin'].abs()
ExasensDF['RealAvg']=ExasensDF['RealAvg'].abs()

plt.close();
sns.set_style("whitegrid");
sns.pairplot(ExasensDF, hue="Diagnosis", height=3);
plt.show()


# In[9]:


# Histograms
ExasensDF.hist(bins=10,figsize=(9, 8))
plt.show()


# In[10]:


# convert the Pandas dataframe above into two numpy arrays
ExasensFeatures = np.asarray(ExasensDF[['RealMin','ImagMin', 'RealAvg', 'ImagAvg','Gender','Age','Smoking']])
ExasensTarget = np.asarray(ExasensDF['Diagnosis'])

# split the whole dataset into training and testing sets for higher validity: 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(ExasensFeatures, ExasensTarget, test_size=0.2, random_state=2)


# In[11]:


# build the KNN Model with Scikit Learn and evaluate it, plotting the accuracy results: 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 
import matplotlib.pyplot as plt
ExasensKNNScores = []
ExasensNeighborsAccuracyArray = np.zeros(19)
kTestIterationList = range(1,20)
for k in kTestIterationList:

    ExasensNeighbors = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
    DiagnosesPredictions=ExasensNeighbors.predict(X_test)
    ExasensNeighborsAccuracyArray[k-1]=metrics.accuracy_score(y_test, DiagnosesPredictions)
    ExasensKNNScores.append(metrics.accuracy_score(y_test, DiagnosesPredictions))
    
plt.plot(kTestIterationList, ExasensKNNScores)
plt.xlabel('Value of K for Respiratory Illness Diagnosis KNN')
plt.ylabel('KNN Testing Accuracy')

# k=9 highest accuracy


# In[12]:


ExasensNeighbors = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
DiagnosesPredictions=ExasensNeighbors.predict(X_test)
#print(DiagnosesPredictions)
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, DiagnosesPredictions))
print("")
print("Classification Report:")
print(metrics.classification_report(y_test, DiagnosesPredictions))
print("Accuracy: ", metrics.accuracy_score(y_test, DiagnosesPredictions))


# In[ ]:




