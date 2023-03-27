#!/usr/bin/env python
# coding: utf-8

# # Mini Project 2

# # Introduction
Seed quality is definitely influential in crop production. Therefore, seed classification is essential for both marketing and production to provide the principles of sustainable agricultural systems.

In this notbook we try different algorithms to classify the most well-known 7 types of beans in Turkey; Barbunya, Bombay, Cali, Dermason, Horoz, Seker and Sira .
# # Features Information:
1.ID: An ID for this instance
2.Area (A): The area of a bean zone and the number of pixels within its boundaries.
3.Perimeter (P): Bean circumference is defined as the length of its border.
4.MajorAxisLength (L): The distance between the ends of the longest line that can be drawn from a bean.
5.MinorAxisLength (l): The longest line that can be drawn from the bean while standing perpendicular to the main axis.
6.AspectRatio (K): Defines the relationship between L and l.
7.Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.
8.ConvexArea (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.
9.EquivDiameter (Ed): The diameter of a circle having the same area as a bean seed area.
10.Extent (Ex): The ratio of the pixels in the bounding box to the bean area.
11.Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.
12. (R): Calculated with the following formula: (4* pi * A)/(P^2)
13.Compactness (CO): Measures the roundness of an object: Ed/L
14. (SF1): L/d
15.ShapeFactor2 (SF2): l/d
16.ShapeFactor3 (SF3): 4A/(L^2 * pi)
17.ShapeFactor4 (SF4): 4A/(L* l * pi)
18.y: The class of the bean. It can be any of BARBUNYA, SIRA, HOROZ, DERMASON, CALI, BOMBAY, and SEKER.
# # Importing required libraries

# In[1]:


import numpy as np  #linear algebra
import pandas as pd # a data processing and CSV I/O library

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', color_codes=True)


# # Reading the dataset

# In[2]:


dry =pd.read_csv("data.csv")
dry.head()


# # Preprocessing

# In[3]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
columns = ["Class"]

for col in columns:
    dry[col] = le.fit_transform(dry[col])
    print(le.classes_)
      
dry.head()


# # Exploratory Data Analysis (EDA)

# In[4]:


sns.displot(dry['Eccentricity'])


# In[5]:


print(dry['Class'].value_counts())
_ = sns.countplot(x='Class', data=dry)


# In[6]:


Numeric_cols = dry.drop(columns=['Class']).columns

fig, ax = plt.subplots(4, 4, figsize=(15, 12))
for variable, subplot in zip(Numeric_cols, ax.flatten()):
    g=sns.histplot(dry[variable],bins=30, kde=True, ax=subplot)
    g.lines[0].set_color('crimson')
    g.axvline(x=dry[variable].mean(), color='m', label='Mean', linestyle='--', linewidth=2)
plt.tight_layout()


# In[7]:


sns.jointplot(x='EquivDiameter', y='Area', data=dry, kind="hex")


# In[8]:


sns.kdeplot(dry['Perimeter'])


# In[11]:


sns.pairplot(dry.drop(['AspectRation','Eccentricity','ConvexArea',	
    'EquivDiameter',	
    'Extent',	
    'Solidity',	
    'roundness',	
    'Compactness',	
    'ShapeFactor1',	
    'ShapeFactor2',	
    'ShapeFactor3',	
    'ShapeFactor4'], axis=1), hue='Class', height=3, diag_kind='hist')


# In[13]:


plt.figure(figsize=(12,12))
sns.heatmap(dry.corr("pearson"),vmin=-1, vmax=1,cmap='coolwarm',annot=True, square=True)


# # Defining X and y

# In[14]:


X = dry[[
    'Area',
    'Perimeter',
    'MajorAxisLength',	
    'MinorAxisLength',	
    'AspectRation',	
    'Eccentricity',
    'ConvexArea',	
    'EquivDiameter',	
    'Extent',	
    'Solidity',	
    'roundness',	
    'Compactness',	
    'ShapeFactor1',	
    'ShapeFactor2',	
    'ShapeFactor3',	
    'ShapeFactor4']].values
y = dry[['Class']]


# # Training Models
# 

# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report, ConfusionMatrixDisplay,precision_score,recall_score, f1_score,roc_auc_score,roc_curve

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2)


# In[16]:


models={
    "Logisitic Regression" :LogisticRegression(max_iter=20000),
    "Decision Tree" :DecisionTreeClassifier(),
    "Random Forest":RandomForestClassifier(),
    "Support Vector Machine": svm.SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3)
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train,y_train.values.ravel()) # Train Model
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred =  model.predict(X_test)

  # Test set performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred) 
    model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') 
    model_test_precision = precision_score(y_test, y_test_pred , average='weighted') 
    model_test_recall  = recall_score(y_test, y_test_pred,average='weighted') 

  # Training set performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred) 
    model_train_f1 = f1_score(y_train, y_train_pred, average= 'weighted') 
    model_train_precision = precision_score(y_train, y_train_pred,average='weighted') 
    model_train_recall = recall_score(y_train, y_train_pred,average='weighted') 

    print(list(models.keys())[i])

    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:4f}'.format(model_train_f1))
    print('- Precision: {:4f}'.format(model_train_precision))
    print('- Recall: {:4f}'.format(model_train_recall))

    print('----------------------------------')

    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy) )
    print('- Fl score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))


    print('='*35)
    print('\n')


# In[ ]:




