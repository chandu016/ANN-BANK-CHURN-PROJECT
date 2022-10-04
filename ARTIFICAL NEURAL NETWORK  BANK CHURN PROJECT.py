#!/usr/bin/env python
# coding: utf-8

# In[236]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[237]:


df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[238]:


df.head()


# In[239]:


df.dtypes


# # convert object to numeric for Total charges

# In[240]:


df.TotalCharges.values


# In[241]:


df.MonthlyCharges.values


# In[242]:


df[pd.to_numeric(df['TotalCharges'], errors = 'coerce').isnull()]


# # remove the NA values in Totalcharges

# In[243]:


df1 = df[df.TotalCharges!=' ']


# In[244]:


df1.shape


# In[245]:


pd.to_numeric(df1['TotalCharges'])


# # knowing the loyal customer from last 10years through graphs

# In[246]:


## tenure shows the service of bank for customer in months

## first knowing the no of customer leaving the bank and not leaving the bank

bank_churn_no = df1[df1.Churn == 'No'].tenure


# In[247]:


bank_churn_no


# In[248]:


bank_churn_yes = df1[df1.Churn == 'Yes'].tenure


# In[249]:


bank_churn_yes


# In[250]:


plt.figure(figsize = (15,10))

plt.hist([bank_churn_yes, bank_churn_no], color=['green','red'], label=['churn=yes','churn=no'])
plt.legend()
plt.title("CUSTOMER CHURN VISUALIZATION")
plt.xlabel('TENSURE')
plt.ylabel('NO OF CUSTOMERS')


# # NOW WE ARE CHECKING THE MONTHLYCHARGES 

# In[251]:


month_churn_yes = df1[df1.Churn == 'Yes'].MonthlyCharges


# In[252]:


month_churn_no = df1[df1.Churn == 'No'].MonthlyCharges


# In[253]:


plt.figure(figsize =(15,10))
plt.hist([month_churn_yes, month_churn_no], color = ['green','red'], label = ['churn = yes, churn =  no'])
plt.legend()
plt.title("CUSTOMER MONTHLY CHARGES VISUALIZATION")
plt.xlabel('Month;yCharges')
plt.ylabel('No of Customers')


# ## finding the unique values in the labels and encoding action has to perform for the prediction regression

# In[254]:


def print_ojective_columns(df):
    for columns in df1:
        if df[columns].dtypes == 'object':
            print(f'{columns} : {df1[columns].unique()}')


# In[255]:


print_ojective_columns(df1)


# In[256]:


df1.replace('No phone service','No', inplace =True)
df1.replace('No internet service','No', inplace =True)


# In[257]:


df1.MultipleLines.unique()


# In[258]:


print_ojective_columns(df1)


# In[259]:


print_ojective_columns(df1)


# In[279]:


churn_yes_no = ['Partner',
            'Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
               'StreamingMovies','PaperlessBilling','Churn']

for col in churn_yes_no:
    df1[col].replace({'Yes' :1, 'No': 0},inplace = True)


# In[280]:


for col in df1:
    print(f'{col}:{df1[col].unique()}')


# In[281]:


df2 = pd.get_dummies(data = df1, columns = ['InternetService','PaymentMethod','Contract'])


# In[282]:


df2.columns


# In[283]:


df2.dtypes


# In[284]:


df2.drop(['customerID'], axis = 1,  inplace = True)


# In[285]:


df2.columns


# In[294]:


df2.TotalCharges = pd.to_numeric(df2.TotalCharges)


# In[296]:


df2.dtypes


# In[298]:


cols_to_sclae =['tenure','MonthlyCharges','TotalCharges']


# In[300]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit_transform(df2[cols_to_sclae])


# In[304]:


df2.head(20)


# In[306]:


df2.sample(5)


# In[312]:


x = df2.drop('Churn', axis = 'columns')
y = df2['Churn']


# In[314]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 42, test_size = 0.2)


# In[321]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[323]:


import tensorflow as tf


# In[326]:


from tensorflow import keras


# In[335]:



model = keras.Sequential([keras.layers.Dense(20, input_shape =(26,), activation='relu'),
                    
                         keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

model.fit(x_train,y_train, epochs = 100)


# In[339]:


model.evaluate(x_test,y_test)


# In[341]:


yp = model.predict(x_test)


# In[343]:


yp[:5]


# In[345]:


y_test[:5]


# In[349]:


yp_predit = []

for element in yp:
    if element >0.5:
        yp_predit.append(1)
    else:
        yp_predit.append(0)


# In[351]:


yp_predit[:10]


# In[354]:


from sklearn.metrics import confusion_matrix, classification_report


# In[356]:


print(confusion_matrix(y_test, yp_predit))


# In[358]:


print(classification_report(y_test,yp_predit))


# In[362]:


import seaborn as sns

cm = tf.math.confusion_matrix(labels = y_test, predictions = yp_predit)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Prediction')
plt.ylabel('Truth')

