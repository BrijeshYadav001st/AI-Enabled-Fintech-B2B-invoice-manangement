#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('https://raw.githubusercontent.com/BrijeshYadav001st/HighRadius-csv-file-/main/H2HBABBA1492.csv')


# # Understanding the Data

# In[3]:


data.head()


# In[4]:


#To check the data length
data.tail()


# In[5]:


#read the header file
data.columns


# In[6]:


data.shape


# # To check nunique values

# In[7]:


data.nunique()


# # Preprocessing

# # Dropping two column becouse they have Null volues or single Values

# In[8]:


data.drop(['area_business','posting_id'] ,axis=1 , inplace=True)
data


# # To Check Null Values

# In[9]:


data.isnull().sum()


# # Interpolate 

# In[10]:


data.invoice_id.interpolate(inplace=True)


# In[11]:


data.isnull().sum()


# # Create a New dateset

# In[12]:


newdata = data[data.clear_date.isna()].copy()
newdata


# # Dropping null Values

# In[13]:


data.dropna(inplace=True)
data


# In[14]:


data.nunique()


# In[15]:


data.isnull().sum()


# In[16]:


data.drop(['isOpen'] ,axis=1 , inplace=True)
data


# In[17]:


data.buisness_year .value_counts()


# # Corrency Conversation

# In[18]:


data['actual_open_amount'] = data['total_open_amount'].where(data['invoice_currency']=='USD', data['total_open_amount'] * 0.8)


# In[19]:


data


# In[20]:


data.drop(['invoice_currency'],axis =1,inplace = True)
data


# # To change the formate of date in %Y%m%d

# In[21]:


data['clear_date']=pd.to_datetime(data['clear_date']).dt.normalize()
data


# In[22]:


data['posting_date']=pd.to_datetime(data['posting_date']).dt.normalize()
data


# In[23]:


#change due_in_date format
data['due_in_date']=pd.to_datetime(data['due_in_date'],format='%Y%m%d')
data


# # delay_days = clear_date - due_in_date

# In[24]:


data['delay_days'] = pd.to_numeric((data['clear_date']- data['due_in_date']).dt.days, downcast='integer')
data


# In[25]:


newdata1 = data[data.due_in_date.isna()].copy()
newdata


# In[26]:


data.drop(['clear_date','due_in_date'] ,axis=1 , inplace=True)
data


# In[27]:


data.shape


# # Sorting and reset the values 

# In[28]:


data.sort_values(by=['posting_date'],inplace=True)
data


# In[29]:


data.set_index
data


# In[30]:


data.reset_index(drop=True, inplace=True)
data


# # Dropping total_open_amount becouse total_open_amount and actual_open_amount is similar

# In[31]:


data.drop('total_open_amount' ,axis=1 , inplace=True)
data


# # Spliting

# In[32]:


X = data.drop('delay_days',axis=1)


# In[33]:


y = data['delay_days']


# In[34]:


from sklearn.model_selection import train_test_split
X_train,X_inter_test,y_train,y_inter_test = train_test_split(X,y,test_size=0.3,random_state=0 , shuffle = False)


# In[35]:


X_val,X_test,y_val,y_test = train_test_split(X_inter_test,y_inter_test,test_size=0.5,random_state=0 , shuffle = False)


# In[36]:


X_train.shape , X_val.shape , X_test.shape


# In[37]:


X_train.info()


# In[38]:


X_val.info()


# In[39]:


X_test.info()


# # EDA(Exploratory data analysis)

# In[40]:


y_train


# In[41]:


sns.distplot(y_train)


# In[42]:


X_train.info()


# In[43]:


X_train.merge(y_train,on=X_train.index)


# In[44]:


X_train.info()


# In[45]:


sns.scatterplot(data=X_train.merge(y_train,on=X_train.index) , x = 'delay_days',y='posting_date')


# In[46]:


sns.scatterplot(data=X_train.merge(y_train,on=X_train.index) , x = 'document_create_date' , y='delay_days')


# # Feature Engineering

# In[47]:


data.columns


# In[48]:


X_train.nunique()


# In[49]:


X_train.drop('buisness_year' ,axis=1 , inplace=True)
X_val.drop('buisness_year' ,axis=1 , inplace=True)
X_test.drop('buisness_year' ,axis=1 , inplace=True)


# In[50]:


X_train.info()


# # Categorical Values --lavel encoding

# In[51]:


from sklearn.preprocessing import LabelEncoder


# # Business_code

# In[52]:


business_code_enc = LabelEncoder()
business_code_enc.fit(X_train['business_code'])
X_train['business_code_enc'] = business_code_enc.transform(X_train['business_code'])


# In[53]:


X_train[['business_code_enc','business_code']]


# In[54]:


X_val['business_code_enc'] = business_code_enc.transform(X_val['business_code'])
X_val[['business_code_enc','business_code']]


# In[55]:


X_test['business_code_enc'] = business_code_enc.transform(X_test['business_code'])
X_test[['business_code_enc','business_code']]


# In[56]:


X_val


# In[57]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(X_train['document type'])
X_train['document_type_enc'] = enc.transform(X_train['document type'])


# In[58]:


X_train[['document_type_enc','document type']]


# In[59]:


X_val['document_type_enc'] = enc.transform(X_val['document type'])
X_val[['document_type_enc','document type']]


# In[60]:


X_test['document_type_enc'] = enc.transform(X_test['document type'])
X_test[['document_type_enc','document type']]


# # Dropping  document type,business_code,cust_number,name_customer and cust_payment_terms

# In[61]:


X_train.drop(['document type','business_code','cust_number','name_customer','cust_payment_terms'],axis=1,inplace = True)
X_test.drop(['document type','business_code','cust_number','name_customer','cust_payment_terms'],axis=1,inplace = True)
X_val.drop(['document type','business_code','cust_number','name_customer','cust_payment_terms'],axis=1,inplace = True)


# In[62]:


X_train.info()


# In[63]:


X_train.nunique()


# # Creating two column by using posting_date (business_days,business_month)

# In[64]:


X_train['business_days'] = pd.DatetimeIndex(X_train['posting_date']).day
X_train['business_month'] = pd. DatetimeIndex(X_train['posting_date']).month
X_train


# In[65]:


X_val['business_days'] = pd.DatetimeIndex(X_val['posting_date']).day
X_val['business_month'] = pd. DatetimeIndex(X_val['posting_date']).month
X_val


# In[66]:


X_test['business_days'] = pd.DatetimeIndex(X_test['posting_date']).day
X_test['business_month'] = pd. DatetimeIndex(X_test['posting_date']).month
X_test


# # Dropping Posting date

# In[67]:


X_train.drop(['posting_date'],axis=1,inplace = True)
X_test.drop(['posting_date'],axis=1,inplace = True)
X_val.drop(['posting_date'],axis=1,inplace = True)


# In[68]:


X_train


# In[69]:


X_train.info()


# # Feature Selection

# In[70]:


colormap = plt.cm.RdBu
plt.figure(figsize=(15,10))

sns.heatmap(X_train.merge(y_train , on = X_train.index ).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# In[71]:


X_train.drop(['document_create_date.1','invoice_id','baseline_create_date'],axis=1,inplace = True)
X_test.drop(['document_create_date.1','invoice_id','baseline_create_date'],axis=1,inplace = True)
X_val.drop(['document_create_date.1','invoice_id','baseline_create_date'],axis=1,inplace = True)


# In[72]:


X_train.info()


# In[73]:


colormap = plt.cm.RdBu
plt.figure(figsize=(15,10))

sns.heatmap(X_train.merge(y_train , on = X_train.index ).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# # Modeling

# # Linear Regression

# In[74]:


from sklearn.linear_model import LinearRegression
base_model = LinearRegression()
base_model.fit(X_train,y_train)


# In[75]:


lr_predict = base_model.predict(X_val)
lr_predict


# In[76]:


y_val


# In[148]:


pd.DataFrame(zip(lr_predict, y_val),columns=['Predicted','Actuals'])


# In[78]:


from sklearn.metrics import mean_squared_error as mse 
mse(y_val,lr_predict,squared=False)


# In[79]:


lr_predict2= base_model.predict(X_test)


# In[80]:


mse(y_test,lr_predict2,squared=False)


# In[81]:


pd.DataFrame(zip(y_val , y_test),columns=['Predicted','Actuals'])


# In[82]:


lr_accuracy = base_model.score(X_test,y_test)


# In[83]:


print(lr_accuracy * 100)


# # Support Vector Regression

# In[84]:


from sklearn.svm import SVR


# In[85]:


svr_model = SVR(kernel = 'rbf')


# In[86]:


svr_model.fit(X_train,y_train)


# In[87]:


svr_predict1 = svr_model.predict(X_val)
svr_predict1


# In[88]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_val , svr_predict1,squared = False)


# In[89]:


svr_accuracy = svr_model.score(X_test,y_test)


# In[90]:


print(svr_accuracy *100)


# # Decision Tree Regressor

# In[91]:


from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor(max_depth=6)
tree_model.fit(X_train,y_train)


# In[92]:


tr_predict = tree_model.predict(X_val)
tr_predict


# In[93]:


mse(y_val,tr_predict,squared=False)


# In[94]:


from sklearn.metrics import mean_squared_error as mse
mse(y_test,tr_predict,squared=False)


# In[95]:


tr_predict1 = tree_model.predict(X_test)


# In[96]:


mse(y_test,tr_predict1,squared=False)


# In[97]:


tr_accuracy = tree_model.score(X_test,y_test)


# In[98]:


print(tr_accuracy * 100)


# # Random Forest Regressor

# In[99]:


from sklearn.ensemble import RandomForestRegressor


# In[100]:


rfr_model = RandomForestRegressor(n_estimators = 100 , random_state = 0)


# In[101]:


rfr_model.fit(X_train,y_train)


# In[102]:


rfr_predict1 = rfr_model.predict(X_val)


# In[103]:


from sklearn.metrics import mean_squared_error as mse
mse(y_val,rfr_predict1,squared =False)


# In[104]:


rfr_predict2 = rfr_model.predict(X_test)
mse(y_test,rfr_predict2,squared = False)


# In[105]:


rfr_accuracy = rfr_model.score(X_test,y_test)


# In[106]:


print(rfr_accuracy * 100)


# 

# # Making Models with LR , SVR , DTR , RFR

# In[107]:


Algn = ['Linear Regression','SVR','Decision Tree Regression','Random Forest Regressor','XGB Regresson']


# In[108]:


mse = [mse(y_test,lr_predict2,squared = False),mse(y_test,svr_predict1, squared=False),
       mse(y_test,tr_predict1,squared=False),mse(y_test, rfr_predict2, squared=False),]


# In[109]:


accuracy = [lr_accuracy,svr_accuracy,tr_accuracy,rfr_accuracy]


# In[110]:


models = pd.DataFrame(list(zip(Algn,mse,accuracy)))
                      


# In[111]:


models


# In[112]:


model = pd.DataFrame(list(zip(Algn,mse,accuracy)))
Frame=pd.DataFrame(models.values, columns = ["Algn", "mse", "accuracy"])
Frame


# In[113]:


X_train.info()


# In[114]:


newdata.info()


# In[115]:


newdata1.info()


# In[116]:


newdata.nunique()


# In[117]:


newdata.isnull().sum()


# In[118]:


#Dropping the clear date
newdata.drop(['clear_date'] ,axis=1 , inplace=True)
data


# In[119]:


#Dropping all not usefull value
newdata.drop(['baseline_create_date','name_customer','isOpen','document_create_date.1','buisness_year',
             'cust_number','cust_payment_terms','invoice_id'] ,axis=1 , inplace=True)
data


# In[120]:


newdata.nunique()


# In[121]:


#currency conversation
newdata['actual_open_amount'] = newdata['total_open_amount'].where(newdata['invoice_currency']=='USD', 
                                                             newdata['total_open_amount'] * 0.8)


# In[122]:


newdata


# In[123]:


#Droping the invoice_currency,total_open_amount
newdata.drop(['invoice_currency','total_open_amount'],axis =1,inplace = True)
newdata


# In[124]:


#preprocessing
newdata['posting_date']=pd.to_datetime(data['posting_date']).dt.normalize()
newdata


# In[125]:


#Create two new column by using posting date
newdata['business_days'] = pd.DatetimeIndex(newdata['posting_date']).day
newdata['business_month'] = pd. DatetimeIndex(newdata['posting_date']).month


# In[126]:


#Droping the osting_date
newdata.drop(['posting_date'],axis=1,inplace = True)


# In[127]:


#convert due_in_date in new format and also data_temp for predication
newdata['due_in_date']=pd.to_datetime(newdata['due_in_date'],format='%Y%m%d')
data_temp = pd.DataFrame(newdata['due_in_date'])


# In[128]:


#Dropping the due_in_date
newdata.drop(['due_in_date'],axis=1,inplace = True)


# In[129]:


#label Encoding for business code
from sklearn.preprocessing import LabelEncoder
business_code_enc = LabelEncoder()
business_code_enc.fit(newdata['business_code'])
newdata['business_code_enc'] = business_code_enc.transform(newdata['business_code'])


# In[130]:


#Dropping the business_code

newdata.drop(['business_code'],axis=1,inplace = True)


# In[131]:


#label encoding for document type
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(newdata['document type'])
newdata['document_type_enc'] = enc.transform(newdata['document type'])


# In[132]:


#dropping the document type
newdata.drop(['document type'],axis=1,inplace = True)


# In[133]:


newdata.info()


# In[134]:


X_train.info()


# In[135]:


#check null values in newdata
newdata.isnull().sum()


# In[136]:


#check null values in newdata1
newdata1.isnull().sum()


# In[137]:


#Droping Null Values 
newdata.dropna(inplace=True)
newdata


# # Data Predication 

# In[138]:


predication_result = base_model.predict(newdata)


# 

# In[139]:


predication_result = pd.Series(predication_result,name='delay_days')


# In[140]:


newdata = newdata.merge(predication_result,on=newdata.index)


# In[141]:


newdata.head()


# In[142]:


#concat 
result=pd.concat([newdata,data_temp],axis=1,join = "inner")


# In[143]:


#result = pd.concat([newdata,data_temp], axis=0, ignore_index=True)


# In[144]:


result.head()


# In[145]:


result['payment_date']=pd.to_datetime(result.due_in_date)+pd.to_timedelta(pd.np.ceil(result.delay_days),unit="D")
result.head()


# # Creating a data Bucket

# In[146]:


result['Aging_Bucket'] =""
result.loc[result['delay_days'].apply(int)>60,'Aging_Bucket']='>60'
result.loc[(result['delay_days'].apply(int)>=46) &(result['delay_days'].apply(int) <= 60),'Aging_Bucket'] = '46-60'
result.loc[(result['delay_days'].apply(int) >= 31) & (result['delay_days'].apply(int) <= 45),'Aging_Bucket'] = '31-45'
result.loc[(result['delay_days'].apply(int) >= 16) & (result['delay_days'].apply(int) <= 30),'Aging_Bucket'] = '16-30'
result.loc[(result['delay_days'].apply(int) >= 1 ) & (result['delay_days'].apply(int) <= 15),'Aging_Bucket'] = '1-15'
result.loc[result['delay_days'].apply(int) <= 0, 'Aging_Bucket'] = 'No delay'                                                    


# # Final Result

# In[147]:


result


# In[ ]:




