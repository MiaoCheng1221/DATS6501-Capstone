
# coding: utf-8

# ## Capstone Project:    Google Analytics Customer Revenue Prediction
# 
# ### Cheng Miao
# 

# In[1]:


import pandas as pd 
import json

# Define COLUMNS
JSON_COLUMNS= ['device', 'geoNetwork', 'totals', 'trafficSource']
print ("Starting read data")
data=pd.read_csv("data.csv",sep=',',header=0,                   converters={column:json.loads for column in JSON_COLUMNS})
print('Load Data in Json, that takes long time')
print ("Done reading data")


# ### Dataset Processing

# In[2]:


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
from pandas.io.json import json_normalize

# json: change into mulitple columns, the object of json_normalize is dict，and the object of json_loads is string
for col in JSON_COLUMNS:
    data_col=json_normalize(data[col])
    data_col.columns = [f"{sub_col}" for sub_col in data_col.columns]
    data = data.drop(col,axis=1).merge(data_col,right_index=True,left_index=True)
print ("end for changing train data")


# In[3]:


# change date format, convenient in statistics

data['date'] = data['date'].astype(str)
data['date'] = data['date'].apply(lambda x:x[:4]+"-"+x[4:6]+"-"+x[6:8])
data['date'] = pd.to_datetime(data['date']) # timestamp
data['month'] = data['date'].apply(lambda x:x.strftime('%Y-%m'))
data['week'] = data['date'].dt.weekday

# transactionRevenue convert
data['transactionRevenue']=data['transactionRevenue'].astype(float).fillna(0)


# In[4]:


# hits、pageviews converting

data['hits'] = data['hits'].astype(float)
data['pageviews'] = data['pageviews'].astype(float)


# In[5]:


data.head()


# In[6]:



# one column has same value, so drop columns
# print those columns who contain same value.

for r in data.columns:
    a = data[r].value_counts()
    if len(a)<2:
        print(r)
        print(a)
        print('----')

# value is constant, get rid of these, I have 38 columns left.      
data = data.drop(['socialEngagementType','browserSize','browserVersion','flashVersion',                  'language','mobileDeviceBranding','mobileDeviceInfo','mobileDeviceMarketingName',                  'mobileInputSelector','operatingSystemVersion','screenResolution','screenColors','cityId',                  'latitude','longitude','networkLocation','visits','adwordsClickInfo.criteriaParameters','campaignCode'],axis=1)


# In[7]:


data.head()


# In[8]:


# drop missing value columns

for r in data.columns:
    a = len(data[r][pd.isnull(data[r])])/len(data)
    if a>0.8:
        print(r)
        print(a)
        print('----')

# starting drop columns
data = data.drop(['adContent','adwordsClickInfo.adNetworkType','adwordsClickInfo.gclId','adwordsClickInfo.isVideoAd','adwordsClickInfo.page','adwordsClickInfo.slot'],axis=1)


# In[9]:


data.head()


# ### EDA: Data Exploration

# In[10]:


# explory data
# relationship between "month, week, source, web brosser, device, country, city, internet, etc" value smaller than 15 and views

x_val = []
for r in data.columns:
    a = data[r].value_counts()
    if len(a)<15: # pick length < 15, Data is too scattered
        x_val.append(r)
        print(a)
        print('...........................................')


# In[11]:


# relationship between with transactionRevenue vs income,take the media as an example for the following picture
import matplotlib.pyplot as plt

for x in x_val:
    b = data['transactionRevenue'].groupby(data[x]).sum()
    print(b)
    print('..................................................')


# In[12]:


# plot graph
data['transactionRevenue'].groupby(data['medium']).sum().plot('bar')


# #### Here can see a distribution of total revenue per user

# In[13]:


import numpy as np

grouped = data.groupby('fullVisitorId')['transactionRevenue'].sum().reset_index()
grouped = grouped.loc[grouped['transactionRevenue'].isna() == False]
plt.hist(np.log(grouped.loc[grouped['transactionRevenue'] > 0, 'transactionRevenue']));
plt.title('Distribution of total revenue per user');


# #### Total revenue by device category and channel

# In[14]:


import seaborn as sns

data['transactionRevenue'] = data['transactionRevenue'].fillna(0)
data['transactionRevenue'] = np.log1p(data['transactionRevenue'])
sns.set(rc={'figure.figsize':(12, 8)})
data_r = data.loc[data['transactionRevenue'] > 0.0]
sns.boxplot(x="deviceCategory", y="transactionRevenue", hue="channelGrouping",  data=data_r)
plt.title("Total revenue by device category and channel.");
plt.xticks(rotation='vertical')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# #### Trends of transactions number by paying and non-paying users

# In[15]:


fig, ax1 = plt.subplots(figsize=(15, 6))
plt.title("Trends of transactions number by paying and non-paying users");
data.groupby(['date'])['transactionRevenue'].count().plot(color='brown')
ax1.set_ylabel('Transaction count', color='b')
plt.legend(['Non-paying users'])
ax2 = ax1.twinx()
data_r.groupby(['date'])['transactionRevenue'].count().plot(color='gold')
ax2.set_ylabel('Transaction count', color='g')
plt.legend(['Paying users'], loc=(0.875, 0.9))
plt.grid(False)


# #### Explore Device

# In[16]:


fig, ax = plt.subplots(2, 2, figsize = (16, 12))
print('Mean revenue per transaction')
sns.pointplot(x="browser", y="transactionRevenue", hue="isMobile", data=data_r, ax = ax[0, 0]);
sns.pointplot(x="deviceCategory", y="transactionRevenue", hue="isMobile", data=data_r, ax = ax[0, 1]);
sns.pointplot(x="operatingSystem", y="transactionRevenue", hue="isMobile", data=data_r, ax = ax[1, 0]);
sns.pointplot(x="isMobile", y="transactionRevenue", data=data_r, ax = ax[1, 1]);
ax[0, 0].xaxis.set_tick_params(rotation=30);
ax[0, 1].xaxis.set_tick_params(rotation=30);
ax[1, 0].xaxis.set_tick_params(rotation=30);
ax[1, 1].xaxis.set_tick_params(rotation=30);


# #### GeoNetwork

# In[17]:


plt.figure(figsize=(15,8))

# explore the browser used by users
sns.countplot(data[data['subContinent']                       .isin(data['subContinent']                             .value_counts()[:15].index.values)]['subContinent'], palette="hls")
plt.title("TOP 15 most frequent SubContinents", fontsize=15)
plt.xlabel("subContinent Names", fontsize=10)
plt.ylabel("SubContinent Count", fontsize=10)
plt.xticks(rotation=45)
plt.show()


# In[18]:


plt.figure(figsize=(15,8))

# explore the browser used by users
sns.countplot(data[data['country'].isin(data['country'].value_counts()[:20].index.values)]['country'],              palette="hls", order = data['country'].value_counts()[:20].index)
plt.title("TOP 20 most frequent Countries", fontsize=15)
plt.xlabel("Country Names", fontsize=10)
plt.ylabel("Country Count", fontsize=10)
plt.xticks(rotation=45)
plt.show()


# #### Operational System

# In[19]:


plt.figure(figsize=(14,7))

# explore the browser used by users
sns.countplot(data["operatingSystem"], palette="hls")
plt.title("Operational System used Count", fontsize=15)
plt.xlabel("Operational System Name", fontsize=10)
plt.ylabel("OS Count", fontsize=10)
plt.xticks(rotation=45)

plt.show()


# ### Start building model

# In[20]:


# consider the variable in x_val，expect "week" other variable value give the integer lable
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
for x in x_val[:-1]: #week is integer
    data[x+'_code']=label.fit_transform(data[x].values.astype(str))
x_val_code=[]
for x in x_val[:-1]:
    x_val_code.append(str(x+"_code"))

data_= pd.concat([data['week'],data[x_val_code]],axis=1) 
# change string to int，in order to do regression


# In[21]:


data_.head()


# In[22]:


X=data_
y=data["transactionRevenue"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# In[23]:


import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

gbm=lgb.LGBMRegressor(num_leaves=31,
                     learning_rate=0.1,
                     n_estimators=20)

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# feature importances
print('Feature importances:', list(gbm.feature_importances_))


# self-defined eval metric
# f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
# Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_true, y_pred):
    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False


print('Starting training with custom eval function...')
# train
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=rmsle,
        early_stopping_rounds=5)

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
print('The rmsle of prediction is:', rmsle(y_test, y_pred)[1])

# other scikit-learn modules
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [10, 20, 40], 
}

gbm = GridSearchCV(estimator, param_grid, cv=3)
gbm.fit(X_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)


# In[24]:


print('LightGBM results:',y_pred)


# In[25]:


#import lightgbm as lgb
# reference: this model code kaggle:
# will reference code in the report

lgb_params = {"objective" : "regression", 
              "metric" : "rmse",
              "num_leaves" : 500, 
              "learning_rate" : 0.1, 
              "bagging_fraction" : 0.5, 
              "feature_fraction" : 0.5,
              "bagging_frequency" : 1,
              "bagging_seed" : 1, 
              "lambda_l1": 3,
              'min_data_in_leaf': 50,
              'verbose':0,
              'min_child_samples':20
}
    
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val)
model = lgb.train(lgb_params, lgb_train, 10000, valid_sets=[lgb_train, lgb_val],
                  early_stopping_rounds=100, verbose_eval=100)


# In[26]:


#len(y_train)


# In[27]:


preds = pd.Series(model.predict(X_test, 
                                num_iteration=model.best_iteration))


# In[29]:


from sklearn.metrics import mean_squared_error

def rsme(y,pred):
    return(mean_squared_error(y,pred)**0.5)

acc=rsme(y_val,preds)
print('rmse:', acc)


# In[30]:


#print(preds)


# In[31]:


lgb.plot_importance(model, figsize=(12, 7))
plt.show()


# In[32]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

rf=RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred_test=rf.predict(X_test)
print('prediction:',rf_pred_test)
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,rf_pred_test)))


# In[33]:


rf_importances=rf.feature_importances_
indices=np.argsort(rf_importances)[::-1]
features = [X.columns[i] for i in indices]

plt.figure(figsize=(12,7))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), rf_importances[indices])

plt.xticks(range(X.shape[1]), features, rotation=90)
plt.show()


# In[34]:


from sklearn import preprocessing
data_standardized=preprocessing.scale(data_)
print("done")


# In[35]:


# reduce dimension of features
from sklearn.decomposition import PCA

pca=PCA(n_components=0.95) #n_components<=n_classes-1

pca_data=pca.fit_transform(data_standardized)
print (pca_data[1])


# In[36]:


from sklearn import preprocessing
data_standardized=preprocessing.scale(data_)
print("done")


# In[37]:


print (data_standardized)


# In[38]:


import torch
import torch.nn.functional as F  # implementation activation function
import matplotlib.pyplot as plt 
from torch.autograd import Variable

X=pca_data
y=data["transactionRevenue"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# In[39]:


#Convert data for pytorch training more easy
X_train=torch.from_numpy(np.array(X_train))
y_train=(torch.from_numpy(np.array(y_train))).view(59999,1)
X, y = Variable(X_train,requires_grad=True), Variable(y_train,requires_grad=True)


# In[40]:


# convert data type float
x=X.float()
y=y.float()


# In[41]:


import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as tf


# define model
class net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(net, self).__init__()
        #self.poly = nn.Linear(6,1)
#         self.l1 = torch.nn.Linear(8, 6)
#         self.l2 = torch.nn.Linear(6, 5)
#         self.l3 = torch.nn.Linear(5, 4)
#         self.l4 = torch.nn.Linear(4, 3)
#         self.l5 = torch.nn.Linear(3, 2) 
#         self.l6 = torch.nn.Linear(2, 1)
        self.hidden = torch.nn.Linear(8, 9)
        self.predict = torch.nn.Linear(9, 1)


    def forward(self, x):
        x = tf.relu(self.hidden(x))  #  function (linear value of hidden layer)
        x = self.predict(x)   # output
        return x
#         x = self.l1(x)
#         x = self.l2(x)
#         x = self.l3(x)
#         x = self.l4(x)
#         x = self.l5(x)
#         out=self.l6(x)
      
        #out = self.poly(x)
        #return out


# In[42]:



net = net(n_feature=8,n_hidden=9,n_output=1)

if __name__ == '__main__':
    #W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
    #b_target = torch.FloatTensor([0.9])

#     if torch.cuda.is_available():
#         model = poly_model().cuda()
#     else:
#         model = poly_model()

# define criterion and optimizer
    criterion = torch.nn.MSELoss(size_average=False)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
#x=torch.optim.SparseAdam
#epoch = 0
# after done define model, starting training   
for t in range(500):
    prediction = net(x)     # give model training dataset: x, output the predict value

    #output = model(x)
    loss = criterion(prediction, y)# calcuate the loss between prediction and y
    print_loss = loss.item()     

    optimizer.zero_grad()   #  clear out the last step the remaining value of parameters
    #loss.backward()         #  loss backward, calculate updated value
    loss.backward
    optimizer.step()
# output loss
print (loss)


# In[43]:


epochs = 2000
optimizer = torch.optim.SparseAdam(net.parameters(), lr=0.0001)
for epoch in range(epochs):

    epoch +=1
    #increase the number of epochs by 1 every time
    #inputs = Variable(torch.from_numpy(x_train))
    #labels = Variable(torch.from_numpy(y_correct))

    #clear grads as discussed in prev post
    optimizer.zero_grad()
    #forward to get predicted values
    outputs = net(x)
    loss = criterion(outputs, y)
    loss.backward()# back props
    #optimizer.step()# update the parameters
print('epoch {}, loss {}'.format(epoch,loss.data[0]))


# In[44]:


print ('MLP results:',prediction)


# In[45]:


print (y)


# In[46]:


print(torch.__version__)

