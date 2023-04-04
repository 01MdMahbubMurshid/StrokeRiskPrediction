#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


dataset=pd.read_csv("brain_stroke.csv")
dataset


# In[4]:


dataset.isnull().sum()


# In[5]:


dataset.info()


# In[7]:


dataset.hypertension.value_counts()


# In[8]:


dataset.gender.value_counts()


# In[9]:


dataset.heart_disease.value_counts()


# In[10]:


dataset.stroke.value_counts()


# In[ ]:





# In[8]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["gender"]=le.fit_transform(dataset["gender"])
dataset["ever_married"]=le.fit_transform(dataset["ever_married"])
dataset["work_type"]=le.fit_transform(dataset["work_type"])
dataset["Residence_type"]=le.fit_transform(dataset["Residence_type"])
dataset["smoking_status"]=le.fit_transform(dataset["smoking_status"])
dataset


# In[9]:


dataset["stroke"].value_counts()


# In[10]:


dataset.corr()


# In[11]:


x=dataset.drop(['stroke'],axis=1)
y = dataset['stroke']


# In[12]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(x,y)


# In[13]:


print(model.feature_importances_)


# In[14]:


ranked_features=pd.Series(model.feature_importances_,index=x.columns)
final = ranked_features.nlargest(9)


# In[15]:


final


# In[16]:


dataset=pd.read_csv("stroke9.csv")
dataset


# In[18]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["gender"]=le.fit_transform(dataset["gender"])
dataset["work_type"]=le.fit_transform(dataset["work_type"])
dataset["Residence_type"]=le.fit_transform(dataset["Residence_type"])
dataset["smoking_status"]=le.fit_transform(dataset["smoking_status"])
dataset


# In[19]:


dataset["stroke"].value_counts()


# In[33]:


dataset.corr()


# In[41]:


#Remove features which are not correlated with the response variable 
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
cor = dataset.corr()
sns.heatmap(cor, cmap='BrBG', center=0, annot=True, xticklabels=False)
plt.show()


# In[20]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=9)


# In[21]:


from sklearn.ensemble import RandomForestClassifier
#Step_2: Create the classifier
rf  = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
#Step_3: Fit the classiffer
rf.fit(x_train, y_train)
#Step_4: Predict the target class
y_pred_rf = rf.predict(x_test)
#Step_5: Calculate te score
from sklearn import metrics
rf_acc = metrics.accuracy_score(y_test, y_pred_rf)*100
print('Random Forest(RF): %.2f' %rf_acc)


# In[27]:


import xgboost as xgb
xgb_c = xgb.XGBClassifier(learning_rate=0.6,max_depth=5)
xgb_c.fit(x_train,y_train)
y_pred_x = xgb_c.predict(x_test)
score = metrics.accuracy_score(y_test, y_pred_x)*100
print("accuracy:   %0.2f" % score)


# In[23]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
ros.fit(x,y)


# In[24]:


from collections import Counter
from sklearn.datasets import make_classification

x_resampled,y_resampled = ros.fit_resample(x,y)
print('Resampled dataset shape {}'.format(Counter(y_resampled )))


# In[25]:



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_resampled,y_resampled,test_size=0.2,random_state=9)


# In[26]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import sklearn.metrics as metrics
import xgboost as xgb

xgb = xgb.XGBClassifier(learning_rate=0.6,max_depth=5)
rf = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
estimator_list = [
    #('knn',knn),
    #('svm_rbf',svm_rbf),
    ('xgb',xgb),
    ('rf',rf) ]
stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=LogisticRegression()
)
stack_model.fit(x_train,y_train)
y_pred_rf = stack_model.predict(x_test)
score = metrics.accuracy_score(y_pred_rf,y_test)*100
print("accuracy:   %0.2f" % score)


# In[28]:


y_pred_rf = stack_model.predict(x_test)
print(classification_report(y_pred_rf,y_test,target_names=['0','1']))

score = metrics.accuracy_score(y_pred_rf,y_test)*100
print("accuracy:   %0.2f" % score)


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(y_test, y_pred_rf)
cm_df = pd.DataFrame(cm,
                     index = ['0', '1'], 
                     columns = ['0', '1'])
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


# In[30]:


# predict probabilities
pred_prob1 = stack_model.predict_proba(x_test)


# In[31]:


from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)


# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[32]:


from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])

print(auc_score1)


# In[31]:


# matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='green', label='Stacking=1')

plt.plot(p_fpr, p_tpr, linestyle='--', color='red')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


# In[32]:


from sklearn.ensemble import RandomForestClassifier
#Step_2: Create the classifier
rf  = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
#Step_3: Fit the classiffer
rf.fit(x_train, y_train)
#Step_4: Predict the target class
y_pred_rf = rf.predict(x_test)
#Step_5: Calculate te score
from sklearn import metrics
rf_acc = metrics.accuracy_score(y_test, y_pred_rf)*100
print('Random Forest(RF): %.2f' %rf_acc)


# In[33]:


print(classification_report(y_pred_rf,y_test,target_names=['0','1']))

score = metrics.accuracy_score(y_pred_rf,y_test)*100
print("accuracy:   %0.2f" % score)


# In[34]:


# predict probabilities
pred_prob3 = rf.predict_proba(x_test)


# In[35]:


from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob3[:,1], pos_label=1)


# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[36]:


from sklearn.metrics import roc_auc_score

# auc scores
auc_score3 = roc_auc_score(y_test, pred_prob3[:,1])

print(auc_score3)


# In[37]:


import xgboost as xgb
xgb_c = xgb.XGBClassifier(learning_rate=0.6,max_depth=5)
xgb_c.fit(x_train,y_train)
y_pred_x = xgb_c.predict(x_test)
score = metrics.accuracy_score(y_test, y_pred_x)*100
print("accuracy:   %0.2f" % score)


# In[38]:


print(classification_report(y_pred_x,y_test,target_names=['0','1']))

score = metrics.accuracy_score(y_pred_x,y_test)*100
print("accuracy:   %0.2f" % score)


# In[45]:


sns.kdeplot(dataset["avg_glucose_level"])
sns.kdeplot(dataset["bmi"])


# In[ ]:





# In[15]:


sns.kdeplot(abc[7])
sns.kdeplot(abc[8])


# In[19]:


y_pred=nb.predict(x_test)
y_pred


# In[ ]:




